import asyncio
import json
import threading
import time

import httpx
import pytest
from fastapi.testclient import TestClient

import batching
from app import create_app, warm_up
from contract import BUDGET_HEADER, MAX_INPUT_BYTES, MAX_INPUT_TOKENS, PROJECTION, SCHEMA

RELEASE = "a" * 64
HEADERS = {"Authorization": "Bearer " + "s" * 32}
REQUEST = {"schema_version": SCHEMA, "projection_version": PROJECTION,
           "release_sha256": RELEASE, "user_text": "Review the deployment"}


class Predictor:
    def __init__(self, output="0,1,0,1,0", tokens=12):
        self.output = output
        self.tokens = tokens

    def encode(self, text):
        return [len(text)] * self.tokens

    def generate(self, batch, max_seconds):
        return [self.output] * len(batch)


class GatedPredictor(Predictor):
    """Holds the first generate call open so later requests queue behind it."""

    def __init__(self):
        super().__init__()
        self.started, self.finish = threading.Event(), threading.Event()
        self.batches = []

    def encode(self, text):
        return [0] * int(text)

    def generate(self, batch, max_seconds):
        self.batches.append([len(tokens) for tokens in batch])
        self.started.set()
        assert self.finish.wait(3)
        return [self.output] * len(batch)


class SubmissionCounter:
    def __init__(self):
        self.count = 0
        self.changed = threading.Condition()

    def record(self):
        with self.changed:
            self.count += 1
            self.changed.notify_all()

    def wait_for(self, count):
        with self.changed:
            return self.changed.wait_for(lambda: self.count >= count, 2)


@pytest.fixture
def submissions(monkeypatch):
    """Counts admission decisions so a test releases the GPU only once its requests are queued or rejected."""
    counter = SubmissionCounter()
    reserve, submit = batching.BatchScheduler.reserve, batching.BatchScheduler.submit

    def counting_reserve(self):
        reserved = reserve(self)
        if not reserved:
            counter.record()
        return reserved

    def counting_submit(self, pending):
        submit(self, pending)
        counter.record()

    monkeypatch.setattr(batching.BatchScheduler, "reserve", counting_reserve)
    monkeypatch.setattr(batching.BatchScheduler, "submit", counting_submit)
    return counter


def request_for(text):
    return REQUEST | {"user_text": text}


async def post_while_gpu_busy(predictor, submissions, first_text, queued, headers=HEADERS, hold_seconds=0.0):
    """Starts one request, waits until the GPU holds it, queues `queued`, then releases the GPU."""
    transport = httpx.ASGITransport(app=create_app(predictor, RELEASE, "s" * 32, batching.ServiceTimeModel()))
    async with httpx.AsyncClient(transport=transport, base_url="https://classifier.test") as client:
        first = asyncio.create_task(client.post("/classify", json=request_for(first_text), headers=HEADERS))
        try:
            assert await asyncio.to_thread(predictor.started.wait, 2)
            later = [asyncio.create_task(client.post("/classify", json=request_for(text), headers=headers))
                     for text in queued]
            assert await asyncio.to_thread(submissions.wait_for, 1 + len(queued))
            await asyncio.sleep(hold_seconds)
        finally:
            predictor.finish.set()
        return await first, await asyncio.gather(*later)


def test_classification_and_auth():
    client = TestClient(create_app(Predictor(), RELEASE, "s" * 32, batching.ServiceTimeModel()))
    assert client.post("/classify", json=REQUEST).status_code == 401
    response = client.post("/classify", json=REQUEST, headers=HEADERS)
    assert response.status_code == 200
    assert response.json() == {"schema_version": SCHEMA, "release_sha256": RELEASE,
                               "output": "0,1,0,1,0", "input_tokens": 12}


@pytest.mark.parametrize("changes,status", [
    ({"release_sha256": "b" * 64}, 409),
    ({"schema_version": "unknown"}, 400),
    ({"model": "request-selected-model"}, 400),
    ({"user_text": ""}, 413),
    ({"user_text": "x" * (MAX_INPUT_BYTES + 1)}, 413),
    ({"user_text": "界" * (MAX_INPUT_BYTES // 3 + 1)}, 413),
    ({"user_text": 42}, 400),
])
def test_bad_requests(changes, status):
    client = TestClient(create_app(Predictor(), RELEASE, "s" * 32, batching.ServiceTimeModel()))
    assert client.post("/classify", json=REQUEST | changes, headers=HEADERS).status_code == status


@pytest.mark.parametrize("budget", ["0", "-5", "1.5", "abc", "10001", b"\xb2", b"\xb9\xb2"])
def test_invalid_budget_header(budget):
    client = TestClient(create_app(Predictor(), RELEASE, "s" * 32, batching.ServiceTimeModel()))
    response = client.post("/classify", json=REQUEST, headers=HEADERS | {BUDGET_HEADER: budget})
    assert response.status_code == 400


@pytest.mark.parametrize("output,tokens", [("0,1,0,1,0\n", 12), ("1,0", 12),
                                            ("0,1,0,1,0", 0), ("0,1,0,1,0", MAX_INPUT_TOKENS + 1)])
def test_bad_predictions(output, tokens):
    client = TestClient(create_app(Predictor(output, tokens), RELEASE, "s" * 32, batching.ServiceTimeModel()))
    assert client.post("/classify", json=REQUEST, headers=HEADERS).status_code == 503


def test_token_limit_from_encoder_is_rejected():
    class LongPredictor(Predictor):
        def encode(self, text):
            raise ValueError("task input exceeds token budget")

    client = TestClient(create_app(LongPredictor(), RELEASE, "s" * 32, batching.ServiceTimeModel()))
    assert client.post("/classify", json=REQUEST, headers=HEADERS).status_code == 413


def test_requests_queued_behind_busy_gpu_run_as_one_batch(submissions):
    predictor = GatedPredictor()
    first, queued = asyncio.run(post_while_gpu_busy(predictor, submissions, "5", ["7", "9", "11"]))
    assert first.status_code == 200
    assert [response.status_code for response in queued] == [200, 200, 200]
    assert [sorted(batch) for batch in predictor.batches] == [[5], [7, 9, 11]]


def test_batch_respects_size_and_padded_token_budget(monkeypatch, submissions):
    monkeypatch.setattr(batching, "MAX_BATCH_SIZE", 2)
    monkeypatch.setattr(batching, "MAX_BATCH_TOKENS", 40)
    predictor = GatedPredictor()
    _, queued = asyncio.run(post_while_gpu_busy(predictor, submissions, "5", ["10", "10", "10", "30"]))
    assert [response.status_code for response in queued] == [200] * 4
    assert sorted(sorted(batch) for batch in predictor.batches) == [[5], [10], [10, 10], [30]]


def test_select_batch_never_starves_an_oversize_oldest_request(monkeypatch):
    monkeypatch.setattr(batching, "MAX_BATCH_TOKENS", 40)
    assert batching.select_batch([50, 5]) == [0]
    assert batching.select_batch([5, 50, 10]) == [0, 2]


def test_select_batch_admits_only_members_whose_deadline_the_batch_can_meet():
    def estimate(padded_tokens):
        return padded_tokens * 0.012

    assert batching.select_batch([10, 10, 10], [5.0, 0.3, 5.0], estimate) == [0, 1]
    assert batching.select_batch([10, 10, 10], [5.0, 5.0, 5.0], estimate) == [0, 1, 2]


def test_service_time_model_fits_seconds_against_padded_tokens():
    model = batching.ServiceTimeModel()
    assert model.estimate(4_000) == 0.0
    model.observe(100, 0.5)
    model.observe(8_100, 1.5)
    assert model.estimate(4_100) == pytest.approx(1.0)
    assert model.estimate(100) == pytest.approx(0.5)


def test_request_that_cannot_finish_before_its_budget_is_dropped_without_inference():
    model = batching.ServiceTimeModel()
    model.observe(12, 1.0)

    class CountingPredictor(Predictor):
        calls = 0

        def generate(self, batch, max_seconds):
            CountingPredictor.calls += 1
            return super().generate(batch, max_seconds)

    client = TestClient(create_app(CountingPredictor(), RELEASE, "s" * 32, model))
    response = client.post("/classify", json=REQUEST, headers=HEADERS | {BUDGET_HEADER: "500"})
    assert response.status_code == 503
    assert response.json()["detail"] == "classification deadline exceeded"
    assert CountingPredictor.calls == 0
    assert client.post("/classify", json=REQUEST, headers=HEADERS | {BUDGET_HEADER: "2000"}).status_code == 200


def test_full_queue_rejects_without_blocking_http_loop(monkeypatch, submissions):
    monkeypatch.setattr(batching, "MAX_QUEUED", 1)
    predictor = GatedPredictor()
    first, (queued, rejected) = asyncio.run(post_while_gpu_busy(predictor, submissions, "5", ["6", "7"]))
    assert first.status_code == 200
    statuses = sorted([(queued.status_code, queued.json().get("detail")),
                       (rejected.status_code, rejected.json().get("detail"))], key=lambda item: item[0])
    assert statuses == [(200, None), (503, "classifier busy")]


def test_expired_and_cancelled_work_does_not_hold_queue_capacity(monkeypatch):
    monkeypatch.setattr(batching, "MAX_QUEUED", 2)
    predictor = GatedPredictor()

    async def scenario():
        loop = asyncio.get_running_loop()
        scheduler = batching.BatchScheduler(predictor, batching.ServiceTimeModel())
        assert scheduler.reserve()
        scheduler.submit(batching.PendingClassification([0] * 5, time.monotonic() + 60, loop.create_future(), loop))
        assert await asyncio.to_thread(predictor.started.wait, 2)
        expired = batching.PendingClassification([0] * 6, time.monotonic() + 0.3, loop.create_future(), loop)
        cancelled = batching.PendingClassification([0] * 7, time.monotonic() + 60, loop.create_future(), loop)
        for pending in (expired, cancelled):
            assert scheduler.reserve()
            scheduler.submit(pending)
        assert not scheduler.reserve()
        cancelled.future.cancel()
        await asyncio.sleep(0.35)
        assert scheduler.reserve()
        assert scheduler.reserve()
        predictor.finish.set()
        with pytest.raises(batching.DeadlineExceeded):
            await expired.future

    asyncio.run(scenario())


def test_rejected_request_is_not_tokenized(monkeypatch):
    monkeypatch.setattr(batching, "MAX_QUEUED", 0)

    class CountingEncoder(Predictor):
        encoded = 0

        def encode(self, text):
            CountingEncoder.encoded += 1
            return super().encode(text)

    client = TestClient(create_app(CountingEncoder(), RELEASE, "s" * 32, batching.ServiceTimeModel()))
    response = client.post("/classify", json=REQUEST, headers=HEADERS)
    assert response.status_code == 503
    assert response.json()["detail"] == "classifier busy"
    assert CountingEncoder.encoded == 0


def test_queued_request_past_its_budget_is_dropped_before_inference(submissions):
    predictor = GatedPredictor()
    _, (expired,) = asyncio.run(post_while_gpu_busy(predictor, submissions, "5", ["8"], headers=HEADERS | {BUDGET_HEADER: "100"}, hold_seconds=0.2))
    assert expired.status_code == 503
    assert expired.json()["detail"] == "classification deadline exceeded"
    assert predictor.batches == [[5]]


def test_cancelled_request_is_skipped_by_the_gpu_worker():
    predictor = GatedPredictor()

    async def scenario():
        loop = asyncio.get_running_loop()
        scheduler = batching.BatchScheduler(predictor, batching.ServiceTimeModel())
        first = batching.PendingClassification([0] * 5, loop.time() + 60, loop.create_future(), loop)
        abandoned = batching.PendingClassification([0] * 9, loop.time() + 60, loop.create_future(), loop)
        kept = batching.PendingClassification([0] * 7, loop.time() + 60, loop.create_future(), loop)
        for pending in (first, abandoned, kept):
            assert scheduler.reserve()
        scheduler.submit(first)
        assert await asyncio.to_thread(predictor.started.wait, 2)
        scheduler.submit(abandoned)
        scheduler.submit(kept)
        abandoned.future.cancel()
        predictor.finish.set()
        assert await first.future == "0,1,0,1,0"
        assert await kept.future == "0,1,0,1,0"

    asyncio.run(scenario())
    assert predictor.batches == [[5], [7]]


def test_disconnected_caller_is_skipped_by_the_gpu_worker(submissions):
    predictor = GatedPredictor()
    app = create_app(predictor, RELEASE, "s" * 32, batching.ServiceTimeModel())

    async def scenario():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="https://classifier.test") as client:
            first = asyncio.create_task(client.post("/classify", json=request_for("5"), headers=HEADERS))
            assert await asyncio.to_thread(predictor.started.wait, 2)
            messages = [{"type": "http.request", "body": json.dumps(request_for("9")).encode(), "more_body": False}]
            hung_up = asyncio.Event()

            async def receive():
                if messages:
                    return messages.pop(0)
                await hung_up.wait()
                return {"type": "http.disconnect"}

            sent = []

            async def send(message):
                sent.append(message)

            headers = [(b"authorization", HEADERS["Authorization"].encode()), (b"content-type", b"application/json")]
            scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST",
                     "scheme": "https", "path": "/classify", "raw_path": b"/classify", "query_string": b"",
                     "root_path": "", "headers": headers, "client": ("caller", 1), "server": ("classifier.test", 443)}
            abandoned = asyncio.create_task(app(scope, receive, send))
            assert await asyncio.to_thread(submissions.wait_for, 2)
            hung_up.set()
            await abandoned
            predictor.finish.set()
            assert (await first).status_code == 200
            return sent

    sent = asyncio.run(scenario())
    assert sent[0]["status"] == 503
    assert predictor.batches == [[5]]


def test_worker_survives_a_failure_while_selecting_a_batch():
    class FlakyServiceTime(batching.ServiceTimeModel):
        failed = False

        def estimate(self, padded_tokens):
            if not FlakyServiceTime.failed:
                FlakyServiceTime.failed = True
                raise RuntimeError("transient selection failure")
            return super().estimate(padded_tokens)

    client = TestClient(create_app(Predictor(), RELEASE, "s" * 32, FlakyServiceTime()))
    assert client.post("/classify", json=REQUEST, headers=HEADERS).status_code == 200
    assert client.post("/classify", json=REQUEST, headers=HEADERS).status_code == 200
    assert FlakyServiceTime.failed


def test_truncated_batches_do_not_train_the_service_time_model():
    service_time = batching.ServiceTimeModel()
    truncated = TestClient(create_app(Predictor(output="1"), RELEASE, "s" * 32, service_time))
    assert truncated.post("/classify", json=REQUEST, headers=HEADERS).status_code == 503
    assert service_time.estimate(1) == 0.0
    complete = TestClient(create_app(Predictor(), RELEASE, "s" * 32, service_time))
    assert complete.post("/classify", json=REQUEST, headers=HEADERS).status_code == 200
    assert service_time.estimate(1) > 0


def test_generate_failure_fails_every_request_in_the_batch():
    class FailingPredictor(Predictor):
        def generate(self, batch, max_seconds):
            raise RuntimeError("CUDA error")

    client = TestClient(create_app(FailingPredictor(), RELEASE, "s" * 32, batching.ServiceTimeModel()))
    response = client.post("/classify", json=REQUEST, headers=HEADERS)
    assert response.status_code == 503
    assert response.json()["detail"] == "classifier unavailable"


def test_warm_up_requires_valid_repeat_pass():
    class ColdPredictor(Predictor):
        def __init__(self, repeat_output):
            super().__init__()
            self.batches = []
            self.repeat_output = repeat_output

        def encode(self, text):
            return [ord(text[0])]

        def generate(self, batch, max_seconds):
            self.batches.append([tokens[0] for tokens in batch])
            return [("1" if len(self.batches) <= 3 else self.repeat_output)] * len(batch)

    warmed, service_time = ColdPredictor("0,1,0,1,0"), batching.ServiceTimeModel()
    warm_up(warmed, ("short", "long"), service_time)
    short, long = ord("s"), ord("l")
    assert warmed.batches == [[short], [long], [short, long]] * 2
    assert service_time.estimate(1) > 0, "the verification pass seeds the service-time model"
    with pytest.raises(RuntimeError, match="invalid output for warmup input 0"):
        warm_up(ColdPredictor("1"), ("short", "long"), batching.ServiceTimeModel())


def test_warm_up_requires_one_output_per_input():
    class ShortPredictor(Predictor):
        def generate(self, batch, max_seconds):
            return ["0,1,0,1,0"]

    with pytest.raises(RuntimeError, match="output count mismatch"):
        warm_up(ShortPredictor(), ("short", "long"), batching.ServiceTimeModel())


def test_warm_up_propagates_inference_errors():
    class BrokenPredictor(Predictor):
        def generate(self, batch, max_seconds):
            raise RuntimeError("Failed to find C compiler")

    with pytest.raises(RuntimeError, match="C compiler"):
        warm_up(BrokenPredictor(), ("short",), batching.ServiceTimeModel())
