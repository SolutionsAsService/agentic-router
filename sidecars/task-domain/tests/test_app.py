import asyncio
import threading

import httpx
import pytest
from fastapi.testclient import TestClient

from app import create_app, warm_up
from contract import MAX_INPUT_BYTES, MAX_INPUT_TOKENS, PROJECTION, SCHEMA

RELEASE = "a" * 64
HEADERS = {"Authorization": "Bearer " + "s" * 32}
REQUEST = {"schema_version": SCHEMA, "projection_version": PROJECTION,
           "release_sha256": RELEASE, "user_text": "Review the deployment"}


class Predictor:
    def __init__(self, output="0,1,0,1,0", tokens=12):
        self.output = output
        self.tokens = tokens

    def predict(self, text):
        return self.output, self.tokens


def test_classification_and_auth():
    client = TestClient(create_app(Predictor(), RELEASE, "s" * 32))
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
    client = TestClient(create_app(Predictor(), RELEASE, "s" * 32))
    assert client.post("/classify", json=REQUEST | changes, headers=HEADERS).status_code == status


@pytest.mark.parametrize("output,tokens", [("0,1,0,1,0\n", 12), ("1,0", 12),
                                            ("0,1,0,1,0", 0), ("0,1,0,1,0", MAX_INPUT_TOKENS + 1)])
def test_bad_predictions(output, tokens):
    client = TestClient(create_app(Predictor(output, tokens), RELEASE, "s" * 32))
    assert client.post("/classify", json=REQUEST, headers=HEADERS).status_code == 503


def test_busy_inference_does_not_block_http_loop():
    started, finish = threading.Event(), threading.Event()

    class BlockingPredictor:
        def predict(self, text):
            started.set()
            assert finish.wait(3)
            return "0,1,0,1,0", 12

    async def scenario():
        transport = httpx.ASGITransport(app=create_app(BlockingPredictor(), RELEASE, "s" * 32))
        async with httpx.AsyncClient(transport=transport, base_url="https://classifier.test") as client:
            first = asyncio.create_task(client.post("/classify", json=REQUEST, headers=HEADERS))
            try:
                assert await asyncio.to_thread(started.wait, 2)
                busy = await asyncio.wait_for(client.post("/classify", json=REQUEST, headers=HEADERS), 1)
                assert busy.status_code == 503
                assert busy.json()["detail"] == "classifier busy"
            finally:
                finish.set()
            assert (await first).status_code == 200
            assert (await client.post("/classify", json=REQUEST, headers=HEADERS)).status_code == 200

    asyncio.run(scenario())


def test_warm_up_requires_valid_repeat_pass():
    class ColdPredictor:
        def __init__(self, repeat_output):
            self.texts = []
            self.repeat_output = repeat_output

        def predict(self, text):
            self.texts.append(text)
            return ("1" if len(self.texts) <= 2 else self.repeat_output), 12

    warmed = ColdPredictor("0,1,0,1,0")
    warm_up(warmed, ("short", "long"))
    assert warmed.texts == ["short", "long", "short", "long"]
    with pytest.raises(RuntimeError, match="invalid output for warmup input 0"):
        warm_up(ColdPredictor("1"), ("short", "long"))


def test_warm_up_propagates_inference_errors():
    class BrokenPredictor:
        def predict(self, text):
            raise RuntimeError("Failed to find C compiler")

    with pytest.raises(RuntimeError, match="C compiler"):
        warm_up(BrokenPredictor(), ("short",))
