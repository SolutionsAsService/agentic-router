"""Only this module's worker thread touches the GPU; request handlers enqueue and await."""

from __future__ import annotations

import asyncio
import threading
import math
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final, Protocol

from contract import MAX_INPUT_TOKENS, OUTPUT

MAX_BATCH_SIZE: Final = 8
# Padded tokens (longest input x batch size) per generate call. One maximum-size input
# already takes ~1.4s on an L4, so larger batches would overrun the generation bound.
MAX_BATCH_TOKENS: Final = MAX_INPUT_TOKENS
MAX_QUEUED: Final = 32
MAX_GENERATION_SECONDS: Final = 2.5
SERVICE_TIME_WINDOW: Final = 64


class DeadlineExceeded(Exception):
    pass


class BatchPredictor(Protocol):
    def generate(self, batch: Sequence[Sequence[int]], max_seconds: float) -> list[str]: ...


@dataclass(eq=False)
class PendingClassification:
    tokens: Sequence[int]
    deadline: float
    future: asyncio.Future[str]
    loop: asyncio.AbstractEventLoop


class ServiceTimeModel:
    """Least-squares fit of generate seconds against padded tokens over recent batches on this GPU."""

    def __init__(self) -> None:
        self._recent: deque[tuple[int, float]] = deque(maxlen=SERVICE_TIME_WINDOW)

    def observe(self, padded_tokens: int, seconds: float) -> None:
        self._recent.append((padded_tokens, seconds))

    def estimate(self, padded_tokens: int) -> float:
        if not self._recent:
            return 0.0
        mean_tokens: float = sum(tokens for tokens, _ in self._recent) / len(self._recent)
        mean_seconds: float = sum(seconds for _, seconds in self._recent) / len(self._recent)
        spread: float = sum((tokens - mean_tokens) ** 2 for tokens, _ in self._recent)
        covariance: float = sum((tokens - mean_tokens) * (seconds - mean_seconds) for tokens, seconds in self._recent)
        per_token: float = max(0.0, covariance / spread) if spread else 0.0
        return max(0.0, mean_seconds + per_token * (padded_tokens - mean_tokens))


def select_batch(
    widths: Sequence[int],
    slack_seconds: Sequence[float] | None = None,
    estimate: Callable[[int], float] = lambda padded_tokens: 0.0,
) -> list[int]:
    """Oldest request first, then later ones while the padded batch fits its budget and every member's deadline."""
    chosen: list[int] = []
    longest: int = 0
    tightest: float = math.inf
    for index, width in enumerate(widths):
        padded: int = max(longest, width)
        slack: float = min(tightest, slack_seconds[index] if slack_seconds else math.inf)
        fits: bool = len(chosen) < MAX_BATCH_SIZE and padded * (len(chosen) + 1) <= MAX_BATCH_TOKENS
        if not chosen or (fits and estimate(padded * (len(chosen) + 1)) <= slack):
            chosen.append(index)
            longest, tightest = padded, slack
    return chosen


def _settle(pending: PendingClassification, output: str | None, error: BaseException | None) -> None:
    def settle() -> None:
        if pending.future.done():
            return
        if error is None:
            pending.future.set_result(output)
        else:
            pending.future.set_exception(error)

    try:
        pending.loop.call_soon_threadsafe(settle)
    except RuntimeError:
        # The request's event loop has closed (shutdown); nobody is awaiting this result.
        return


class BatchScheduler:
    def __init__(self, predictor: BatchPredictor, service_time: ServiceTimeModel) -> None:
        self._predictor = predictor
        self._service_time = service_time
        self._queue: list[PendingClassification] = []
        self._reserved: int = 0
        self._ready = threading.Condition()
        threading.Thread(target=self._run, name="task-domain-gpu", daemon=True).start()

    def reserve(self) -> bool:
        """Claims a queue slot before tokenization, so rejected requests cost no CPU."""
        with self._ready:
            now: float = time.monotonic()
            for pending in self._queue:
                if not pending.future.done() and pending.deadline <= now:
                    _settle(pending, None, DeadlineExceeded())
            # Abandoned or expired work must not hold capacity a live request needs.
            self._queue = [p for p in self._queue if not p.future.done() and p.deadline > now]
            if len(self._queue) + self._reserved >= MAX_QUEUED:
                return False
            self._reserved += 1
            return True

    def release(self) -> None:
        with self._ready:
            self._reserved -= 1

    def submit(self, pending: PendingClassification) -> None:
        with self._ready:
            self._reserved -= 1
            self._queue.append(pending)
            self._ready.notify()

    def _take_batch(self) -> list[PendingClassification]:
        with self._ready:
            while not self._queue:
                self._ready.wait()
            now: float = time.monotonic()
            live: list[PendingClassification] = []
            for pending in self._queue:
                if pending.future.done():
                    continue
                # Work that cannot finish even alone before its caller gives up would only be discarded.
                if pending.deadline - now < self._service_time.estimate(len(pending.tokens)):
                    _settle(pending, None, DeadlineExceeded())
                    continue
                live.append(pending)
            chosen: list[int] = select_batch(
                [len(p.tokens) for p in live], [p.deadline - now for p in live], self._service_time.estimate,
            )
            batch: list[PendingClassification] = [live[i] for i in chosen]
            self._queue = [pending for pending in live if pending not in batch]
            return batch

    def _run(self) -> None:
        while True:
            batch: list[PendingClassification] = []
            try:
                batch = self._take_batch()
                self._process(batch)
            except Exception as error:
                # One failed batch must not stop the only GPU worker.
                for pending in batch:
                    _settle(pending, None, error)

    def _process(self, batch: list[PendingClassification]) -> None:
        if not batch:
            return
        started: float = time.monotonic()
        max_seconds: float = min(MAX_GENERATION_SECONDS, max(p.deadline for p in batch) - started)
        if max_seconds <= 0:
            for pending in batch:
                _settle(pending, None, DeadlineExceeded())
            return
        outputs: list[str] = self._predictor.generate([p.tokens for p in batch], max_seconds)
        if len(outputs) != len(batch):
            raise RuntimeError("batch output count mismatch")
        # A batch cut off by its time bound understates the real cost, so only complete batches train the estimate.
        if all(OUTPUT.fullmatch(output) for output in outputs):
            self._service_time.observe(max(len(p.tokens) for p in batch) * len(batch), time.monotonic() - started)
        for pending, output in zip(batch, outputs):
            _settle(pending, output, None)
