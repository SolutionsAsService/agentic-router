"""Bounded authenticated classification; no prompt logging or model selection."""

from __future__ import annotations

import asyncio
import hmac
import json
import time
from collections.abc import Sequence
from typing import Literal, Protocol

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, ValidationError
from starlette.concurrency import run_in_threadpool

from batching import (
    MAX_GENERATION_SECONDS,
    BatchScheduler,
    DeadlineExceeded,
    PendingClassification,
    ServiceTimeModel,
    select_batch,
)
from contract import BUDGET_HEADER, MAX_BUDGET_MILLISECONDS, MAX_INPUT_BYTES, MAX_INPUT_TOKENS, OUTPUT, PROJECTION, SCHEMA

# Callers that predate the budget header get the router's three-second budget less its commit reserve.
DEFAULT_BUDGET_SECONDS = 2.9


class Predictor(Protocol):
    def encode(self, text: str) -> list[int]: ...

    def generate(self, batch: Sequence[Sequence[int]], max_seconds: float) -> list[str]: ...


class ClassificationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: Literal["task_domain_classifier_v1"]
    release_sha256: str
    projection_version: Literal["initial_logical_user_turn_v1"]
    user_text: str


def warm_up(predictor: Predictor, texts: Sequence[str], service_time: ServiceTimeModel) -> None:
    """Primes CUDA, then seeds the service-time model from a verification pass that must be valid."""
    encoded: list[list[int]] = [predictor.encode(text) for text in texts]
    batches: list[list[list[int]]] = [[tokens] for tokens in encoded]
    batches.append([encoded[i] for i in select_batch([len(tokens) for tokens in encoded])])

    def run_pass(observe: bool) -> list[str]:
        outputs: list[str] = []
        for batch in batches:
            started: float = time.monotonic()
            batch_outputs: list[str] = predictor.generate(batch, MAX_GENERATION_SECONDS)
            if len(batch_outputs) != len(batch):
                raise RuntimeError("task classifier warmup output count mismatch")
            outputs.extend(batch_outputs)
            if observe:
                service_time.observe(max(len(tokens) for tokens in batch) * len(batch), time.monotonic() - started)
        return outputs

    # The first CUDA pass is slow enough to truncate generation, so only the repeat pass is checked and timed.
    run_pass(observe=False)
    for index, output in enumerate(run_pass(observe=True)):
        if not OUTPUT.fullmatch(output):
            raise RuntimeError(f"task classifier warmup produced invalid output for warmup input {index}")


def request_deadline(budget_header: str | None) -> float:
    if budget_header is None:
        return time.monotonic() + DEFAULT_BUDGET_SECONDS
    if not (budget_header.isascii() and budget_header.isdecimal()) or not 1 <= int(budget_header) <= MAX_BUDGET_MILLISECONDS:
        raise HTTPException(400, "invalid classification budget")
    return time.monotonic() + int(budget_header) / 1000


async def cancel_on_disconnect(request: Request, future: asyncio.Future[str]) -> None:
    """Cancels queued work once the caller hangs up, so the GPU worker skips it."""
    while (await request.receive())["type"] != "http.disconnect":
        pass
    future.cancel()


def create_app(predictor: Predictor, release_sha256: str, bearer: str, service_time: ServiceTimeModel) -> FastAPI:
    if len(bearer) < 32 or "\n" in bearer or "\r" in bearer:
        raise ValueError("task classifier requires a strong bearer secret")
    app: FastAPI = FastAPI()
    scheduler: BatchScheduler = BatchScheduler(predictor, service_time)

    @app.post("/classify")
    async def classify(request: Request) -> dict[str, str | int]:
        if not hmac.compare_digest(request.headers.get("authorization", "").encode(), ("Bearer " + bearer).encode()):
            raise HTTPException(401, "unauthorized")
        deadline: float = request_deadline(request.headers.get(BUDGET_HEADER))
        body: bytearray = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > MAX_INPUT_BYTES * 6 + 1024:
                raise HTTPException(413, "input too large")
        try:
            classification: ClassificationRequest = ClassificationRequest.model_validate(json.loads(body))
        except (ValueError, ValidationError):
            raise HTTPException(400, "invalid classification request") from None
        if classification.release_sha256 != release_sha256 or classification.projection_version != PROJECTION:
            raise HTTPException(409, "release mismatch")
        if not classification.user_text or len(classification.user_text.encode()) > MAX_INPUT_BYTES:
            raise HTTPException(413, "input too large")
        if not scheduler.reserve():
            raise HTTPException(503, "classifier busy")
        try:
            tokens: list[int] = await run_in_threadpool(predictor.encode, classification.user_text)
        except ValueError:
            scheduler.release()
            raise HTTPException(413, "input token limit exceeded") from None
        except BaseException:
            scheduler.release()
            raise
        loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
        pending: PendingClassification = PendingClassification(tokens, deadline, loop.create_future(), loop)
        scheduler.submit(pending)
        disconnect_watcher: asyncio.Task[None] = asyncio.create_task(cancel_on_disconnect(request, pending.future))
        try:
            output: str = await pending.future
        except asyncio.CancelledError:
            if not disconnect_watcher.done():
                raise
            raise HTTPException(503, "caller disconnected") from None
        except DeadlineExceeded:
            raise HTTPException(503, "classification deadline exceeded") from None
        except Exception:
            raise HTTPException(503, "classifier unavailable") from None
        finally:
            disconnect_watcher.cancel()
        if not OUTPUT.fullmatch(output) or not 1 <= len(tokens) <= MAX_INPUT_TOKENS:
            raise HTTPException(503, "invalid classifier output")
        return {"schema_version": SCHEMA, "release_sha256": release_sha256, "output": output, "input_tokens": len(tokens)}

    return app
