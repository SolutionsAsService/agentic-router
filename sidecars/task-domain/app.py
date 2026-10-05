"""Bounded authenticated classification; no prompt logging or model selection."""

from __future__ import annotations

import hmac
import json
import threading
from typing import Literal, Protocol

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, ValidationError
from starlette.concurrency import run_in_threadpool

from contract import MAX_INPUT_BYTES, MAX_INPUT_TOKENS, OUTPUT, PROJECTION, SCHEMA


class Predictor(Protocol):
    def predict(self, text: str) -> tuple[str, int]: ...


class ClassificationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: Literal["task_domain_classifier_v1"]
    release_sha256: str
    projection_version: Literal["initial_logical_user_turn_v1"]
    user_text: str


def create_app(predictor: Predictor, release_sha256: str, bearer: str) -> FastAPI:
    if len(bearer) < 32 or "\n" in bearer or "\r" in bearer:
        raise ValueError("task classifier requires a strong bearer secret")
    app: FastAPI = FastAPI()
    capacity: threading.Lock = threading.Lock()

    def predict(text: str) -> tuple[str, int]:
        # Release in the worker even if the caller disconnects during CUDA work.
        try:
            return predictor.predict(text)
        finally:
            capacity.release()

    @app.post("/classify")
    async def classify(request: Request) -> dict[str, str | int]:
        if not hmac.compare_digest(request.headers.get("authorization", "").encode(), ("Bearer " + bearer).encode()):
            raise HTTPException(401, "unauthorized")
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
        if not capacity.acquire(blocking=False):
            raise HTTPException(503, "classifier busy")
        try:
            output, tokens = await run_in_threadpool(predict, classification.user_text)
        except ValueError:
            raise HTTPException(413, "input token limit exceeded") from None
        except Exception:
            raise HTTPException(503, "classifier unavailable") from None
        if not OUTPUT.fullmatch(output) or not 1 <= tokens <= MAX_INPUT_TOKENS:
            raise HTTPException(503, "invalid classifier output")
        return {"schema_version": SCHEMA, "release_sha256": release_sha256, "output": output, "input_tokens": tokens}

    return app
