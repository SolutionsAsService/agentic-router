"""Serve an operator-staged checkpoint; never download or publish model artifacts."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import torch
import uvicorn
from transformers import AutoTokenizer, Qwen3_5ForCausalLM

from app import create_app, warm_up
from batching import ServiceTimeModel
from contract import MAX_INPUT_TOKENS, SYSTEM_PROMPT, verify_release

WARMUP_TEXTS: Final = (
    "Fix the failing unit test in the payment service.",
    "Add a dark mode toggle to the settings page and persist the choice.",
    "Write a migration that backfills the region column from the address table.",
    "Explain why this deployment configuration fails.\n" + "service: example\nreplicas: 3\n" * 600,
)


class QwenPredictor:
    def __init__(self, model_path: Path) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
        self.model = Qwen3_5ForCausalLM.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=False,
            use_safetensors=True, dtype=torch.bfloat16,
        ).to("cuda").eval()

    def encode(self, text: str) -> list[int]:
        prompt: str = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        tokens: list[int] = self.tokenizer.encode(prompt, add_special_tokens=False)
        if not 1 <= len(tokens) <= MAX_INPUT_TOKENS:
            raise ValueError("task input exceeds token budget")
        return tokens

    def generate(self, batch: Sequence[Sequence[int]], max_seconds: float) -> list[str]:
        # Left padding keeps every prompt's final token adjacent to its generated tokens.
        width: int = max(len(tokens) for tokens in batch)
        pad: int = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        with torch.inference_mode():
            encoded = torch.tensor([[pad] * (width - len(tokens)) + list(tokens) for tokens in batch], device="cuda")
            attention_mask = torch.tensor([[0] * (width - len(tokens)) + [1] * len(tokens) for tokens in batch], device="cuda")
            generated = self.model.generate(
                input_ids=encoded, attention_mask=attention_mask,
                max_new_tokens=16, do_sample=False, max_time=max_seconds,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return [self.tokenizer.decode(row[width:], skip_special_tokens=True).strip() for row in generated]


def main() -> None:
    model_path: Path = Path(os.environ["TASK_DOMAIN_MODEL_PATH"])
    release_sha256: str = verify_release(Path(os.environ["TASK_DOMAIN_RELEASE_PATH"]), model_path, os.environ["TASK_DOMAIN_RELEASE_SHA256"])
    predictor: QwenPredictor = QwenPredictor(model_path)
    service_time: ServiceTimeModel = ServiceTimeModel()
    app = create_app(predictor, release_sha256, os.environ["TASK_DOMAIN_BEARER"], service_time)
    warm_up(predictor, WARMUP_TEXTS, service_time)
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8095")), access_log=False)


if __name__ == "__main__":
    main()
