"""Serve an operator-staged checkpoint; never download or publish model artifacts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

import torch
import uvicorn
from transformers import AutoTokenizer, Qwen3_5ForCausalLM

from app import create_app, warm_up
from contract import MAX_INPUT_TOKENS, SYSTEM_PROMPT, verify_release

WARMUP_TEXTS: Final = (
    "Fix the failing unit test in the payment service.",
    "Explain why this deployment configuration fails.\n" + "service: example\nreplicas: 3\n" * 600,
)


class QwenPredictor:
    def __init__(self, model_path: Path) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
        self.model = Qwen3_5ForCausalLM.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=False,
            use_safetensors=True, dtype=torch.bfloat16,
        ).to("cuda").eval()

    def predict(self, text: str) -> tuple[str, int]:
        prompt: str = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        tokens: list[int] = self.tokenizer.encode(prompt, add_special_tokens=False)
        if not 1 <= len(tokens) <= MAX_INPUT_TOKENS:
            raise ValueError("task input exceeds token budget")
        with torch.inference_mode():
            encoded = torch.tensor([tokens], device="cuda")
            generated = self.model.generate(
                input_ids=encoded, attention_mask=torch.ones_like(encoded),
                max_new_tokens=16, do_sample=False, max_time=2.5,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        output: str = self.tokenizer.decode(generated[0, len(tokens):], skip_special_tokens=True).strip()
        return output, len(tokens)


def main() -> None:
    model_path: Path = Path(os.environ["TASK_DOMAIN_MODEL_PATH"])
    release_sha256: str = verify_release(Path(os.environ["TASK_DOMAIN_RELEASE_PATH"]), model_path, os.environ["TASK_DOMAIN_RELEASE_SHA256"])
    predictor: QwenPredictor = QwenPredictor(model_path)
    app = create_app(predictor, release_sha256, os.environ["TASK_DOMAIN_BEARER"])
    warm_up(predictor, WARMUP_TEXTS)
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8095")), access_log=False)


if __name__ == "__main__":
    main()
