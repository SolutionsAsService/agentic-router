"""Serve an operator-staged checkpoint; never download or publish model artifacts."""

from __future__ import annotations

import os
from pathlib import Path

import torch
import uvicorn
from transformers import AutoTokenizer, Qwen3_5ForCausalLM

from app import create_app
from contract import MAX_INPUT_TOKENS, SYSTEM_PROMPT, verify_release


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
    app = create_app(QwenPredictor(model_path), release_sha256, os.environ["TASK_DOMAIN_BEARER"])
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8095")), access_log=False)


if __name__ == "__main__":
    main()
