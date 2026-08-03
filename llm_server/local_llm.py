"""The model, loaded on this box's GPU. The only file in the repo that imports torch.

Deliberately dumb: it takes a (system, user) pair and returns the completion.
No prompts, no JSON schema, no room directory — all of that lives on the client
side (`patrol/patrol_intent.py`, `patrol_report.py`) and arrives over the
wire. That is what lets
`llm_server/` stay independent of `patrol/`: copy this folder to any GPU box and
it runs.

    from llm_server.local_llm import LocalLLM
    llm = LocalLLM(device="cuda:1")
    llm.generate("Reply with JSON only.", "거실 소파 찾아줘")

The default model is `Qwen/Qwen2.5-3B-Instruct` — fits one RTX 3080 in fp16.
For larger models pass `device_map="auto"` to shard across GPUs via accelerate.
"""
from __future__ import annotations

from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"


class LocalLLM:
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        device: str = "cuda:0",
        dtype: str = "float16",
        device_map: Optional[str] = None,
    ):
        self.model_id = model_id
        print(f"[llm] loading {model_id} (device={device}, dtype={dtype}, "
              f"device_map={device_map}) ...")
        torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                       "float32": torch.float32}[dtype]
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        kwargs = {"torch_dtype": torch_dtype}
        if device_map:
            kwargs["device_map"] = device_map
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        if not device_map:
            self.model.to(device)
        self.model.eval()
        self.device = next(self.model.parameters()).device

    @torch.no_grad()
    def generate(self, system: str, user: str, max_new_tokens: int = 256) -> str:
        """One system+user turn, greedy. Returns only the newly generated text.

        Greedy (`do_sample=False`) on purpose: intent parsing has to be
        reproducible, and the client sends temperature 0 to match.
        """
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user.strip()},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(
            out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True
        )
