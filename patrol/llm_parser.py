"""In-process LLM intent parser — loads the model on this machine's GPU.

Takes a Korean/English natural-language drone command and returns a structured
`ParsedIntent` suitable for downstream detection matching + path planning:

    {
        "target_object": "toilet",          # English CLIP prompt phrase
        "clip_prompt":   "a toilet",        # ready-to-feed CLIP query
        "location_hint": "upstairs room",   # free-form, may be empty
        "action":        "take_photo",
        "return_home":   true,
        "raw":           {... full LLM JSON ...}
    }

The default model is `Qwen/Qwen2.5-3B-Instruct` — small enough to fit on one
RTX 3080 (8 GB) in fp16. Override with --llm-model. For larger models, set
--llm-device-map auto to shard across GPUs via accelerate.

This module is the ONLY one in `patrol/` that imports torch. The prompt, the
schema and the JSON coercion live in `patrol/llm_base.py` so that a machine
without torch can drive the same pipeline through `patrol/remote_llm.py`.
Import it lazily (see `server.py`) — a top-level import here would drag torch
into processes that only need the HTTP client.
"""
from __future__ import annotations

from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from patrol.llm_base import (SYSTEM_PROMPT, BaseLLMParser,  # noqa: F401
                             ParsedIntent, _coerce_room_id, _extract_json)


class LocalLLMParser(BaseLLMParser):
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-3B-Instruct",
        device: str = "cuda:0",
        dtype: str = "float16",
        device_map: Optional[str] = None,
    ):
        self.model_id = model_id
        print(f"[llm] loading {model_id} (device={device}, dtype={dtype}, device_map={device_map}) ...")
        torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
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
        """Raw single-turn completion. Shared by `parse`, the patrol intent
        parser (patrol/patrol_intent.py) and the patrol report writer
        (patrol/patrol_report.py) so they all reuse ONE loaded model."""
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
            temperature=1.0,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(
            out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True
        )
