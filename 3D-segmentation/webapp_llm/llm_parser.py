"""Local LLM intent parser.

Takes a Korean/English natural-language drone command and returns a structured
dict suitable for downstream 3D segmentation + path planning:

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
--llm-tp >1 to shard across GPUs via accelerate device_map="auto".
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM_PROMPT = """You parse natural-language drone commands into JSON.

The drone operates in a pre-scanned 3D indoor scene made of numbered rooms. Extract:
- target_object: the object to find, as a short English noun phrase (e.g. "toilet", "tv", "refrigerator", "sofa").
- clip_prompt:   an English CLIP-friendly prompt, usually "a <target_object>".
- location_hint: free-form location/region context from the user (e.g. "upstairs bathroom", "next room", "kitchen"). Empty string if none.
- target_room:   the ROOM NUMBER to search, as an integer, when the user names a specific room (e.g. "3번 방" -> 3, "room 5" -> 5). null if no specific room number is given.
- scope:         "room" if the user wants a specific room (target_room set), "building" if the user wants to search the WHOLE house/building (e.g. "집 전체", "온 집", "all rooms", "whole house"), or "" if unspecified.
- action:        one of ["take_photo", "inspect", "goto", "other"].
- return_home:   true if the user asks the drone to come back, else false.

Always answer with a single JSON object, no prose, no markdown fences.
Translate Korean object names to English. Keep it terse.

Examples:

User: 3번 방에서 의자 찾아줘
{"target_object":"chair","clip_prompt":"a chair","location_hint":"room 3","target_room":3,"scope":"room","action":"goto","return_home":false}

User: 5번 방에 있는 tv 사진 찍어와줘
{"target_object":"tv","clip_prompt":"a tv","location_hint":"room 5","target_room":5,"scope":"room","action":"take_photo","return_home":true}

User: 집 전체에서 냉장고 찾아줘
{"target_object":"refrigerator","clip_prompt":"a refrigerator","location_hint":"whole house","target_room":null,"scope":"building","action":"goto","return_home":false}

User: 옆 방에 있는 TV 사진 찍어와줘
{"target_object":"tv","clip_prompt":"a tv","location_hint":"next room","target_room":null,"scope":"","action":"take_photo","return_home":true}

User: 거실 소파 위에 누가 있는지 확인해줘
{"target_object":"sofa","clip_prompt":"a sofa","location_hint":"living room","target_room":null,"scope":"","action":"inspect","return_home":false}
"""


@dataclass
class ParsedIntent:
    target_object: str
    clip_prompt: str
    location_hint: str
    action: str
    return_home: bool
    raw: dict
    target_room: Optional[int] = None   # 1-based room number, or None
    scope: str = ""                     # "room" | "building" | ""
    raw_text: str = ""  # full LLM generation before JSON parsing


class LocalLLMParser:
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
    def parse(self, user_text: str, max_new_tokens: int = 200) -> ParsedIntent:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text.strip()},
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
        gen = self.tokenizer.decode(
            out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True
        )
        data = _extract_json(gen)

        target = str(data.get("target_object", "")).strip().lower() or "object"
        clip_prompt = str(data.get("clip_prompt", "")).strip() or f"a {target}"
        return ParsedIntent(
            target_object=target,
            clip_prompt=clip_prompt,
            location_hint=str(data.get("location_hint", "")).strip(),
            action=str(data.get("action", "goto")).strip(),
            return_home=bool(data.get("return_home", False)),
            target_room=_coerce_room(data.get("target_room")),
            scope=str(data.get("scope", "")).strip().lower(),
            raw=data,
            raw_text=gen,
        )


def _coerce_room(v) -> Optional[int]:
    """Best-effort int room number from the LLM field (handles 3, '3', '3번')."""
    if v is None:
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return int(v)
    m = re.search(r"\d+", str(v))
    return int(m.group(0)) if m else None


def _extract_json(text: str) -> dict:
    # Strip fences if model added them anyway.
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    # First {...} block.
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return {"_parse_error": text}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError as e:
        return {"_parse_error": f"{e}: {m.group(0)}"}
