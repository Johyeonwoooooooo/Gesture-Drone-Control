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
- target_object: the object to find. MUST be exactly one of these detector classes:
  cabinet, bed, chair, sofa, table, door, window, bookshelf, picture, counter,
  desk, curtain, refrigerator, shower curtain, toilet, sink, bathtub, otherfurniture.
  Pick the closest class for anything else (tv/모니터/전자기기 -> otherfurniture,
  장롱/옷장 -> cabinet, 세면대/싱크대 -> sink).
- clip_prompt:   an English CLIP-friendly prompt, usually "a <target_object>".
- location_hint: free-form location/region context from the user (e.g. "upstairs bathroom", "next room", "kitchen"). Empty string if none.
- target_room:   the ROOM ID to search. Rooms are identified by a code like "001_004" (two numbers, the scene file-name suffix). Copy that code as a string when the user names a specific room (e.g. "001_004 방" -> "001_004", "room 002_011" -> "002_011"). null if no specific room is given.
- scope:         "room" if the user wants a specific room (target_room set), "building" if the user wants to search the WHOLE house/building (e.g. "집 전체", "온 집", "all rooms", "whole house"), or "" if unspecified.
- action:        one of ["take_photo", "inspect", "goto", "other"].
- return_home:   true if the user asks the drone to come back, else false.

Always answer with a single JSON object, no prose, no markdown fences.
Translate Korean object names to English. Keep it terse.

Examples:

User: 001_004 방에서 의자 찾아줘
{"target_object":"chair","clip_prompt":"a chair","location_hint":"room 001_004","target_room":"001_004","scope":"room","action":"goto","return_home":false}

User: 002_011 방에 있는 tv 사진 찍어와줘
{"target_object":"otherfurniture","clip_prompt":"a tv","location_hint":"room 002_011","target_room":"002_011","scope":"room","action":"take_photo","return_home":true}

User: 집 전체에서 냉장고 찾아줘
{"target_object":"refrigerator","clip_prompt":"a refrigerator","location_hint":"whole house","target_room":null,"scope":"building","action":"goto","return_home":false}

User: 옆 방에 있는 TV 사진 찍어와줘
{"target_object":"otherfurniture","clip_prompt":"a tv","location_hint":"next room","target_room":null,"scope":"","action":"take_photo","return_home":true}

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
    target_room: Optional[str] = None   # room id suffix like "001_004", or None
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
    def generate(self, system: str, user: str, max_new_tokens: int = 256) -> str:
        """Raw single-turn completion. Shared by `parse`, the patrol intent
        parser (webapp_llm_v2/patrol_intent.py) and the patrol report writer
        (webapp_llm_v2/patrol_report.py) so they all reuse ONE loaded model."""
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

    @torch.no_grad()
    def parse(self, user_text: str, max_new_tokens: int = 200,
              room_directory: str = "") -> ParsedIntent:
        system = SYSTEM_PROMPT
        if room_directory:
            system = (
                f"{SYSTEM_PROMPT}\n"
                "Use this room directory to pick target_room by floor/type when the "
                "user names a room by description (e.g. '위층 화장실', '거실'). Choose "
                "the room_id whose floor and type/aliases best match; copy its code "
                "into target_room and set scope='room'.\n"
                f"{room_directory}\n"
            )
        gen = self.generate(system, user_text, max_new_tokens=max_new_tokens)
        data = _extract_json(gen)

        target = str(data.get("target_object", "")).strip().lower() or "object"
        clip_prompt = str(data.get("clip_prompt", "")).strip() or f"a {target}"
        return ParsedIntent(
            target_object=target,
            clip_prompt=clip_prompt,
            location_hint=str(data.get("location_hint", "")).strip(),
            action=str(data.get("action", "goto")).strip(),
            return_home=bool(data.get("return_home", False)),
            target_room=_coerce_room_id(data.get("target_room")),
            scope=str(data.get("scope", "")).strip().lower(),
            raw=data,
            raw_text=gen,
        )


def _coerce_room_id(v) -> Optional[str]:
    """Best-effort room id like '001_004' from the LLM field.

    Accepts '001_004', '1_4', 'room 002_011', etc. Each numeric part is
    zero-padded to 3 digits to match the region file-name suffix. Returns None
    if no `<num>_<num>` pattern is present.
    """
    if v is None or isinstance(v, bool):
        return None
    m = re.search(r"(\d+)_(\d+)", str(v))
    if not m:
        return None
    return f"{int(m.group(1)):03d}_{int(m.group(2)):03d}"


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
