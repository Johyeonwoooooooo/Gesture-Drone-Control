"""The torch-free half of the intent parser: prompt, schema, JSON coercion.

`parse()` is written entirely in terms of `generate(system, user)`, so the two
parsers share it:

    LocalLLMParser   (patrol/llm_parser.py)  — generate() runs the model here
    RemoteLLMParser  (patrol/remote_llm.py)  — generate() POSTs to an LLM server

That split is the whole point of this module. Nothing here imports torch, so a
laptop with only numpy can run `patrol/server.py --llm-url ...` and drive the
pipeline against a GPU box that serves the model. See README §4-B.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

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


class BaseLLMParser:
    """Everything the pipeline needs from an LLM, minus the LLM itself.

    Subclasses implement `generate`; `patrol_intent.parse_patrol` and
    `patrol_report.build_report` call that directly, `server.py` calls `parse`.
    """

    model_id: str = ""

    def generate(self, system: str, user: str, max_new_tokens: int = 256) -> str:
        """Raw single-turn completion for a (system, user) prompt pair."""
        raise NotImplementedError

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
