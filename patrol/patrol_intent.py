"""Patrol intent parsing — "어디를 순찰할까" (vs. "무엇을 찾을까").

`patrol/llm_base.py` answers the object-finding question and its SYSTEM_PROMPT
is shared with the v1 webapp, so it is left untouched. This module adds a
SECOND prompt over the SAME parser handle (`BaseLLMParser.generate`, whether
that runs the model in-process or over HTTP) and resolves its answer to
concrete `RoomInfo`s.

    "현우방만 탐색해줘"     -> mode=patrol, rooms=[002_012]
    "2층 전부 순찰해줘"     -> mode=patrol, rooms=[all floor-1 rooms]
    "거실 소파 찾아줘"      -> mode=find   (server falls through to run_query)

Room resolution never relies on the LLM alone — an alias/keyword/floor scan of
the raw text backs it up, mirroring `litept_backend.resolve_room`'s 3-tier
fallback. A 3B model reliably fails at least one of these on its own.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

try:
    from patrol.litept_backend import ROOM_KW_MAP
    from patrol.room_index import RoomInfo, room_directory_text
except ImportError:  # plain-script import path
    from litept_backend import ROOM_KW_MAP  # type: ignore
    from room_index import RoomInfo, room_directory_text  # type: ignore

PATROL_SYSTEM_PROMPT = """You route natural-language drone commands for an indoor patrol drone.

Decide first WHAT KIND of command it is:
- mode "patrol": the user wants the drone to SEARCH / SWEEP / CHECK an AREA
  (탐색, 순찰, 확인, 둘러봐, 사람 있는지 봐줘, patrol, sweep, check).
- mode "find":   the user wants a specific OBJECT (소파 찾아줘, 냉장고 어디야,
  find the chair, tv 사진 찍어줘).
When both appear ("현우방에서 의자 찾아줘"), the object wins -> "find".

Then fill in the patrol area (leave empty for mode "find"):
- target_rooms: list of ROOM IDs like "002_012" (two numbers). Use the room
  directory below: match the user's words against the aliases (별칭), the floor
  (층) and the room type. Empty list if the user did not name a specific room.
- room_types:   list from [living, bedroom, kitchen, bathroom, office, dining]
  when the user names a KIND of room ("화장실 전부", "모든 침실"). Else [].
- floors:       list of FLOOR NUMBERS AS THE USER SAYS THEM ("2층" -> 2,
  "1층부터 2층" -> [1,2]). Empty list if no floor is mentioned.
- scope:        "room" (specific rooms), "floor" (whole floor(s)),
  "building" (집 전체 / 온 집 / whole house), or "" if unclear.
- return_home:  true unless the user says not to come back. Default true.

Always answer with a single JSON object, no prose, no markdown fences.

Examples:

User: 현우방만 탐색해줘
{"mode":"patrol","target_rooms":["002_012"],"room_types":[],"floors":[],"scope":"room","return_home":true}

User: 2층 전부 순찰해줘
{"mode":"patrol","target_rooms":[],"room_types":[],"floors":[2],"scope":"floor","return_home":true}

User: 집 전체 돌면서 사람 있는지 확인해줘
{"mode":"patrol","target_rooms":[],"room_types":[],"floors":[],"scope":"building","return_home":true}

User: 3층 화장실들 확인해줘
{"mode":"patrol","target_rooms":[],"room_types":["bathroom"],"floors":[3],"scope":"floor","return_home":true}

User: 거실 소파 찾아줘
{"mode":"find","target_rooms":[],"room_types":[],"floors":[],"scope":"","return_home":false}
"""

# Korean words that force a patrol regardless of what the LLM decided.
PATROL_KW = ("순찰", "탐색", "둘러", "살펴", "수색", "정찰", "돌아봐", "돌면서",
             "확인해", "점검", "patrol", "sweep", "scan")
# ...and words that force object-find (an object query mentioning "확인해줘").
FIND_KW = ("찾아", "어디", "가져", "find", "where")

_FLOOR_RE = re.compile(r"(\d+)\s*층")
_UP_WORDS = ("위층", "윗층", "위 층", "upstairs")
_DOWN_WORDS = ("아래층", "아랫층", "아래 층", "downstairs", "1층")


@dataclass
class PatrolIntent:
    mode: str = "find"                      # "patrol" | "find"
    target_rooms: List[str] = field(default_factory=list)
    room_types: List[str] = field(default_factory=list)
    floors_kr: List[int] = field(default_factory=list)  # as the user says them
    scope: str = ""                         # room | floor | building | ""
    return_home: bool = True
    raw: dict = field(default_factory=dict)
    raw_text: str = ""

    @property
    def is_patrol(self) -> bool:
        return self.mode == "patrol"


def parse_patrol(llm, user_text: str, index: Dict[str, RoomInfo],
                 max_new_tokens: int = 200) -> PatrolIntent:
    """LLM patrol routing, with keyword overrides for the mode decision."""
    system = f"{PATROL_SYSTEM_PROMPT}\n{room_directory_text(index)}\n"
    gen = llm.generate(system, user_text, max_new_tokens=max_new_tokens)
    data = _extract_json(gen)

    mode = str(data.get("mode", "")).strip().lower()
    low = user_text.lower()
    # The 3B model happily calls a bare "현우방만 탐색해줘" a "find". Korean verb
    # keywords are unambiguous here, so they win — but an explicit object query
    # ("소파 찾아줘") stays a find even if it also says "확인해줘".
    if any(k in low for k in FIND_KW):
        mode = "find"
    elif any(k in low for k in PATROL_KW):
        mode = "patrol"
    elif mode not in ("patrol", "find"):
        mode = "find"

    return PatrolIntent(
        mode=mode,
        target_rooms=[r for r in map(_coerce_room_id,
                                     _as_list(data.get("target_rooms"))) if r],
        room_types=[str(t).strip().lower()
                    for t in _as_list(data.get("room_types")) if str(t).strip()],
        floors_kr=[int(f) for f in _as_list(data.get("floors"))
                   if str(f).strip().lstrip("-").isdigit()],
        scope=str(data.get("scope", "")).strip().lower(),
        return_home=bool(data.get("return_home", True)),
        raw=data if isinstance(data, dict) else {},
        raw_text=gen,
    )


def resolve_rooms(intent: PatrolIntent, index: Dict[str, RoomInfo],
                  raw_text: str, *, min_area_m2: float = 2.0,
                  max_rooms: int = 12) -> tuple[List[RoomInfo], str]:
    """(rooms to patrol, one-line explanation of how they were chosen).

    Tiers, most specific first. A room named outright (id or alias) is never
    dropped by the small-room filter; broad scopes are filtered and capped so
    "집 전체 순찰" does not schedule a 22-room flight through every closet.
    """
    text = raw_text.lower()
    floors = _resolve_floors(intent, raw_text, index)
    offset = next(iter(index.values())).floor_offset if index else 1

    # 1. explicit room ids from the LLM
    picked = [index[r] for r in intent.target_rooms if r in index]
    if picked:
        return _dedup(picked)[:max_rooms], "방 코드 지정"

    # 2. alias substring match on the raw text (longest alias first)
    alias_hits: List[RoomInfo] = []
    pairs = sorted(((a, r) for r in index.values() for a in r.aliases),
                   key=lambda p: -len(p[0]))
    for alias, room in pairs:
        if alias.lower() in text and room not in alias_hits:
            alias_hits.append(room)
    if alias_hits:
        return _dedup(alias_hits)[:max_rooms], f"별칭 매칭 ({alias_hits[0].aliases[0]})"

    # 3. room types (LLM list, else a Korean keyword scan)
    types = list(intent.room_types)
    if not types:
        for kw, rt in sorted(ROOM_KW_MAP.items(), key=lambda x: -len(x[0])):
            if kw in raw_text:
                types.append(rt)
                break
    pool = list(index.values())
    if floors:
        pool = [r for r in pool if r.floor in floors]
    if types:
        typed = [r for r in pool if r.room_type in types]
        if typed:
            return (_filter_small(typed, min_area_m2)[:max_rooms],
                    f"방 종류={'/'.join(types)}"
                    + (f", 층={sorted(floors)}" if floors else ""))

    # 4. floor sweep
    if floors:
        return (_filter_small(pool, min_area_m2)[:max_rooms],
                "/".join(f"{f + offset}층" for f in sorted(floors)) + " 전체")

    # 5. whole building
    if intent.scope == "building" or any(
            k in raw_text for k in ("집 전체", "온 집", "전체", "모든 방", "whole house", "all rooms")):
        return _filter_small(list(index.values()), min_area_m2)[:max_rooms], "건물 전체"

    return [], "구역을 특정하지 못함"


# ------------------------------------------------------------------- internals

def _resolve_floors(intent: PatrolIntent, raw_text: str,
                    index: Dict[str, RoomInfo]) -> List[int]:
    """LLM floors + a "N층 / 위층 / 아래층" scan -> 0-based floor indices."""
    offset = next(iter(index.values())).floor_offset if index else 1
    available = sorted({r.floor for r in index.values()})
    kr = list(intent.floors_kr)
    kr += [int(m) for m in _FLOOR_RE.findall(raw_text)]
    floors = sorted({k - offset for k in kr})
    floors = [f for f in floors if f in available]
    if floors:
        return floors
    if any(w in raw_text for w in _UP_WORDS) and available:
        return [available[-1]]
    if any(w in raw_text for w in _DOWN_WORDS) and available:
        return [available[0]]
    return []


def _filter_small(rooms: Sequence[RoomInfo], min_area_m2: float) -> List[RoomInfo]:
    """Drop closet-sized rooms from a broad sweep; never return an empty list."""
    big = [r for r in rooms if float(r.size_xy[0] * r.size_xy[1]) >= min_area_m2]
    return _dedup(big if big else list(rooms))


def _dedup(rooms: Sequence[RoomInfo]) -> List[RoomInfo]:
    seen, out = set(), []
    for r in rooms:
        if r.room_name not in seen:
            seen.add(r.room_name)
            out.append(r)
    return out


def _as_list(v) -> list:
    if v is None or isinstance(v, bool):
        return []
    if isinstance(v, (list, tuple)):
        return list(v)
    return [v]


def _coerce_room_id(v) -> Optional[str]:
    if v is None or isinstance(v, bool):
        return None
    m = re.search(r"(\d+)_(\d+)", str(v))
    return f"{int(m.group(1)):03d}_{int(m.group(2)):03d}" if m else None


def _extract_json(text: str) -> dict:
    """First {...} block of a generation. Near-twin of
    `patrol.llm_base._extract_json`, but with a different failure contract, so
    it stays its own function: a failed parse degrades to {} — the keyword
    fallbacks then carry the query — where llm_base reports {"_parse_error":...}
    because a find query has nothing to fall back on.
    """
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip(),
                  flags=re.MULTILINE).strip()
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}
