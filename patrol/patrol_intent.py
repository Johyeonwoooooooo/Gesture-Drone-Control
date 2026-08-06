"""Patrol intent parsing — "어디를 순찰할까".

The one LLM prompt in the pipeline that decides where the drone goes. It sits
on top of `generate(system, user)` and resolves its answer to concrete
`RoomInfo`s.

    "현우방만 탐색해줘"     -> rooms=[002_012]
    "2층 전부 순찰해줘"     -> rooms=[all floor-1 rooms]

Room resolution never relies on the LLM alone — an alias/keyword/floor scan of
the raw text backs it up in five tiers. A 3B model reliably fails at least one
of these on its own.
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

PATROL_SYSTEM_PROMPT = """You pick the patrol area for an indoor drone from a natural-language command.

Fill in:
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
{"target_rooms":["002_012"],"room_types":[],"floors":[],"scope":"room","return_home":true}

User: 2층 전부 순찰해줘
{"target_rooms":[],"room_types":[],"floors":[2],"scope":"floor","return_home":true}

User: 집 전체 돌면서 사람 있는지 확인해줘
{"target_rooms":[],"room_types":[],"floors":[],"scope":"building","return_home":true}

User: 3층 화장실들 확인해줘
{"target_rooms":[],"room_types":["bathroom"],"floors":[3],"scope":"floor","return_home":true}
"""

_FLOOR_RE = re.compile(r"(\d+)\s*층")
_UP_WORDS = ("위층", "윗층", "위 층", "upstairs")
_DOWN_WORDS = ("아래층", "아랫층", "아래 층", "downstairs", "1층")


@dataclass
class PatrolIntent:
    target_rooms: List[str] = field(default_factory=list)
    room_types: List[str] = field(default_factory=list)
    floors_kr: List[int] = field(default_factory=list)  # as the user says them
    scope: str = ""                         # room | floor | building | ""
    return_home: bool = True
    raw: dict = field(default_factory=dict)
    raw_text: str = ""


def parse_patrol(llm, user_text: str, index: Dict[str, RoomInfo],
                 max_new_tokens: int = 200) -> PatrolIntent:
    """Ask the LLM which area to patrol. Resolution happens in resolve_rooms."""
    system = f"{PATROL_SYSTEM_PROMPT}\n{room_directory_text(index)}\n"
    gen = llm.generate(system, user_text, max_new_tokens=max_new_tokens)
    data = _extract_json(gen)

    return PatrolIntent(
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
    # The floors the USER actually spelled out, with the model's guess left out.
    # `intent.floors_kr` is not a claim that a floor was mentioned — the 3B model
    # volunteers one for nearly every query, floor 1 by default. That guess is
    # fine as a hint for the broad tiers, but it must never veto a room the user
    # named outright, so the alias tier filters on this instead.
    text_floors = _resolve_floors(PatrolIntent(), raw_text, index)
    offset = next(iter(index.values())).floor_offset if index else 1

    # 1. explicit room ids from the LLM — but only ones the TEXT backs up.
    #    Asked something off-topic ("냉장고 찾아줘"), the 3B model still fills
    #    target_rooms with plausible-looking ids. Silently patrolling three
    #    invented rooms is worse than admitting we did not understand, so a id
    #    only counts if its code, one of its aliases, or its type/floor shows
    #    up in what the user actually wrote.
    picked = [index[r] for r in intent.target_rooms
              if r in index and _text_backs_room(index[r], text, floors)]
    if picked:
        return _dedup(picked)[:max_rooms], "방 코드 지정"

    # 2. alias substring match on the raw text (longest alias first).
    #    A floor SPELLED IN THE TEXT constrains this tier. "2층 복도 순찰해줘"
    #    must not drag in the 3층 corridors just because those two letters
    #    appear — an alias is a substring test, it knows nothing about 층.
    #    `text_floors`, not `floors`: filtering on the model's guessed floor
    #    silently vetoes rooms the user named in full ("규철의 지하조직 본부" ->
    #    the model volunteers 2층, the room is on 3층, the name loses).
    #    When the filter empties the list we FALL THROUGH rather than return the
    #    wrong-floor room: the user named a floor, so the type/floor tiers below
    #    answer it better than a confident miss.
    alias_hits: List[RoomInfo] = []
    pairs = sorted(((a, r) for r in index.values() for a in r.aliases),
                   key=lambda p: -len(p[0]))
    for alias, room in pairs:
        if alias.lower() in text and room not in alias_hits:
            alias_hits.append(room)
    if text_floors:
        alias_hits = [r for r in alias_hits if r.floor in text_floors]
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

def _text_backs_room(room: RoomInfo, text: str, floors: Sequence[int]) -> bool:
    """Is there anything in the user's words pointing at this room?"""
    if room.room_name in text or room.room_name.split("_")[-1] in text:
        return True
    if any(a.lower() in text for a in room.aliases):
        return True
    if room.floor in floors:
        return True
    for kw, rt in ROOM_KW_MAP.items():
        if rt == room.room_type and kw in text:
            return True
    return False


def _resolve_floors(intent: PatrolIntent, raw_text: str,
                    index: Dict[str, RoomInfo]) -> List[int]:
    """LLM floors + a "N층 / 위층 / 아래층" scan -> 0-based floor indices."""
    offset = next(iter(index.values())).floor_offset if index else 1
    available = sorted({r.floor for r in index.values()})
    # An explicit "N층" in the text WINS over the LLM's list — it is not merged
    # with it. The 3B model answers [1, 2] to "2층 전부 순찰해줘" often enough
    # that a union quietly doubles the mission.
    spelled = [int(m) for m in _FLOOR_RE.findall(raw_text)]
    kr = spelled if spelled else list(intent.floors_kr)
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
    """First {...} block of a generation, or {} if there is none.

    Failure is deliberately quiet: an unparseable answer degrades to an empty
    dict and the five resolution tiers below carry the query on the raw text
    alone. Raising here would turn a model hiccup into a dead query.
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
