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
    from patrol.room_index import RoomInfo, person_groups, room_directory_text
except ImportError:  # plain-script import path
    from litept_backend import ROOM_KW_MAP  # type: ignore
    from room_index import (RoomInfo, person_groups,  # type: ignore
                            room_directory_text)

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
- order:        list of ROOM IDs in the order the user asked to visit them,
  when the user STATES an order ("A 갔다가 B", "A부터", "A 먼저", "A 다음에 B").
  Empty list if the user stated no order — do NOT invent one, an empty list is
  the right answer for "2층 전부 순찰해줘".

Always answer with a single JSON object, no prose, no markdown fences.

Examples:

User: 채원의 금고가 있다는 소문의 방만 탐색해줘
{"target_rooms":["002_012"],"room_types":[],"floors":[],"scope":"room","return_home":true,"order":[]}

User: 2층 전부 순찰해줘
{"target_rooms":[],"room_types":[],"floors":[2],"scope":"floor","return_home":true,"order":[]}

User: 집 전체 돌면서 사람 있는지 확인해줘
{"target_rooms":[],"room_types":[],"floors":[],"scope":"building","return_home":true,"order":[]}

User: 3층 화장실들 확인해줘
{"target_rooms":[],"room_types":["bathroom"],"floors":[3],"scope":"floor","return_home":true,"order":[]}

User: 채원의 금고가 있다는 소문의 방 갔다가 규철의 지하조직 본부 순찰해줘
{"target_rooms":["002_012","002_016"],"room_types":[],"floors":[],"scope":"room","return_home":true,"order":["002_012","002_016"]}

User: 지윤의 위장 세탁소 먼저 보고 현우의 이중장부 서재 확인해줘
{"target_rooms":["002_015","002_014"],"room_types":[],"floors":[],"scope":"room","return_home":true,"order":["002_015","002_014"]}
"""

_FLOOR_RE = re.compile(r"(\d+)\s*층")
# "그 종류를 남김없이" 를 뜻하는 말. 홀로 선 '다' 만 세고 '갔다가/났다' 같은
# 어미는 세지 않는다 — 어미까지 세면 웬만한 문장이 전부 '전체' 요청이 된다.
_ALL_RE = re.compile(r"전부|모두|모든|전체|싹|죄다|(^|\s)다(\s|$)")
_UP_WORDS = ("위층", "윗층", "위 층", "upstairs")
_DOWN_WORDS = ("아래층", "아랫층", "아래 층", "downstairs", "1층")


@dataclass
class PatrolIntent:
    target_rooms: List[str] = field(default_factory=list)
    room_types: List[str] = field(default_factory=list)
    floors_kr: List[int] = field(default_factory=list)  # as the user says them
    scope: str = ""                         # room | floor | building | ""
    return_home: bool = True
    order: List[str] = field(default_factory=list)   # visit order the user SAID
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
        order=[r for r in map(_coerce_room_id,
                              _as_list(data.get("order"))) if r],
        raw=data if isinstance(data, dict) else {},
        raw_text=gen,
    )


def resolve_order(intent: PatrolIntent, rooms: Sequence[RoomInfo]) -> List[RoomInfo]:
    """The visit order the user actually stated, as far as it can be trusted.

    The model's `order` is a suggestion about the SENTENCE, not about the map, so
    it is checked against the rooms we already resolved rather than believed:
    ids we did not pick are dropped (the model likes to name a neighbour it saw
    in the directory) and duplicates collapse. Whatever is left is a prefix —
    `order_rooms` fills the rest by distance, so a partial or empty answer costs
    nothing. That is what makes trusting the model here cheap: it can only
    reorder rooms that some other tier already decided we are visiting.
    """
    by_name = {r.room_name: r for r in rooms}
    out: List[RoomInfo] = []
    for rid in intent.order:
        room = by_name.get(rid)
        if room is not None and room not in out:
            out.append(room)
    return out


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

    # 1. rooms the user NAMED, from two directions that are unioned rather than
    #    raced: ids the model proposed (backed by the text) and aliases found in
    #    the text. Racing them loses rooms — asked "본부 갔다가 금고 순찰해줘"
    #    the model answers ["002_016"], and if that short list wins outright the
    #    금고 the user typed by name never gets patrolled. Neither side is
    #    complete on its own, so take both.
    #
    #    The model's ids still need backing: asked something off-topic ("냉장고
    #    찾아줘") a 3B fills target_rooms with plausible-looking codes, and
    #    silently patrolling three invented rooms is worse than admitting we did
    #    not understand. An id counts only if its code, one of its aliases, or
    #    its type/floor shows up in what the user actually wrote.
    #    `text_floors` again, not `floors`: backing an id with the model's own
    #    guessed floor is circular — it lets the model vouch for itself. Asked
    #    "무기고 확인해줘" it answers 000_002 (the 벙커) and volunteers 1층, and
    #    the id passes because it sits on the floor the same answer invented.
    picked = [index[r] for r in intent.target_rooms
              if r in index and _text_backs_room(index[r], text, text_floors)]

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

    # 2a. a PERSON named in the text, when no single room was ("규철이방 다
    #     순찰해줘", "현우 관련 구역 전부"). The screen names of this building are
    #     organised by person — 규철 owns four of them — so a person reads as a
    #     group the same way 화장실 reads as a kind. Neither the LLM nor the rest
    #     of the tiers can do it: a 3B scores 0/5 picking "the 현우 ones" out of
    #     the directory (7B too), and without this the only thing left matching
    #     "규철이방" is the single letter 방, which lands on every bedroom.
    #
    #     Gated on there being NO alias hit, because a full room name contains a
    #     person too: "현우의 이중장부 서재" must stay one room, not become all
    #     four 현우 rooms. Naming a room is the more specific request, so it wins.
    #     The model's `picked` is dropped here as well — for a group request its
    #     ids are guesses at which member to visit, and we want all of them.
    person_hits: List[RoomInfo] = []
    people: List[str] = []
    if not alias_hits:
        for who, group in person_groups(index).items():
            if who in raw_text:
                people.append(who)
                person_hits.extend(group)
    if text_floors:
        person_hits = [r for r in person_hits if r.floor in text_floors]
    if person_hits:
        return _capped(_dedup(person_hits),
                       f"{'/'.join(sorted(people))} 관련 구역", max_rooms)

    # 2b. a KIND of room the model named ("3층 침실 전부") is a claim about the
    #     WHOLE request, so it joins the union too instead of losing to whatever
    #     partial id list arrived beside it — the model answers 침실 3개 for a
    #     floor that has 4 and, raced, that short list would win.
    #     Gated on the user actually saying "전부/모두/…", not merely on the
    #     model having filled room_types. The model sets that field for by-name
    #     queries too — it answers bedroom for "채원의 심문실(추정) 순찰해줘" —
    #     so keying on the field alone drags every bedroom into a request for one
    #     room. "전부" is the thing that means "every room of this kind"; without
    #     it a type is at most a description of the room already named.
    #     `intent.room_types` only, never the keyword scan in tier 3: that one
    #     fires on a bare "방" sitting inside a room's own name ("채원의 금고가
    #     있다는 소문의 방").
    typed_hits: List[RoomInfo] = []
    n_small = 0
    if intent.room_types and _wants_all(raw_text):
        pool = [r for r in index.values() if not floors or r.floor in floors]
        of_type = [r for r in pool if r.room_type in intent.room_types]
        typed_hits = _filter_small(of_type, min_area_m2)
        n_small = len(of_type) - len(typed_hits)

    named = _dedup(picked + alias_hits + typed_hits)
    if named:
        bits = []
        if picked and alias_hits:
            bits.append("방 이름 지정")
        elif picked:
            bits.append("방 코드 지정")
        elif alias_hits:
            bits.append(f"별칭 매칭 ({alias_hits[0].aliases[0]})")
        if typed_hits:
            bits.append(f"방 종류={'/'.join(intent.room_types)}"
                        + (f", 층={sorted(floors)}" if floors else ""))
        return _capped(named, " + ".join(bits), max_rooms, n_small)

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
            kept = _filter_small(typed, min_area_m2)
            return _capped(kept, f"방 종류={'/'.join(types)}"
                           + (f", 층={sorted(floors)}" if floors else ""),
                           max_rooms, len(typed) - len(kept))

    # 4. floor sweep
    if floors:
        kept = _filter_small(pool, min_area_m2)
        return _capped(kept, "/".join(f"{f + offset}층" for f in sorted(floors))
                       + " 전체", max_rooms, len(pool) - len(kept))

    # 5. whole building
    if intent.scope == "building" or any(
            k in raw_text for k in ("집 전체", "온 집", "전체", "모든 방", "whole house", "all rooms")):
        every = list(index.values())
        kept = _filter_small(every, min_area_m2)
        return _capped(kept, "건물 전체", max_rooms, len(every) - len(kept))

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


def _wants_all(raw_text: str) -> bool:
    """Did the user ask for EVERY room of a kind ("3층 침실 전부")?"""
    return bool(_ALL_RE.search(raw_text))


def _capped(rooms: Sequence[RoomInfo], why: str, max_rooms: int,
            n_small: int = 0) -> tuple[List[RoomInfo], str]:
    """Apply the room cap and say so — both trims belong in `why`.

    Dropping rooms silently reads on screen as "this is your whole patrol". 3층
    has 13 rooms against a default cap of 12, and 1층's second room is a 0.7 m²
    detection the closet filter removes, so "3층 전부"/"1층 전부" both come back
    short on this very building. The console prints `why` verbatim, so the count
    is the only place a user can notice.
    """
    out = list(rooms[:max_rooms])
    notes = []
    if n_small:
        notes.append(f"작은 방 {n_small}개 제외")
    if len(rooms) > max_rooms:
        notes.append(f"{len(rooms)}개 중 {max_rooms}개만, 상한")
    return out, why + (f" ({', '.join(notes)})" if notes else "")


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
