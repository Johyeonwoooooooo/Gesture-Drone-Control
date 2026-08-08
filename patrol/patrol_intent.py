"""Patrol intent parsing — "어디를 순찰할까".

The one LLM prompt in the pipeline that decides where the drone goes. It sits
on top of `generate(system, user)` and resolves its answer to concrete
`RoomInfo`s.

    "현우방만 탐색해줘"     -> rooms=[현우 소유 방 전부]
    "2층 전부 순찰해줘"     -> rooms=[all floor-1 rooms]
    "전체 방 순찰해줘"      -> rooms=[every room, uncapped]
    "절반만 해줘"           -> rooms=[a random half, floor-ascending]
    "아무거나 해줘"         -> rooms=[3 random rooms, floor-ascending]

Room resolution never relies on the LLM alone — an alias/keyword/floor scan of
the raw text backs it up in five tiers. A 3B model reliably fails at least one
of these on its own.
"""
from __future__ import annotations

import json
import random
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
_UP_WORDS = ("위층", "윗층", "위 층", "upstairs")
_DOWN_WORDS = ("아래층", "아랫층", "아래 층", "downstairs", "1층")

# 몇 개를 돌지를 정하는 말. 어느 방인지(층·종류·별칭)와 직교한다 — "2층 절반만"
# 은 층으로 후보를 고르고 개수로 자른다. LLM 필드가 아니라 원문 스캔인 이유는
# `_resolve_floors` 의 "N층" 과 같다: 3B 모델이 자주 흘리는데, 글자는 안 흘린다.
# 오프라인 모드(--llm-url "")에서도 그대로 동작하는 건 덤이다.
#
# "남김없이" 는 두 곳이 같은 뜻으로 쓴다 — `_quantity` 의 "all"(상한을 풀고
# 후보 전체를 돈다)과 `_wants_all`(방 종류를 "그 종류 전부"로 읽어도 되는가).
# 같은 말이므로 정규식도 하나다. 홀로 선 '다' 만 세고 '갔다가/났다' 같은 어미는
# 세지 않는다 — 어미까지 세면 웬만한 문장이 전부 '전체' 요청이 된다.
_ALL_RE = re.compile(r"전부|모두|모든|전체|싹|죄다|온 집|(^|\s)다(\s|$)"
                     r"|whole house|all rooms|everything")
_HALF_WORDS = ("절반", "반만", "반 정도", "half")
_ANY_WORDS = ("아무", "랜덤", "random")

# "아무거나 해줘" 로 뽑는 방 개수. 데모용 짧은 순찰이 목적이라 작게 잡는다.
ANY_ROOMS = 3


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
    qty = _quantity(raw_text)
    offset = next(iter(index.values())).floor_offset if index else 1

    # 개수만 말하고 층은 안 말했으면 층은 없는 것이다. 이 자리에 남아 있는 층은
    # 모델이 지어낸 것뿐인데("절반만 해줘" 에 floors:[1] 을 심심찮게 답한다),
    # 그게 남으면 집 전체의 절반이 1층의 절반으로 조용히 쪼그라든다. `text_floors`
    # 가 비었다는 게 곧 "사용자는 층을 말하지 않았다" 이므로 그걸 그대로 쓴다.
    if qty and not text_floors:
        floors = []

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
    matched: List[str] = []          # the alias strings that actually hit
    pairs = sorted(((a, r) for r in index.values() for a in r.aliases),
                   key=lambda p: -len(p[0]))
    for alias, room in pairs:
        if alias.lower() in text:
            matched.append(alias)
            if room not in alias_hits:
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
    #     A person whose name sits INSIDE a room name we already matched does not
    #     count: "현우의 이중장부 서재" must stay one room, not become all four
    #     현우 rooms — naming a room is the more specific request. But that check
    #     is per person, not "any alias matched at all". In "규철이방 다랑 현우의
    #     이중장부 서재 가줘" 현우 is spoken for by the matched name while 규철 is
    #     not, so 규철 still opens up into its group and the two are unioned.
    #     The model's `picked` is dropped for the group part — for a group request
    #     its ids are guesses at which member to visit, and we want all of them.
    spoken_for = " ".join(matched)
    person_hits: List[RoomInfo] = []
    people: List[str] = []
    for who, group in person_groups(index).items():
        if who in raw_text and who not in spoken_for:
            people.append(who)
            person_hits.extend(group)
    if text_floors:
        person_hits = [r for r in person_hits if r.floor in text_floors]
    if person_hits:
        why = f"{'/'.join(sorted(people))} 관련 구역"
        if alias_hits:
            why += " + 방 이름 지정"
        return _capped(_dedup(person_hits + alias_hits), why, max_rooms)

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
    # "전체 방"/"아무 방" 의 '방' 은 방 종류가 아니라 그냥 방이다. ROOM_KW_MAP 이
    # '방'->bedroom 을 들고 있어서 "전체 방 순찰해줘" 가 침실만 골랐다. 개수를
    # 말한 질의에서는, 진짜 종류 단어(거실·화장실·주방…)가 원문에 없으면 종류를
    # 통째로 버린다 — 모델이 채운 room_types 도 같이 (같은 이유로 틀리므로).
    if qty and not any(kw in raw_text for kw in ROOM_KW_MAP if kw != "방"):
        types = []
    pool = list(index.values())
    if floors:
        pool = [r for r in pool if r.floor in floors]
    if types:
        typed = [r for r in pool if r.room_type in types]
        if typed:
            kept = _filter_small(typed, min_area_m2)
            return _take(kept, qty, max_rooms,
                         f"방 종류={'/'.join(types)}"
                         + (f", 층={sorted(floors)}" if floors else ""),
                         len(typed) - len(kept))

    # 4. floor sweep
    if floors:
        kept = _filter_small(pool, min_area_m2)
        return _take(kept, qty, max_rooms,
                     "/".join(f"{f + offset}층" for f in sorted(floors)) + " 전체",
                     len(pool) - len(kept))

    # 5. whole building. 개수를 말한 것("절반만", "아무거나", "전부")도 여기로 온다
    #    — 어디인지는 안 말했으니 후보는 집 전체이고, 자르는 건 `_take` 다.
    #    예전의 "전체/모든 방/온 집/whole house/all rooms" 문자열 목록은 그대로
    #    `_ALL_RE`(-> qty == "all") 가 됐으므로 `qty` 하나로 갈음한다.
    if intent.scope == "building" or qty or "집 전체" in raw_text:
        every = list(index.values())
        kept = _filter_small(every, min_area_m2)
        return _take(kept, qty, max_rooms, "건물 전체", len(every) - len(kept))

    # 6. 아무 규칙에도 안 걸렸다 = 원문에 우리가 아는 말이 하나도 없다. 마지막
    #    수단으로 LLM 이 고른 방을 그냥 쓴다. 위(1번)에서는 `_text_backs_room`
    #    이 이걸 걸렀는데, 그건 규칙이 답을 낼 수 있을 때 모델의 추측이 끼어드는
    #    걸 막으려는 것이었다. 여기까지 왔으면 낼 답이 없으므로, 남은 선택지는
    #    모델의 추측과 빈 손 둘뿐이다 — 추측이라고 말하고 내보내는 쪽이 낫다.
    guessed = _dedup([index[r] for r in intent.target_rooms if r in index])
    if guessed:
        return _take(guessed, qty, max_rooms, "LLM 추정 (원문에 근거 없음)")

    return [], "구역을 특정하지 못함"


# ------------------------------------------------------------------- internals

def _quantity(raw_text: str) -> str:
    """"전부" | "절반" | "아무거나" -> "all" | "half" | "any", 없으면 "".

    좁은 쪽부터 본다: "전체의 절반" 은 절반이지 전체가 아니다.
    """
    t = raw_text.lower()
    if any(w in t for w in _HALF_WORDS):
        return "half"
    if any(w in t for w in _ANY_WORDS):
        return "any"
    if _ALL_RE.search(t):
        return "all"
    return ""


def _take(rooms: Sequence[RoomInfo], qty: str, max_rooms: int,
          why: str, n_small: int = 0) -> tuple[List[RoomInfo], str]:
    """후보를 요청한 '양' 으로 자른다. -> (방 목록, 설명).

    `max_rooms` 는 넓은 질의가 22개짜리 비행이 되는 걸 막는 안전장치인데,
    **`"all"` 에는 걸지 않는다** — "전체 방 순찰해줘" 는 전부 돌라는 뜻이고,
    12개만 조용히 돌면 물어본 것과 다른 일을 하는 것이다.

    무작위 선택은 층 오름차순으로 되돌려 놓는다. 두 진입점 모두 그 뒤
    `room_index.order_rooms` 로 다시 정렬하지만(그쪽도 층이 1순위), 층 이동은
    올라가는 한 방향만 남기는 게 최소이고 그 성질이 이 함수 밖의 정렬에
    의존하면 안 된다.

    개수 말이 없으면 `_capped` 에 그대로 넘긴다 — 상한과 '작은 방 제외' 를
    설명에 적는 건 저쪽 일이다. 둘이 각자 자르면 설명이 둘로 갈린다.
    """
    rooms = _dedup(rooms)
    if qty == "all":
        # 상한을 풀되 '작은 방 N개 제외' 는 그대로 알린다. 상한 자리에 실제
        # 개수를 넣으면 `_capped` 가 상한 문구만 빼고 나머지를 그대로 쓴다.
        return _capped(rooms, why, len(rooms), n_small)
    if qty in ("half", "any") and rooms:
        n = (max(1, len(rooms) // 2) if qty == "half"
             else min(ANY_ROOMS, len(rooms)))
        picked = sorted(random.sample(list(rooms), n),
                        key=lambda r: (r.floor, r.room_name))
        label = "무작위 절반" if qty == "half" else "무작위"
        return picked, f"{why} 중 {label} {n}개"
    return _capped(rooms, why, max_rooms, n_small)


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


# ------------------------------------------------------------------ self-check

def _demo() -> None:
    """`python -m patrol.patrol_intent` — 개수 표현("전부/절반/아무거나") 확인.

    LLM 도 탐지 데이터도 필요 없다. 22개 방(3개 층)을 합성해서 `resolve_rooms`
    만 돌린다. `intent` 는 모델이 답했을 법한 값을 손으로 넣는다 — 특히 층을
    지어낸 경우가 이 검사의 절반이다.
    """
    import sys

    import numpy as np

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass

    # 00809 와 같은 모양: 1층 3개 / 2층 7개 / 3층 12개.
    index: Dict[str, RoomInfo] = {}
    for floor, count in ((0, 3), (1, 7), (2, 12)):
        for i in range(count):
            name = f"{floor:03d}_{len(index):03d}"
            index[name] = RoomInfo(
                room_name=name, room_dir=name, room_type="bedroom", floor=floor,
                centroid=np.array([float(i), 0.0, float(floor) * 3]),
                floor_z=float(floor) * 3,
                bbox_min=np.array([0.0, 0.0, 0.0]),
                bbox_max=np.array([3.0, 3.0, 2.5]), n_points=1000)
    all_names = set(index)

    def ask(text: str, intent: Optional[PatrolIntent] = None, **kw):
        return resolve_rooms(intent or PatrolIntent(), index, text,
                             max_rooms=kw.pop("max_rooms", 12), **kw)

    def ascending(rooms) -> bool:
        """층이 한 방향으로만 간다 = 층 이동이 그 조합에서 최소다."""
        return all(a.floor <= b.floor for a, b in zip(rooms[:-1], rooms[1:]))

    random.seed(0)   # 무작위여도 검사는 재현되게

    # 1) 전체 = 전부. max_rooms 가 잘라서는 안 된다 — 12개만 돌면 물어본 것과
    #    다른 일을 하는 것이다.
    rooms, why = ask("전체 방 순찰해줘")
    assert len(rooms) == len(index) == 22, (len(rooms), why)
    assert {r.room_name for r in rooms} == all_names
    print(f"  1) 전체 방          {len(rooms)}개 ({why}) — max_rooms=12 무시")

    # 2) 절반 = 무작위 절반, 층 오름차순.
    rooms, why = ask("절반만 해줘")
    assert len(rooms) == 11, (len(rooms), why)
    assert {r.room_name for r in rooms} <= all_names
    assert ascending(rooms), [r.room_name for r in rooms]
    print(f"  2) 절반만           {len(rooms)}개 ({why}) "
          f"층 {[r.floor for r in rooms]}")

    # 3) 아무거나 = ANY_ROOMS 개, 역시 층 오름차순.
    rooms, why = ask("아무거나 해줘")
    assert len(rooms) == ANY_ROOMS, (len(rooms), why)
    assert ascending(rooms), [r.room_name for r in rooms]
    print(f"  3) 아무거나         {len(rooms)}개 ({why}) "
          f"층 {[r.floor for r in rooms]}")

    # 정말 무작위인지 — 같은 질의를 여러 번 돌려 서로 다른 조합이 나와야 한다.
    seen = {tuple(r.room_name for r in ask("아무거나 해줘")[0]) for _ in range(20)}
    assert len(seen) > 1, "아무거나가 매번 같은 방을 고른다"

    # 4) 층과 개수는 직교한다: "2층 절반만" 은 2층 안에서만 절반.
    rooms, why = ask("2층 절반만 해줘")
    assert rooms and all(r.floor == 1 for r in rooms), [r.display for r in rooms]
    assert len(rooms) == 7 // 2, (len(rooms), why)
    print(f"  4) 2층 절반만       {len(rooms)}개 ({why})")

    # 5) 층을 말하지 않았는데 모델이 층을 지어낸 경우. 그대로 두면 집 전체의
    #    절반이 1층의 절반(1개)으로 쪼그라든다 — 3B 모델이 실제로 이렇게 답한다.
    rooms, why = ask("절반만 해줘", PatrolIntent(floors_kr=[1], scope="floor"))
    assert len(rooms) == 11, (len(rooms), why)
    assert len({r.floor for r in rooms}) > 1, "지어낸 층에 갇혔다"
    print(f"  5) 절반만(모델이 1층이라 우김) {len(rooms)}개 ({why})")

    # 6) 기존 동작은 그대로. 층을 말했으면 그 층 전체고, 개수 말이 없으면
    #    max_rooms 가 다시 상한이다.
    rooms, why = ask("2층 전부 순찰해줘")
    assert len(rooms) == 7 and all(r.floor == 1 for r in rooms), why
    rooms, _ = ask("3층 순찰해줘", max_rooms=5)
    assert len(rooms) == 5, len(rooms)
    assert not ask("냉장고 찾아줘")[0], "개수 말이 없는 헛질의가 방을 잡았다"
    print("  6) 2층 전부 7개 / 3층 상한 5개 / 헛질의 0개 — 기존 동작 유지")

    # 7) 어떤 규칙에도 안 걸리면 LLM 이 고른 방으로 떨어진다. 규칙이 답을 낼 수
    #    있을 때는 여전히 규칙이 이긴다 (2층을 말했으면 모델의 3층 추측은 무시).
    guess = PatrolIntent(target_rooms=["002_014"], scope="room")
    rooms, why = ask("냉장고 찾아줘", guess)
    assert [r.room_name for r in rooms] == ["002_014"], (rooms, why)
    assert "LLM" in why, why
    rooms, why = ask("2층 순찰해줘", guess)
    assert len(rooms) == 7 and all(r.floor == 1 for r in rooms), why
    print("  7) 규칙 미매칭이면 LLM 추정 1개, 규칙이 있으면 규칙 우선")

    print("patrol_intent: 전부·절반·아무거나 통과 "
          "— 전부는 안 잘리고, 무작위는 매번 다르고, 순서는 층 오름차순")

    _demo_naming()


def _demo_naming() -> None:
    """이름으로 방을 부르는 티어(1/2/2a/2b) 검사.

    `room_aliases.json` 을 실제로 읽되 탐지 데이터(`data/final_npy`, 저장소 밖)도
    LLM 도 쓰지 않는다 — 방의 기하는 이 티어들이 안 보기 때문에 아무 값이나 넣어도
    된다. `intent` 는 3B 가 실제로 답하던 값을 손으로 넣는다: 특히 **층을 지어낸**
    경우가 핵심이라, 이 검사 절반이 그것 때문에 있다.

    여기 있는 케이스는 전부 한때 틀렸던 것들이다. 별칭에 맨이름("현우")을 넣어
    사람 그룹을 흉내내던 시절엔 1)이 4개를 잡았고, 별칭 티어가 모델이 지어낸 층으로
    필터링하던 시절엔 2)가 **다른 방으로 날아갔다**.
    """
    import json as _json
    import sys
    from pathlib import Path

    import numpy as np

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass

    path = Path(__file__).with_name("room_aliases.json")
    raw = _json.loads(path.read_text(encoding="utf-8"))
    offset = int(raw.get("floor_offset", 1))

    index: Dict[str, RoomInfo] = {}
    for name, al in raw["aliases"].items():
        floor = int(name.split("_")[0])
        index[name] = RoomInfo(
            room_name=name, room_dir=name, room_type="bedroom", floor=floor,
            centroid=np.array([0.0, 0.0, float(floor) * 3]),
            floor_z=float(floor) * 3,
            bbox_min=np.array([0.0, 0.0, 0.0]),
            bbox_max=np.array([4.0, 4.0, 2.5]), n_points=1000,
            aliases=list(al), floor_offset=offset)

    def ask(text: str, **kw) -> List[str]:
        rooms, why = resolve_rooms(PatrolIntent(**kw), index, text,
                                   max_rooms=12)
        ask.why = why                                        # type: ignore[attr-defined]
        return [r.room_name for r in rooms]

    def group(who: str) -> List[str]:
        return sorted(r.room_name for r in person_groups(index)[who])

    # 1) 방 하나를 화면 이름으로 부르면 그 방 하나. 모델이 같은 사람의 다른 방을
    #    함께 답해도(늘 그런다) 이름이 더 구체적인 요청이므로 이름이 이긴다.
    got = ask("현우의 이중장부 서재만 순찰해줘",
              target_rooms=["002_019"], floors_kr=[3], scope="room")
    assert got == ["002_014"], (got, ask.why)               # type: ignore[attr-defined]

    # 2) 모델이 지어낸 층은 이름을 거부하지 못한다. 본부는 3층인데 3B 는 2층이라
    #    답한다 — 예전엔 그 필터에 걸려 2층의 다른 규철 방으로 날아갔다.
    got = ask("규철의 지하조직 본부 순찰해줘",
              target_rooms=["002_016"], floors_kr=[2], scope="room")
    assert got == ["002_016"], (got, ask.why)               # type: ignore[attr-defined]

    # 3) 사람 이름은 그 사람 방 전부. LLM 이 그중 하나만 답해도 전부로 편다.
    got = ask("현우방 다 순찰해줘", target_rooms=["002_014"], scope="room")
    assert sorted(got) == group("현우"), (got, ask.why)      # type: ignore[attr-defined]

    # 4) 그룹과 이름을 한 문장에. 이름으로 불린 사람(현우)은 그 방 하나만,
    #    안 불린 사람(규철)은 그룹 전체 — 그리고 둘은 합집합이다.
    got = ask("규철이방 다랑 현우의 이중장부 서재 가줘")
    assert sorted(got) == sorted(group("규철") + ["002_014"]), (got, ask.why)  # type: ignore[attr-defined]

    # 5) 사용자가 입력한 층은 별칭을 좁힌다. '복도' 는 세 층에 다 있다.
    assert ask("2층 복도만 순찰해줘") == ["001_010"], ask.why  # type: ignore[attr-defined]
    assert sorted(ask("복도 전부 순찰해줘")) == ["001_010", "002_011", "002_023"]

    # 6) 한 문장의 방 여러 개는 합집합이다. 모델은 보통 둘 중 하나만 답한다.
    got = ask("지하조직 본부랑 금고 순찰해줘", target_rooms=["002_016"],
              scope="room")
    assert sorted(got) == ["002_012", "002_016"], (got, ask.why)  # type: ignore[attr-defined]

    # 7) 아는 말이 하나도 없으면 방을 잡지 않는다 (LLM 추정도 없을 때).
    assert ask("냉장고 찾아줘") == [], ask.why                 # type: ignore[attr-defined]

    print(f"patrol_intent: 이름 티어 통과 ({len(index)}개 방) — 이름이 지어낸 층을 "
          "이기고, 사람은 그룹으로 펴지고, 이름 지정은 그룹으로 안 번진다")


if __name__ == "__main__":
    _demo()
