"""Per-room label loading + natural-language room resolution.

Reads `cache/<building>/labels.json` (schema: per region id -> {label, floor,
aliases, notes}) and exposes helpers so the LLM pipeline can resolve a room from
a natural query ("위층 화장실", "거실") instead of only an explicit `NNN_NNN`
code:

  * `load_room_labels`    - read the file (returns {} if missing — safe default)
  * `room_directory_text` - compact listing injected into the LLM system prompt
  * `match_room_by_hint`  - deterministic fallback when the LLM gives a hint but
                            no resolvable room id

All functions are pure / dependency-free (json + stdlib) so importing this
module never pulls torch or the webapp.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

RoomLabels = Dict[str, Dict]  # region_id -> {label, floor, aliases, notes}

# Korean / English floor words -> floor label as written in labels.json ("1F"...).
# "위층"/"upstairs" and "아래층"/"downstairs" are relative — resolved separately.
_FLOOR_WORDS = {
    "1층": "1F", "일층": "1F", "1f": "1F", "first floor": "1F",
    "2층": "2F", "이층": "2F", "2f": "2F", "second floor": "2F",
    "3층": "3F", "삼층": "3F", "3f": "3F", "third floor": "3F",
}
_UP_WORDS = ("위층", "윗층", "upstairs", "위 층")
_DOWN_WORDS = ("아래층", "아랫층", "downstairs", "아래 층", "1층")


def load_room_labels(building_id: str, cache_dir) -> RoomLabels:
    """Return {region_id: {...}} for a building, or {} if no labels.json."""
    path = Path(cache_dir) / building_id / "labels.json"
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
    rooms = doc.get("rooms", {})
    return {k: v for k, v in rooms.items() if isinstance(v, dict)}


def room_directory_text(labels: RoomLabels) -> str:
    """One line per room: `room_id | floor | label | aliases`.

    Injected into the LLM prompt so it can pick `target_room` by floor/type.
    Empty string if there are no usable (labeled) rooms.
    """
    lines: List[str] = []
    for rid in sorted(labels):
        info = labels[rid]
        label = str(info.get("label", "")).strip()
        floor = str(info.get("floor", "")).strip()
        if not label and not floor:
            continue
        aliases = info.get("aliases") or []
        alias_s = ", ".join(str(a) for a in aliases)
        # Use the trailing room-code (e.g. 001_004) — that's what target_room is.
        code = _room_code(rid)
        lines.append(f"{code} | {floor} | {label} | {alias_s}")
    if not lines:
        return ""
    return "Rooms in this building (room_id | floor | type | aliases):\n" + \
        "\n".join(lines)


def match_room_by_hint(location_hint: str, target_object: str,
                       labels: RoomLabels) -> Optional[str]:
    """Best-effort room code from a free-form hint. Returns 'NNN_NNN' or None.

    Strategy: score each room by token overlap of the hint against its
    label + aliases, with a floor constraint when the hint names a floor
    (absolute "2층" or relative "위층"/"아래층"). Returns the trailing room code
    of the best-scoring room, or None if nothing matches.
    """
    if not labels:
        return None
    hint = (location_hint or "").strip().lower()
    if not hint:
        return None

    want_floor = _floor_from_hint(hint, labels)

    best_rid: Optional[str] = None
    best_score = 0
    for rid, info in labels.items():
        if want_floor is not None and str(info.get("floor", "")).strip() != want_floor:
            continue
        terms = [str(info.get("label", ""))] + \
            [str(a) for a in (info.get("aliases") or [])]
        score = 0
        for t in terms:
            t = t.strip().lower()
            if t and t in hint:
                score += 2 if t == str(info.get("label", "")).strip().lower() else 1
        if score > best_score:
            best_score = score
            best_rid = rid

    # If only a floor was named (no type match), don't guess a specific room.
    if best_score == 0:
        return None
    return _room_code(best_rid) if best_rid else None


# --------------------------------------------------------------------------- #
# internals
# --------------------------------------------------------------------------- #
def _room_code(region_id: str) -> str:
    """Trailing `NNN_NNN` of a full region id, matching `target_room`."""
    m = re.search(r"(\d{3}_\d{3})$", region_id)
    return m.group(1) if m else region_id


def _available_floors(labels: RoomLabels) -> List[str]:
    floors = {str(v.get("floor", "")).strip() for v in labels.values()}
    floors.discard("")
    # Sort by the leading integer so index 0 = lowest floor.
    return sorted(floors, key=lambda f: int(re.match(r"\d+", f).group()) if re.match(r"\d+", f) else 0)


def _floor_from_hint(hint: str, labels: RoomLabels) -> Optional[str]:
    """Resolve an absolute or relative floor word in the hint to a floor label."""
    for word, floor in _FLOOR_WORDS.items():
        if word in hint:
            return floor
    floors = _available_floors(labels)
    if not floors:
        return None
    if any(w in hint for w in _UP_WORDS):
        return floors[-1]   # highest available floor
    if any(w in hint for w in _DOWN_WORDS):
        return floors[0]    # lowest available floor
    return None
