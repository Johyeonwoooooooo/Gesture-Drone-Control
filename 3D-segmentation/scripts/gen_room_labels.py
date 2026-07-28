"""Auto-fill `cache/<building>/labels.json` from UniDet3D detections.

For each room (region) of a building this reads the precomputed detection pkl
(`cache/<building>/unidet3d/<region>.pkl`, keys `bboxes/scores/labels/classes`)
and fills the per-room label schema used by the webapp_llm LLM room resolver:

    floor   - exact, from the region-id middle field (..._000_xxx=1F, _001=2F, ...)
    notes   - top detected object classes (score-filtered, count-ranked)
    label   - HEURISTIC room type guessed from dominant classes
    aliases - Korean synonyms for the guessed label

`floor` and `notes` are exact; `label`/`aliases` are best-effort guesses meant
for a human to refine. By default only BLANK fields are filled (re-runnable,
non-destructive); pass --overwrite to replace existing values.

Usage:
    python 3D-segmentation/scripts/gen_room_labels.py --building 00800_TEEsavR23oF
    python 3D-segmentation/scripts/gen_room_labels.py --building 00809_Qpor2mEya8F
    python 3D-segmentation/scripts/gen_room_labels.py --all          # every building w/ unidet3d/
"""
from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_THIS = Path(__file__).resolve()
_DEFAULT_CACHE = _THIS.parents[1] / "cache"

_SCHEMA = {
    "label": "room-type label, free text (e.g. 'bedroom', 'kitchen', 'bathroom')",
    "floor": "floor label, free text (e.g. '1F', '2F', 'basement')",
    "aliases": "list of synonyms / Korean names — used by LLM location matching",
    "notes": "any free-form note",
}

# Room-type heuristic: first matching rule wins. Each rule = (label, trigger classes).
# Order matters — more specific / decisive rooms first.
_LABEL_RULES: List[tuple] = [
    ("bathroom", {"toilet", "sink", "bathtub", "shower", "urinal", "bidet"}),
    ("kitchen", {"refrigerator", "oven", "stove", "microwave", "dishwasher",
                 "kitchen cabinet", "range hood", "kitchen counter"}),
    ("bedroom", {"bed", "nightstand", "wardrobe", "crib"}),
    ("livingroom", {"sofa", "tv", "television", "couch", "coffee table"}),
    ("dining", {"dining table"}),
    ("stairway", {"stairs", "staircase", "step"}),
    ("office", {"desk", "monitor", "computer", "keyboard"}),
]

# Korean synonyms for each guessed label (used by LLM location matching).
_ALIASES: Dict[str, List[str]] = {
    "bathroom": ["화장실", "욕실", "변기"],
    "kitchen": ["주방", "부엌"],
    "bedroom": ["침실", "방"],
    "livingroom": ["거실"],
    "dining": ["식당", "다이닝"],
    "stairway": ["계단", "계단실"],
    "office": ["사무실", "서재"],
    "hallway": ["복도"],
    "room": ["방"],
}

# Generic structural classes that don't tell us the room type — kept out of the
# label heuristic but still allowed into notes if nothing better dominates.
_STRUCTURAL = {"wall", "floor", "ceiling", "door", "window", "light switch",
               "socket", "smoke detector", "ceiling lamp", "painting", "picture"}


def floor_label(region: str) -> str:
    """Floor from the region-id middle field: ..._000_xxx -> 1F, _001 -> 2F, ..."""
    parts = region.split("_")
    if len(parts) >= 2:
        try:
            return f"{int(parts[-2]) + 1}F"
        except ValueError:
            pass
    return ""


def detected_classes(pkl_path: Path, score_thr: float) -> Counter:
    """Counter of class-name -> count for boxes above the score threshold."""
    if not pkl_path.exists():
        return Counter()
    with open(pkl_path, "rb") as f:
        det = pickle.load(f)
    labels = np.asarray(det["labels"]).astype(int)
    scores = np.asarray(det["scores"]).astype(float)
    classes = list(det["classes"])
    keep = scores >= score_thr
    names = [classes[l] for l, k in zip(labels, keep)
             if k and 0 <= l < len(classes)]
    return Counter(names)


def guess_label(counts: Counter) -> str:
    present = set(counts)
    for label, triggers in _LABEL_RULES:
        if present & triggers:
            return label
    # No decisive object: a structural-only room is likely a hallway/corridor.
    informative = present - _STRUCTURAL
    if not informative:
        return "hallway"
    return "room"


def notes_text(counts: Counter, top_n: int = 8) -> str:
    """Comma-joined top object classes, most frequent first."""
    return ", ".join(name for name, _ in counts.most_common(top_n))


def fill_building(building_id: str, cache_dir: Path, score_thr: float,
                  overwrite: bool) -> None:
    bdir = cache_dir / building_id
    feat_dir = bdir / "feat"
    if not feat_dir.is_dir():
        raise SystemExit(f"No feat/ dir for building {building_id}: {feat_dir}")
    regions = sorted(p.name for p in feat_dir.iterdir()
                     if (p / "feat.npy").exists())
    if not regions:
        raise SystemExit(f"No regions found under {feat_dir}")

    labels_path = bdir / "labels.json"
    if labels_path.exists():
        with open(labels_path, "r", encoding="utf-8") as f:
            doc = json.load(f)
    else:
        doc = {}
    doc.setdefault("_schema", _SCHEMA)
    rooms = doc.setdefault("rooms", {})

    n_filled = 0
    for region in regions:
        counts = detected_classes(
            bdir / "unidet3d" / f"{region}.pkl", score_thr)
        guessed = guess_label(counts)
        room = rooms.setdefault(region, {
            "label": "", "floor": "", "aliases": [], "notes": ""})

        def put(key: str, value) -> None:
            nonlocal n_filled
            cur = room.get(key)
            blank = (cur in ("", None, []) or cur == [])
            if overwrite or blank:
                if room.get(key) != value:
                    n_filled += 1
                room[key] = value

        put("floor", floor_label(region))
        put("label", guessed)
        put("aliases", _ALIASES.get(guessed, _ALIASES["room"]))
        put("notes", notes_text(counts))

    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"[{building_id}] {len(regions)} rooms, {n_filled} field(s) written "
          f"-> {labels_path}")


def buildings_with_detections(cache_dir: Path) -> List[str]:
    if not cache_dir.is_dir():
        return []
    return sorted(p.name for p in cache_dir.iterdir()
                  if p.is_dir() and (p / "unidet3d").is_dir())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--building", default=None,
                    help="Building id, e.g. 00800_TEEsavR23oF")
    ap.add_argument("--all", action="store_true",
                    help="Process every building that has a unidet3d/ dir")
    ap.add_argument("--cache-dir", default=str(_DEFAULT_CACHE))
    ap.add_argument("--score-thr", type=float, default=0.30,
                    help="Min detection score for a box to count (default 0.30)")
    ap.add_argument("--overwrite", action="store_true",
                    help="Replace existing field values (default: fill blanks only)")
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    if args.all:
        targets = buildings_with_detections(cache_dir)
        if not targets:
            raise SystemExit(f"No buildings with unidet3d/ under {cache_dir}")
    elif args.building:
        targets = [args.building]
    else:
        raise SystemExit("Pass --building <id> or --all")

    for b in targets:
        fill_building(b, cache_dir, args.score_thr, args.overwrite)


if __name__ == "__main__":
    main()
