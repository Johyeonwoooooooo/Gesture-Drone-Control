"""Per-room geometry index — the unit a PATROL mission plans over.

The object-finding path (`litept_backend.py`) only ever needs one detection
center. A patrol needs the ROOM itself: where its middle is, how high the floor
is, how big it is, and what a human calls it ("현우방", "2층 화장실").

Everything here is derived from the same `--data-dir` layout the backend uses::

    <building>_<floor>_<room>/coord.npy    (N,3) world meters, Z-up
    <building>_<floor>_<room>/centers.pkl  pred_labels (1 = floor)
    detections.json                        room_name -> room_type

The centroid/floor-z math is the generalization of
`litept_backend.default_home()` (which now calls into here) to every room.

Floor numbering: the dir suffix `000_002` means floor index 0. Korean floor
labels are `floor + floor_offset` (default 1), i.e. floor 0 == "1층". Override
`floor_offset` in room_aliases.json if a building is numbered differently.

Self-test::

    python -m patrol.room_index --list          # cwd = repo root
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

try:
    from patrol import planner
    from patrol.litept_backend import ROOM_TYPE_KR
except ImportError:  # running as a plain script from patrol/
    import planner  # type: ignore
    from litept_backend import ROOM_TYPE_KR  # type: ignore

INDEX_VERSION = 3  # bump when the cached fields change


@dataclass
class RoomInfo:
    room_name: str            # "002_011"
    room_dir: str             # "00809_Qpor2mEya8F_002_011"
    room_type: str            # living / bedroom / bathroom / ...
    floor: int                # 0-based floor index from the dir suffix
    centroid: np.ndarray      # (3,) xy centroid, z = median of all points
    floor_z: float            # world z of the floor plane (meters)
    bbox_min: np.ndarray      # (3,)
    bbox_max: np.ndarray      # (3,)
    n_points: int
    n_detections: int = 0
    aliases: List[str] = field(default_factory=list)
    floor_offset: int = 1     # floor + offset = the Korean 층 number

    # ------------------------------------------------------------------ labels
    @property
    def type_kr(self) -> str:
        return ROOM_TYPE_KR.get(self.room_type, self.room_type)

    @property
    def floor_kr(self) -> str:
        return f"{self.floor + self.floor_offset}층"

    @property
    def display(self) -> str:
        """'2층 침실 002_011 (현우방)' — used in banners and the report."""
        base = f"{self.floor_kr} {self.type_kr} {self.room_name}"
        return f"{base} ({self.aliases[0]})" if self.aliases else base

    @property
    def size_xy(self) -> np.ndarray:
        return (self.bbox_max - self.bbox_min)[:2]

    # ------------------------------------------------------------ (de)serialize
    def to_json(self) -> dict:
        return {
            "room_name": self.room_name, "room_dir": self.room_dir,
            "room_type": self.room_type, "floor": self.floor,
            "centroid": self.centroid.tolist(), "floor_z": self.floor_z,
            "bbox_min": self.bbox_min.tolist(), "bbox_max": self.bbox_max.tolist(),
            "n_points": self.n_points, "n_detections": self.n_detections,
            "aliases": list(self.aliases), "floor_offset": self.floor_offset,
        }

    @staticmethod
    def from_json(d: dict) -> "RoomInfo":
        return RoomInfo(
            room_name=d["room_name"], room_dir=d["room_dir"],
            room_type=d["room_type"], floor=int(d["floor"]),
            centroid=np.asarray(d["centroid"], dtype=float),
            floor_z=float(d["floor_z"]),
            bbox_min=np.asarray(d["bbox_min"], dtype=float),
            bbox_max=np.asarray(d["bbox_max"], dtype=float),
            n_points=int(d["n_points"]), n_detections=int(d.get("n_detections", 0)),
            aliases=list(d.get("aliases", [])),
            floor_offset=int(d.get("floor_offset", 1)),
        )


# ---------------------------------------------------------------------- aliases

def load_aliases(path: Optional[str | Path]) -> tuple[Dict[str, List[str]], int]:
    """(room_name -> [alias, ...], floor_offset) from room_aliases.json.

    Missing file is fine — patrols then address rooms by code/type/floor only.
    """
    if path is None:
        path = Path(__file__).resolve().parent / "room_aliases.json"
    path = Path(path)
    if not path.exists():
        return {}, 1
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    raw = data.get("aliases", {}) or {}
    aliases = {str(k): [str(a) for a in (v if isinstance(v, list) else [v])]
               for k, v in raw.items()}
    return aliases, int(data.get("floor_offset", 1))


# ------------------------------------------------------------------ room geometry

def _floor_z_of(room_dir: Path, sample: np.ndarray) -> float:
    """World z of the floor plane: median of the floor-labelled points from
    centers.pkl, sanity-checked against the room's own z range.

    The LitePT segmentation is not always right about which plane is the floor —
    in 00809 room 002_012 it labels the CEILING as floor (median z 5.53 while
    the room spans 2.74‥5.61). Hovering at `floor_z + hover_height` there would
    put the drone above the roof, so a pkl estimate that is not near the bottom
    of the room is rejected in favour of the geometric 2nd percentile.
    """
    bottom = float(np.percentile(sample[:, 2], 2.0))
    pkl = room_dir / "centers.pkl"
    if pkl.exists():
        try:
            with open(pkl, "rb") as f:
                ck = pickle.load(f)
            labels = np.asarray(ck["pred_labels"])
            floor_pts = np.asarray(ck["coord"])[labels == 1]  # 1 = floor
            if len(floor_pts):
                z = float(np.median(floor_pts[:, 2]))
                if bottom - 0.5 <= z <= bottom + 1.0:
                    return z
        except Exception:
            pass
    return bottom


def _room_from_dir(room_dir: Path, room_type: str, n_detections: int,
                   aliases: Sequence[str], floor_offset: int,
                   stride: int) -> RoomInfo:
    coord = np.load(room_dir / "coord.npy", mmap_mode="r")
    sample = np.asarray(coord[::max(1, stride)], dtype=np.float64)
    m = re.search(r"(\d+)_(\d+)$", room_dir.name)
    room_name = f"{int(m.group(1)):03d}_{int(m.group(2)):03d}" if m else room_dir.name
    floor = int(m.group(1)) if m else 0
    centroid = np.array([sample[:, 0].mean(), sample[:, 1].mean(),
                         np.median(sample[:, 2])], dtype=float)
    return RoomInfo(
        room_name=room_name, room_dir=room_dir.name, room_type=room_type,
        floor=floor, centroid=centroid, floor_z=_floor_z_of(room_dir, sample),
        bbox_min=sample.min(axis=0), bbox_max=sample.max(axis=0),
        n_points=int(coord.shape[0]), n_detections=n_detections,
        aliases=list(aliases), floor_offset=floor_offset,
    )


def build_room_index(backend, aliases_path: Optional[str | Path] = None,
                     cache_path: Optional[str | Path] = None,
                     stride: int = 50,
                     refresh: bool = False) -> Dict[str, RoomInfo]:
    """room_name -> RoomInfo for every room folder the backend found.

    Reading 22 rooms' coord.npy takes a few seconds even mmap'd + strided, so
    the result is cached as JSON (default `out/room_index.json`). Aliases are
    re-applied on every load, so editing room_aliases.json does NOT require
    deleting the cache.
    """
    aliases, floor_offset = load_aliases(aliases_path)
    if cache_path is None:
        cache_path = Path(__file__).resolve().parent / "out" / "room_index.json"
    cache_path = Path(cache_path)

    types: Dict[str, str] = {}
    counts: Dict[str, int] = {}
    for d in backend.detections:
        types.setdefault(d.room_name, d.room_type)
        counts[d.room_name] = counts.get(d.room_name, 0) + 1

    index: Dict[str, RoomInfo] = {}
    if cache_path.exists() and not refresh:
        try:
            with open(cache_path, encoding="utf-8") as f:
                blob = json.load(f)
            if (blob.get("version") == INDEX_VERSION
                    and blob.get("data_dir") == str(backend.data_dir)
                    and len(blob.get("rooms", [])) == len(backend.room_dirs)):
                index = {r["room_name"]: RoomInfo.from_json(r)
                         for r in blob["rooms"]}
        except Exception:
            index = {}

    if not index:
        for room_dir in backend.room_dirs:
            info = _room_from_dir(
                room_dir, "unknown", 0, [], floor_offset, stride)
            index[info.room_name] = info
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump({"version": INDEX_VERSION,
                           "data_dir": str(backend.data_dir),
                           "rooms": [r.to_json() for r in index.values()]},
                          f, ensure_ascii=False, indent=1)
        except OSError:
            pass  # cache is an optimization, never a hard requirement

    # Re-apply the mutable, cheap fields after a cache hit.
    for name, info in index.items():
        info.room_type = types.get(name, "unknown")
        info.n_detections = counts.get(name, 0)
        info.aliases = aliases.get(name, [])
        info.floor_offset = floor_offset
    return index


# ------------------------------------------------------------------- scan poses

def scan_pose(room: RoomInfo, hover_height: float,
              gm: Optional[planner.GridMeta] = None) -> np.ndarray:
    """World-meter hover point where the drone performs its 360° room scan.

    Ideally the room's xy centroid at `floor_z + hover_height`. That voxel is
    often occupied (a bed/table sits dead-center, or the centroid of an L-shaped
    room lands in a wall), so fall back to the nearest FREE point still inside
    the room's own xy bbox before resorting to planner.find_nearest_free — a
    blind nearest-free search can escape into the neighbouring room.
    """
    base = np.array([room.centroid[0], room.centroid[1],
                     room.floor_z + hover_height], dtype=float)
    if gm is None:
        return base
    if planner.grid_at(gm, *planner.world_to_voxel(gm, base)) == 0:
        return base

    step = max(0.3, gm.resolution * 3.0)
    half = np.maximum(room.size_xy * 0.5 - step, step)
    n = int(min(12, max(2, np.floor(half.max() / step))))
    offsets = []
    for ring in range(1, n + 1):
        r = ring * step
        for ang in range(0, 360, 30):
            a = np.radians(ang)
            offsets.append((r * np.cos(a), r * np.sin(a)))
    for dz in (0.0, 0.3, -0.3, 0.6, 0.9):
        for dx, dy in offsets:
            if abs(dx) > half[0] or abs(dy) > half[1]:
                continue
            p = base + np.array([dx, dy, dz])
            if planner.grid_at(gm, *planner.world_to_voxel(gm, p)) == 0:
                return p

    v = planner.find_nearest_free(gm, planner.world_to_voxel(gm, base))
    return planner.voxel_to_world(gm, v) if v is not None else base


def order_rooms(rooms: Sequence[RoomInfo], start_world) -> List[RoomInfo]:
    """Greedy nearest-neighbour visit order from the drone's current position.

    Floor is weighted heavily so a multi-floor patrol finishes one floor before
    climbing — going up and down repeatedly is both slower and harder to watch.
    """
    remaining = list(rooms)
    pos = np.asarray(start_world, dtype=float)
    ordered: List[RoomInfo] = []
    while remaining:
        nxt = min(remaining, key=lambda r: (r.floor,
                                            float(np.linalg.norm(r.centroid - pos))))
        ordered.append(nxt)
        remaining.remove(nxt)
        pos = nxt.centroid
    return ordered


def default_home(index: Dict[str, RoomInfo], hover_height: float = 1.0):
    """Launch point: the lowest-floor, most-detected room, hovering above its
    floor. Replaces the `room_dirs[0]` heuristic in litept_backend."""
    if not index:
        return None
    room = min(index.values(), key=lambda r: (r.floor, -r.n_detections, r.room_name))
    return np.array([room.centroid[0], room.centroid[1],
                     room.floor_z + hover_height], dtype=float), room


# ------------------------------------------------------------- web room ids

def web_room_id(room_name: str) -> str:
    """Our room id in the web console's spelling: "002_012" -> "012".

    The web floor plan (web/uploads/web_meta.json) keys rooms by the bare
    3-digit suffix and carries the floor separately, while we keep the floor in
    the id. Same rooms, same numbers — only the prefix differs, so this is a
    spelling change and not a mapping table.
    """
    return room_name.rsplit("_", 1)[-1]


def by_web_room_id(index: Dict[str, RoomInfo], web_id: str) -> Optional[RoomInfo]:
    """Look a room up by the web console's id ("012")."""
    want = str(web_id).strip().lstrip("0") or "0"
    for name, info in index.items():
        if web_room_id(name).lstrip("0") == want:
            return info
    return None


# ------------------------------------------------------------- LLM prompt text

def room_directory_text(index: Dict[str, RoomInfo]) -> str:
    """Room directory for the LLM system prompt (patrol + object find).

    Extends litept_backend.room_directory_text with the Korean floor label and
    the human aliases, which is what lets "현우방만 탐색해줘" resolve at all.
    """
    lines = ["Room directory (room_id: floor, type, aliases):"]
    for name in sorted(index):
        r = index[name]
        line = f"- {name}: {r.floor_kr} (floor {r.floor}), {r.room_type} ({r.type_kr})"
        if r.aliases:
            line += f", 별칭: {', '.join(r.aliases)}"
        lines.append(line)
    return "\n".join(lines)


# --------------------------------------------------------------- self-test CLI
def _main() -> None:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from patrol.litept_backend import LitePTBackend

    ap = argparse.ArgumentParser(description="Room index self-test")
    ap.add_argument("--data-dir",
                    default=str(Path(__file__).resolve().parents[1]
                                / "data" / "final_npy"))
    ap.add_argument("--aliases", default=None)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--directory", action="store_true",
                    help="Print the LLM prompt block.")
    args = ap.parse_args()

    be = LitePTBackend(args.data_dir)
    index = build_room_index(be, args.aliases, refresh=args.refresh)
    print(f"{len(index)} rooms")
    if args.list or not args.directory:
        for name in sorted(index):
            r = index[name]
            print(f"  {name}  {r.floor_kr:>3} {r.type_kr:<5} "
                  f"centroid={np.round(r.centroid, 2)} floor_z={r.floor_z:6.2f} "
                  f"size={np.round(r.size_xy, 1)} n={r.n_points} "
                  f"det={r.n_detections}"
                  + (f"  별칭={r.aliases}" if r.aliases else ""))
    if args.directory:
        print(room_directory_text(index))
    home = default_home(index)
    if home is not None:
        print(f"home = {np.round(home[0], 2)}  ({home[1].display})")


if __name__ == "__main__":
    _main()
