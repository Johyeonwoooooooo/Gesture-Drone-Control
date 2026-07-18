"""LitePT detection backend — precomputed closed-set detections for webapp_llm_v2.

Replaces the Mosaic3D CLIP-heatmap path. Detections come from the LitePT
pipeline (minyeong-3d branch, `litept_indoor/`): per-room semantic segmentation
(ScanNet-20) + DBSCAN instance centers, aggregated offline into one
`detections.json`. This module only READS that output — no torch, no CLIP,
no GPU.

Expected data layout (``--data-dir``, default ``<repo>/data/final_npy``)::

    detections.json                      # flat list of instances (export_json.py)
    <building>_<floor>_<room>/           # e.g. 00809_Qpor2mEya8F_000_002
        coord.npy    (N,3) float32 world meters, Z-up
        color.npy    (N,3) uint8
        normal.npy   (N,3) float32
        centers.pkl  per-room instance dict (infer_centers.py)

Each detections.json entry:
    {room, room_type, label, label_idx, center [x,y,z], n_points, room_name}
"""
from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ScanNet-20 label set produced by LitePT (index = label_idx).
SCANNET20: List[str] = [
    "wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door",
    "window", "bookshelf", "picture", "counter", "desk", "curtain",
    "refrigerator", "shower curtain", "toilet", "sink", "bathtub",
    "otherfurniture",
]
# wall/floor carry no instances — the classes a query may target.
INSTANCE_CLASSES: List[str] = [c for c in SCANNET20 if c not in ("wall", "floor")]

# Korean keyword maps, vendored from minyeong-3d:litept_indoor/query.py.
ROOM_KW_MAP: Dict[str, str] = {
    "거실": "living", "리빙룸": "living",
    "침실": "bedroom", "안방": "bedroom", "방": "bedroom", "침방": "bedroom",
    "주방": "kitchen", "부엌": "kitchen", "키친": "kitchen",
    "화장실": "bathroom", "욕실": "bathroom", "바스룸": "bathroom",
    "서재": "office", "사무실": "office", "오피스": "office",
    "식당": "dining", "다이닝": "dining", "식사": "dining",
}
LABEL_KW_MAP: Dict[str, str] = {
    "의자": "chair", "체어": "chair", "좌석": "chair",
    "테이블": "table", "탁자": "table", "식탁": "table",
    "책상": "desk", "데스크": "desk",
    "소파": "sofa", "쇼파": "sofa", "카우치": "sofa",
    "침대": "bed", "베드": "bed",
    "캐비닛": "cabinet", "서랍장": "cabinet", "장롱": "cabinet",
    "붙박이": "cabinet", "수납장": "cabinet",
    "책장": "bookshelf", "책꽂이": "bookshelf", "선반": "bookshelf",
    "문": "door", "도어": "door",
    "창문": "window", "창": "window",
    "그림": "picture", "액자": "picture",
    "카운터": "counter", "조리대": "counter",
    "커튼": "curtain", "블라인드": "curtain",
    "냉장고": "refrigerator",
    "변기": "toilet",
    "세면대": "sink", "싱크대": "sink", "싱크": "sink",
    "욕조": "bathtub",
    "가구": "otherfurniture",
}
# Common free-form (English) aliases the LLM may emit despite the closed list.
LABEL_ALIAS_MAP: Dict[str, str] = {
    "tv": "otherfurniture", "television": "otherfurniture",
    "monitor": "otherfurniture", "couch": "sofa", "fridge": "refrigerator",
    "bookcase": "bookshelf", "shelf": "bookshelf", "drawer": "cabinet",
    "wardrobe": "cabinet", "closet": "cabinet", "basin": "sink",
    "washbasin": "sink", "tub": "bathtub", "painting": "picture",
    "photo": "picture", "frame": "picture",
}
# label -> Korean, for status banners.
LABEL_KR: Dict[str, str] = {
    "cabinet": "수납장", "bed": "침대", "chair": "의자", "sofa": "소파",
    "table": "테이블", "door": "문", "window": "창문", "bookshelf": "책장",
    "picture": "액자", "counter": "카운터", "desk": "책상", "curtain": "커튼",
    "refrigerator": "냉장고", "shower curtain": "샤워커튼", "toilet": "변기",
    "sink": "세면대", "bathtub": "욕조", "otherfurniture": "기타 가구",
}
ROOM_TYPE_KR: Dict[str, str] = {
    "living": "거실", "bedroom": "침실", "kitchen": "주방",
    "bathroom": "화장실", "office": "서재", "dining": "식당",
    "unknown": "미분류",
}


@dataclass
class Detection:
    room: str            # full room dir name, e.g. 00809_Qpor2mEya8F_000_002
    room_name: str       # short id, e.g. 000_002
    room_type: str       # living / bedroom / ...
    label: str           # ScanNet class name
    center: np.ndarray   # (3,) world meters, Z-up
    n_points: int
    room_match: bool = field(default=False)  # matched the query's room filter

    @property
    def label_kr(self) -> str:
        return LABEL_KR.get(self.label, self.label)

    @property
    def room_kr(self) -> str:
        return ROOM_TYPE_KR.get(self.room_type, self.room_type)

    def describe(self) -> str:
        return (f"{self.label} ({self.label_kr}) @ {self.room_name} "
                f"[{self.room_kr}] center={np.round(self.center, 2)} "
                f"n={self.n_points}")


class LitePTBackend:
    """Query interface over the precomputed LitePT detections of one building."""

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        det_path = self.data_dir / "detections.json"
        if not det_path.exists():
            raise FileNotFoundError(
                f"No detections.json under {self.data_dir} — run the LitePT "
                "pipeline (litept_indoor/infer_centers.py + export_json.py, "
                "minyeong-3d branch) first.")
        with open(det_path, encoding="utf-8") as f:
            raw = json.load(f)
        self.detections: List[Detection] = [
            Detection(
                room=e["room"],
                room_name=e.get("room_name", e["room"][-7:]),
                room_type=e.get("room_type", "unknown"),
                label=e["label"],
                center=np.asarray(e["center"], dtype=np.float64),
                n_points=int(e["n_points"]),
            )
            for e in raw
        ]
        self.room_dirs: List[Path] = sorted(
            p for p in self.data_dir.iterdir()
            if p.is_dir() and (p / "coord.npy").exists())
        if not self.room_dirs:
            raise FileNotFoundError(
                f"No room folders with coord.npy under {self.data_dir}")
        self._points_cache: Optional[np.ndarray] = None

    # ------------------------------------------------------------- point cloud
    def load_points(self, stride: int = 4) -> np.ndarray:
        """Merged (M,3) world-coord cloud of all rooms, stride-subsampled.

        Feeds planner.voxelize; ~5M raw points over 22 rooms -> ~1.3M at
        stride 4. Cached after the first call.
        """
        if self._points_cache is None:
            parts = []
            for r in self.room_dirs:
                c = np.load(r / "coord.npy", mmap_mode="r")
                parts.append(np.asarray(c[::max(1, stride)], dtype=np.float64))
            self._points_cache = np.concatenate(parts, axis=0)
        return self._points_cache

    # ---------------------------------------------------------------- matching
    def resolve_label(self, target_object: str, raw_query: str) -> Optional[str]:
        """ScanNet class for the query, or None.

        Priority: LLM target_object if on-list -> English alias map ->
        Korean keyword scan of the raw query (longest keyword first).
        """
        t = (target_object or "").strip().lower()
        if t in INSTANCE_CLASSES:
            return t
        if t in LABEL_ALIAS_MAP:
            return LABEL_ALIAS_MAP[t]
        for kw, cls in sorted(LABEL_KW_MAP.items(), key=lambda x: -len(x[0])):
            if kw in raw_query:
                return cls
        return None

    def resolve_room(self, target_room: Optional[str], location_hint: str,
                     raw_query: str) -> Tuple[Optional[str], Optional[str]]:
        """(room_name, room_type) filter for the query; both may be None.

        target_room from the LLM ("000_002") matches room_name exactly;
        otherwise a Korean room-type keyword in the hint/query maps via
        ROOM_KW_MAP.
        """
        names = {d.room_name for d in self.detections}
        if target_room and target_room in names:
            return target_room, None
        text = f"{location_hint or ''} {raw_query}"
        for kw, rt in sorted(ROOM_KW_MAP.items(), key=lambda x: -len(x[0])):
            if kw in text:
                return None, rt
        return None, None

    def candidates(self, label: str, room_name: Optional[str] = None,
                   room_type: Optional[str] = None) -> List[Detection]:
        """Ranked candidates for a class, room-filtered when possible.

        Room-matching instances first, then by n_points (bigger = more
        salient). A room filter with zero hits falls back to the whole
        building (room_match stays False so the caller can warn).
        """
        pool = [d for d in self.detections if d.label == label]
        for d in pool:
            d.room_match = bool(
                (room_name and d.room_name == room_name)
                or (room_type and d.room_type == room_type))
        if (room_name or room_type) and any(d.room_match for d in pool):
            pool = [d for d in pool if d.room_match] + \
                   [d for d in pool if not d.room_match]
            pool.sort(key=lambda d: (not d.room_match, -d.n_points))
        else:
            pool.sort(key=lambda d: -d.n_points)
        return pool

    # ------------------------------------------------------------------- home
    def default_home(self) -> np.ndarray:
        """Launch point: first room's xy centroid, 1 m above its floor."""
        r0 = self.room_dirs[0]
        coord = np.load(r0 / "coord.npy", mmap_mode="r")
        sample = np.asarray(coord[::50], dtype=np.float64)
        xy = sample[:, :2].mean(axis=0)
        floor_z = None
        pkl = r0 / "centers.pkl"
        if pkl.exists():
            try:
                with open(pkl, "rb") as f:
                    ck = pickle.load(f)
                labels = np.asarray(ck["pred_labels"])
                fl = np.asarray(ck["coord"])[labels == 1]  # 1 = floor
                if len(fl):
                    floor_z = float(np.median(fl[:, 2]))
            except Exception:
                floor_z = None
        if floor_z is None:
            floor_z = float(np.percentile(sample[:, 2], 2.0))
        return np.array([xy[0], xy[1], floor_z + 1.0], dtype=np.float64)

    # -------------------------------------------------------- LLM room context
    def room_directory_text(self) -> str:
        """Room directory string for the LLM prompt (replaces room_labels.py)."""
        rooms: Dict[str, str] = {}
        for d in self.detections:
            rooms.setdefault(d.room_name, d.room_type)
        lines = ["Room directory (room_id: floor, type):"]
        for name in sorted(rooms):
            floor = name.split("_")[0]
            rt = rooms[name]
            kr = ROOM_TYPE_KR.get(rt, rt)
            lines.append(f"- {name}: floor {int(floor)}, {rt} ({kr})")
        return "\n".join(lines)


# --------------------------------------------------------------- self-test CLI
def _main() -> None:
    ap = argparse.ArgumentParser(description="LitePT backend self-test")
    ap.add_argument("query", nargs="?", default="거실 소파")
    ap.add_argument("--data-dir",
                    default=str(Path(__file__).resolve().parents[2]
                                / "data" / "final_npy"))
    ap.add_argument("--target-object", default="",
                    help="Simulated LLM target_object (else keyword-only).")
    ap.add_argument("--target-room", default=None)
    ap.add_argument("--points", action="store_true",
                    help="Also load + report the merged point cloud.")
    args = ap.parse_args()

    be = LitePTBackend(args.data_dir)
    print(f"loaded {len(be.detections)} detections, {len(be.room_dirs)} rooms")
    label = be.resolve_label(args.target_object, args.query)
    room_name, room_type = be.resolve_room(args.target_room, "", args.query)
    print(f"query={args.query!r} -> label={label} "
          f"room_name={room_name} room_type={room_type}")
    if label is None:
        print("no label resolved")
        return
    cands = be.candidates(label, room_name, room_type)
    for i, d in enumerate(cands[:10]):
        star = "*" if d.room_match else " "
        print(f"  [{i+1}]{star} {d.describe()}")
    print(f"home = {np.round(be.default_home(), 2)}")
    if args.points:
        pts = be.load_points()
        print(f"points: {pts.shape} min={pts.min(0).round(2)} "
              f"max={pts.max(0).round(2)}")


if __name__ == "__main__":
    _main()
