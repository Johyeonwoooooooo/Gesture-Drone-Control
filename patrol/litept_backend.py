"""LitePT detection backend — precomputed closed-set detections for patrol.

Detections come from the LitePT
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
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

# ScanNet-20 label set produced by LitePT (index = label_idx).
SCANNET20: List[str] = [
    "wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door",
    "window", "bookshelf", "picture", "counter", "desk", "curtain",
    "refrigerator", "shower curtain", "toilet", "sink", "bathtub",
    "otherfurniture",
]
# wall/floor carry no instances — the classes a query may target.
# Korean keyword maps, vendored from minyeong-3d:litept_indoor/query.py.
ROOM_KW_MAP: Dict[str, str] = {
    "거실": "living", "리빙룸": "living",
    "침실": "bedroom", "안방": "bedroom", "방": "bedroom", "침방": "bedroom",
    "주방": "kitchen", "부엌": "kitchen", "키친": "kitchen",
    "화장실": "bathroom", "욕실": "bathroom", "바스룸": "bathroom",
    "서재": "office", "사무실": "office", "오피스": "office",
    "식당": "dining", "다이닝": "dining", "식사": "dining",
}
# label -> Korean, for the room summary.
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

    @property
    def label_kr(self) -> str:
        return LABEL_KR.get(self.label, self.label)

    @property
    def room_kr(self) -> str:
        return ROOM_TYPE_KR.get(self.room_type, self.room_type)


class LitePTBackend:
    """Reader for one building's precomputed LitePT detections.

    Loads `detections.json` (what is where) and the per-room point clouds (what
    the planner treats as obstacles). Nothing here searches for objects — the
    pipeline patrols areas, and the room index is built on top of this in
    `room_index.py`.
    """

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


# --------------------------------------------------------------- self-test CLI
def _main() -> None:
    ap = argparse.ArgumentParser(description="LitePT backend self-test")
    ap.add_argument("--data-dir",
                    default=str(Path(__file__).resolve().parents[1]
                                / "data" / "final_npy"))
    ap.add_argument("--points", action="store_true",
                    help="Also load + report the merged point cloud.")
    args = ap.parse_args()

    be = LitePTBackend(args.data_dir)
    print(f"loaded {len(be.detections)} detections, {len(be.room_dirs)} rooms")
    by_room: Dict[str, int] = {}
    for d in be.detections:
        by_room[d.room_name] = by_room.get(d.room_name, 0) + 1
    for name in sorted(by_room):
        rt = next(d.room_type for d in be.detections if d.room_name == name)
        print(f"  {name}: {by_room[name]:3d} instances, {rt}")
    print(f"home = {np.round(be.default_home(), 2)}")
    if args.points:
        pts = be.load_points()
        print(f"points: {pts.shape} min={pts.min(0).round(2)} "
              f"max={pts.max(0).round(2)}")


if __name__ == "__main__":
    _main()
