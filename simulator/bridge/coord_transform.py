"""
Affine transform between the mosaic3d world frame and the Unity world frame.

mosaic3d frame (cache/<building>/feat/<region>/coord.npy): right-handed, Z-up,
meters — the frame every planner waypoint and candidate center lives in.

Unity frame (test.unity): left-handed, Y-up. The house glb is placed as a prefab
instance with scale (5,5,5), rotation -90deg about X and position (1.26, 0, 0),
so the two frames differ by a uniform scale, an axis permutation (mosaic z ->
Unity y), possible axis sign flips from the glTF handedness conversion, and a
translation. The exact signs/translation are picked empirically by
calibrate_transform.py against the Unity-exported voxel map and stored as JSON
in simulator/bridge/transforms/<building>.json.

p_unity = matrix @ p_mosaic + translation
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

TRANSFORMS_DIR = Path(__file__).resolve().parent / "transforms"

# Unity scene placement of the house prefab (test.unity).
DEFAULT_SCALE = 5.0
DEFAULT_TRANSLATION = (1.26, 0.0, 0.0)


@dataclass
class SimTransform:
    matrix: np.ndarray        # (3,3), includes scale
    translation: np.ndarray   # (3,)
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.matrix = np.asarray(self.matrix, dtype=np.float64).reshape(3, 3)
        self.translation = np.asarray(self.translation, dtype=np.float64).reshape(3)
        self._inv = np.linalg.inv(self.matrix)

    def mosaic_to_unity(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float64)
        return pts @ self.matrix.T + self.translation

    def unity_to_mosaic(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float64)
        return (pts - self.translation) @ self._inv.T

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "matrix": self.matrix.tolist(),
            "translation": self.translation.tolist(),
            "meta": self.meta,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "SimTransform":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            matrix=np.array(data["matrix"], dtype=np.float64),
            translation=np.array(data["translation"], dtype=np.float64),
            meta=data.get("meta", {}),
        )


def transform_path_for_building(building: str) -> Path:
    return TRANSFORMS_DIR / f"{building}.json"


def load_building_transform(building: str) -> SimTransform:
    path = transform_path_for_building(building)
    if not path.exists():
        raise FileNotFoundError(
            f"No calibrated transform for '{building}' at {path}. "
            f"Run simulator/bridge/calibrate_transform.py first."
        )
    return SimTransform.load(path)


def candidate_transforms(
    scale: float = DEFAULT_SCALE,
    translation: tuple[float, float, float] = DEFAULT_TRANSLATION,
) -> dict[str, SimTransform]:
    """Sign-flip candidates for the mosaic->Unity mapping.

    The axis permutation is fixed by the scene extents (mosaic z is the up axis
    and must map to Unity y): unity = (sx*S*mx, sy*S*mz, sz*S*my). All 8 sign
    combinations are generated; the glTF right->left handedness conversion plus
    the -90deg X prefab rotation make the signs hard to derive reliably on
    paper, so calibrate_transform.py scores every candidate against the
    Unity-exported voxel map. Nominal expectation is (+, +, -).
    """
    out: dict[str, SimTransform] = {}
    for sx in (1.0, -1.0):
        for sy in (1.0, -1.0):
            for sz in (1.0, -1.0):
                m = np.array(
                    [
                        [sx * scale, 0.0, 0.0],
                        [0.0, 0.0, sy * scale],
                        [0.0, sz * scale, 0.0],
                    ]
                )
                name = "x{}y{}z{}".format(
                    "+" if sx > 0 else "-",
                    "+" if sy > 0 else "-",
                    "+" if sz > 0 else "-",
                )
                out[name] = SimTransform(
                    matrix=m,
                    translation=np.array(translation),
                    meta={"candidate": name, "scale": scale},
                )
    return out
