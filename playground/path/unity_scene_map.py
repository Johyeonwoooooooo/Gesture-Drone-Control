"""
Utilities for loading Unity-exported occupancy grids and converting between
Unity world coordinates and planner grid coordinates.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DRONE_PATHFINDING_DIR = REPO_ROOT / "drone_pathfinding"

import sys

if str(DRONE_PATHFINDING_DIR) not in sys.path:
    sys.path.insert(0, str(DRONE_PATHFINDING_DIR))

from maps import GridMap2D  # type: ignore  # noqa: E402


@dataclass
class UnitySceneGrid:
    map_: GridMap2D
    origin_x: float
    origin_z: float
    cell_size: float
    flight_height: float
    drone_radius: float

    def world_to_grid(self, x: float, z: float) -> Tuple[int, int]:
        gx = int(round((x - self.origin_x) / self.cell_size))
        gy = int(round((z - self.origin_z) / self.cell_size))
        return gx, gy

    def grid_to_world(self, gx: float, gy: float) -> Tuple[float, float]:
        x = self.origin_x + gx * self.cell_size
        z = self.origin_z + gy * self.cell_size
        return x, z

    def clamp_grid_point(self, point: Tuple[int, int]) -> Tuple[int, int]:
        x = max(0, min(self.map_.width - 1, point[0]))
        y = max(0, min(self.map_.height - 1, point[1]))
        return x, y


def load_unity_scene_grid(path: str | Path) -> UnitySceneGrid:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    width = int(payload["size"]["width"])
    height = int(payload["size"]["height"])
    cell_size = float(payload["cell_size"])
    origin_x = float(payload["origin"]["x"])
    origin_z = float(payload["origin"]["z"])
    flight_height = float(payload["flight_height"])
    drone_radius = float(payload.get("drone_radius", cell_size * 0.5))

    map_ = GridMap2D(width, height, resolution=cell_size)
    map_.grid = np.array(payload["grid"], dtype=np.int8)

    return UnitySceneGrid(
        map_=map_,
        origin_x=origin_x,
        origin_z=origin_z,
        cell_size=cell_size,
        flight_height=flight_height,
        drone_radius=drone_radius,
    )
