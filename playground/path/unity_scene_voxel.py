"""
Utilities for loading Unity-exported 3D voxel maps and converting between
Unity world coordinates and planner voxel coordinates.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DRONE_PATHFINDING_DIR = REPO_ROOT / "drone_pathfinding"

import sys

if str(DRONE_PATHFINDING_DIR) not in sys.path:
    sys.path.insert(0, str(DRONE_PATHFINDING_DIR))

from maps import VoxelMap3D  # type: ignore  # noqa: E402


VoxelPoint = Tuple[int, int, int]


@dataclass
class UnitySceneVoxel:
    map_: VoxelMap3D
    origin_x: float
    origin_y: float
    origin_z: float
    voxel_size: float
    drone_radius: float

    def world_to_grid(self, x: float, y: float, z: float) -> VoxelPoint:
        gx = int(round((x - self.origin_x) / self.voxel_size))
        gy = int(round((y - self.origin_y) / self.voxel_size))
        gz = int(round((z - self.origin_z) / self.voxel_size))
        return gx, gy, gz

    def grid_to_world(self, gx: float, gy: float, gz: float) -> Tuple[float, float, float]:
        x = self.origin_x + gx * self.voxel_size
        y = self.origin_y + gy * self.voxel_size
        z = self.origin_z + gz * self.voxel_size
        return x, y, z

    def clamp_grid_point(self, point: VoxelPoint) -> VoxelPoint:
        x = max(0, min(self.map_.width - 1, point[0]))
        y = max(0, min(self.map_.height - 1, point[1]))
        z = max(0, min(self.map_.depth - 1, point[2]))
        return x, y, z

    def find_nearest_free(self, point: VoxelPoint, max_radius: int = 12) -> VoxelPoint | None:
        point = self.clamp_grid_point(point)
        if self.map_.is_free(point):
            return point

        queue = deque([point])
        visited = {point}
        while queue:
            current = queue.popleft()
            if self.map_.is_free(current):
                return current

            if max(
                abs(current[0] - point[0]),
                abs(current[1] - point[1]),
                abs(current[2] - point[2]),
            ) >= max_radius:
                continue

            for neighbor in self.map_.get_neighbors(current):
                neighbor = (int(neighbor[0]), int(neighbor[1]), int(neighbor[2]))
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return None


def _mark_occupied(map_: VoxelMap3D, occupied: Iterable[Iterable[int]]) -> None:
    for cell in occupied:
        x, y, z = int(cell[0]), int(cell[1]), int(cell[2])
        if 0 <= x < map_.width and 0 <= y < map_.height and 0 <= z < map_.depth:
            map_.grid[z, y, x] = 1


def load_unity_scene_voxel(path: str | Path) -> UnitySceneVoxel:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    width = int(payload["size"]["width"])
    height = int(payload["size"]["height"])
    depth = int(payload["size"]["depth"])
    voxel_size = float(payload["voxel_size"])
    origin_x = float(payload["origin"]["x"])
    origin_y = float(payload["origin"]["y"])
    origin_z = float(payload["origin"]["z"])
    drone_radius = float(payload.get("drone_radius", voxel_size * 0.5))

    map_ = VoxelMap3D(width, height, depth, resolution=voxel_size)
    _mark_occupied(map_, payload["occupied"])

    return UnitySceneVoxel(
        map_=map_,
        origin_x=origin_x,
        origin_y=origin_y,
        origin_z=origin_z,
        voxel_size=voxel_size,
        drone_radius=drone_radius,
    )
