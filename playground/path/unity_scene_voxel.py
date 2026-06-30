"""
Utilities for loading Unity-exported 3D voxel maps and converting between
Unity world coordinates and planner voxel coordinates.
"""

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence, Tuple

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
        # Export samples occupancy at the voxel CENTER (origin + (g + 0.5) * voxel_size).
        # A world point falls into voxel g when origin + g*size <= coord < origin + (g+1)*size,
        # so floor (not round) is the correct inverse of grid_to_world below.
        gx = int(math.floor((x - self.origin_x) / self.voxel_size))
        gy = int(math.floor((y - self.origin_y) / self.voxel_size))
        gz = int(math.floor((z - self.origin_z) / self.voxel_size))
        return gx, gy, gz

    def grid_to_world(self, gx: float, gy: float, gz: float) -> Tuple[float, float, float]:
        # Return the voxel CENTER so waypoints align with the geometry that was
        # sampled during export (ExportVoxelMap3D samples at origin + (g + 0.5) * voxel_size).
        x = self.origin_x + (gx + 0.5) * self.voxel_size
        y = self.origin_y + (gy + 0.5) * self.voxel_size
        z = self.origin_z + (gz + 0.5) * self.voxel_size
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


@dataclass
class IntrusionReport:
    """Result of checking a world-space point sequence against the voxel map.

    This is independent of Unity physics: it answers "did these positions pass
    through cells the planner considers occupied?" directly from the voxel grid.
    """

    total_points: int
    intrusion_points: int
    intrusion_ratio: float
    min_clearance_m: float  # smallest distance from any sampled point to an occupied voxel center


def _nearest_occupied_distance(scene_voxel: "UnitySceneVoxel", world_point: Sequence[float], search_radius: int = 6) -> float:
    """Distance (meters) from a world point to the nearest occupied voxel center.

    Returns math.inf when no occupied voxel is found within search_radius cells.
    """
    cx, cy, cz = scene_voxel.world_to_grid(*world_point)
    best = math.inf
    for dx in range(-search_radius, search_radius + 1):
        for dy in range(-search_radius, search_radius + 1):
            for dz in range(-search_radius, search_radius + 1):
                cell = (cx + dx, cy + dy, cz + dz)
                if not scene_voxel.map_.is_in_bounds(cell):
                    continue
                if scene_voxel.map_.is_free(cell):
                    continue
                center = scene_voxel.grid_to_world(*cell)
                dist = math.dist(world_point, center)
                if dist < best:
                    best = dist
    return best


def check_world_points_against_voxels(
    scene_voxel: "UnitySceneVoxel",
    world_points: Sequence[Sequence[float]],
    clearance_search_radius: int = 6,
) -> IntrusionReport:
    """Count how many world points land inside occupied voxels and the min clearance.

    Reuses world_to_grid + VoxelMap3D.is_free so the verdict matches exactly what
    the A* planner treated as free/occupied space.
    """
    total = len(world_points)
    intrusions = 0
    min_clearance = math.inf

    for point in world_points:
        grid_point = scene_voxel.world_to_grid(*point)
        if scene_voxel.map_.is_in_bounds(grid_point) and not scene_voxel.map_.is_free(grid_point):
            intrusions += 1

        clearance = _nearest_occupied_distance(scene_voxel, point, search_radius=clearance_search_radius)
        if clearance < min_clearance:
            min_clearance = clearance

    ratio = (intrusions / total) if total else 0.0
    return IntrusionReport(
        total_points=total,
        intrusion_points=intrusions,
        intrusion_ratio=ratio,
        min_clearance_m=min_clearance,
    )


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
