"""
astar.py
========
Pure-Python A* on a :class:`~voxel_io.GridMeta` with 26-connected
neighbourhood and post-hoc path smoothing.

Design constraints
------------------
* This module must be **importable at the top level** (no ``if __name__``
  guard around function definitions) so that ``multiprocessing`` can pickle
  the worker function on all platforms.
* No global mutable state – every function is stateless.
"""

from __future__ import annotations

import heapq
import math
from typing import Optional

import numpy as np

from voxel_io import GridMeta


# ---------------------------------------------------------------------------
# 26-connected neighbourhood directions (dx, dy, dz, euclidean cost)
# ---------------------------------------------------------------------------
_DIRECTIONS: list[tuple[int, int, int, float]] = [
    (dx, dy, dz, math.sqrt(dx * dx + dy * dy + dz * dz))
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if not (dx == dy == dz == 0)
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_nearest_free(
    gm: GridMeta,
    voxel: tuple[int, int, int],
    max_radius: int = 25,
) -> Optional[tuple[int, int, int]]:
    """Return the nearest free voxel within *max_radius*, or ``None``."""
    ix, iy, iz = voxel
    if gm.at(ix, iy, iz) == 0:
        return voxel
    for radius in range(1, max_radius + 1):
        best: Optional[tuple[int, int, int]] = None
        best_d = math.inf
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    if abs(dx) != radius and abs(dy) != radius and abs(dz) != radius:
                        continue
                    c = (ix + dx, iy + dy, iz + dz)
                    if gm.at(*c) == 0:
                        d = dx * dx + dy * dy + dz * dz
                        if d < best_d:
                            best, best_d = c, d
        if best is not None:
            return best
    return None


def _line_of_sight(gm: GridMeta, a: np.ndarray, b: np.ndarray) -> bool:
    """Bresenham-style LOS check between two world-space points."""
    dist = float(np.linalg.norm(b - a))
    n = max(2, math.ceil(dist / (gm.resolution * 0.7)))
    for t in np.linspace(0.0, 1.0, n + 1):
        p = a + (b - a) * t
        if gm.at(*gm.to_voxel(p)) != 0:
            return False
    return True


def _smooth(path: list[np.ndarray], gm: GridMeta) -> list[np.ndarray]:
    """Greedy string-pulling: skip intermediate waypoints when LOS exists."""
    if len(path) < 3:
        return path
    result = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1:
            if _line_of_sight(gm, path[i], path[j]):
                break
            j -= 1
        i = j
        result.append(path[i])
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def astar(
    gm: GridMeta,
    start: np.ndarray,
    goal: np.ndarray,
    max_iter: int = 500_000,
    smooth: bool = True,
) -> tuple[Optional[list[np.ndarray]], dict]:
    """
    Run A* from *start* to *goal* on *gm*.

    Parameters
    ----------
    gm:
        Voxel grid for the current room.
    start, goal:
        World-space 3-D coordinates (numpy arrays, shape ``(3,)``).
    max_iter:
        Maximum number of node expansions before giving up.
    smooth:
        Whether to apply greedy path smoothing after finding the path.

    Returns
    -------
    path:
        Ordered list of world-space waypoints, or ``None`` on failure.
    info:
        Dictionary with diagnostic keys ``expanded``, ``success``.
    """
    sv = _find_nearest_free(gm, gm.to_voxel(start))
    ev = _find_nearest_free(gm, gm.to_voxel(goal))
    if sv is None or ev is None:
        return None, {"expanded": 0, "success": False}

    nx, ny, nz = gm.shape

    # ---- encode / decode voxel ↔ integer key --------------------------------
    def encode(v: tuple[int, int, int]) -> int:
        return v[0] * ny * nz + v[1] * nz + v[2]

    def decode(k: int) -> tuple[int, int, int]:
        ix = k // (ny * nz)
        iy = (k % (ny * nz)) // nz
        iz = k % nz
        return int(ix), int(iy), int(iz)

    def heuristic(v: tuple[int, int, int]) -> float:
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(v, ev)))

    # ---- initialise ---------------------------------------------------------
    start_key = encode(sv)
    goal_key  = encode(ev)
    came_from: dict[int, int] = {start_key: -1}
    g_score: dict[int, float] = {start_key: 0.0}
    heap: list[tuple[float, float, tuple[int, int, int]]] = [
        (heuristic(sv), 0.0, sv)
    ]
    expanded = 0

    # ---- main loop ----------------------------------------------------------
    while heap:
        _, g, cur = heapq.heappop(heap)
        key = encode(cur)

        if g > g_score.get(key, math.inf) + 1e-9:
            continue                         # stale entry

        expanded += 1
        if expanded > max_iter:
            return None, {"expanded": expanded, "success": False}

        if key == goal_key:
            # Reconstruct
            path_voxels: list[tuple[int, int, int]] = []
            k = key
            while k != -1:
                path_voxels.append(decode(k))
                k = came_from[k]
            path_voxels.reverse()
            world_path = [gm.to_world(v) for v in path_voxels]
            if smooth:
                world_path = _smooth(world_path, gm)
            return world_path, {"expanded": expanded, "success": True}

        ix, iy, iz = cur
        for dx, dy, dz, step_cost in _DIRECTIONS:
            nxt = (ix + dx, iy + dy, iz + dz)
            if gm.at(*nxt) != 0:
                continue
            nk = encode(nxt)
            ng = g + step_cost
            if ng < g_score.get(nk, math.inf):
                g_score[nk] = ng
                came_from[nk] = key
                heapq.heappush(heap, (ng + heuristic(nxt), ng, nxt))

    return None, {"expanded": expanded, "success": False}


def path_length(path: Optional[list[np.ndarray]]) -> float:
    """Euclidean arc-length of *path*, or ``inf`` for ``None``."""
    if not path or len(path) < 2:
        return math.inf
    return float(sum(np.linalg.norm(path[i] - path[i - 1]) for i in range(1, len(path))))


# ---------------------------------------------------------------------------
# Multiprocessing worker (must be picklable → module-level function)
# ---------------------------------------------------------------------------

def _worker_payload(args: tuple) -> dict:
    """
    Thin wrapper called by each worker process.

    *args* is a tuple produced by :func:`parallel_planner.build_tasks`:
    ``(room_id, gm, start_world, goal_world, max_iter, smooth)``

    Returns a result dict consumed by :func:`parallel_planner.run_parallel`.
    """
    import time

    room_id, gm, start_world, goal_world, max_iter, smooth = args
    t0 = time.perf_counter()
    path, info = astar(gm, start_world, goal_world, max_iter=max_iter, smooth=smooth)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "room_id":    room_id,
        "path":       path,
        "success":    info["success"],
        "expanded":   info["expanded"],
        "time_ms":    elapsed_ms,
        "length_m":   path_length(path),
        "waypoints":  len(path) if path else 0,
    }
