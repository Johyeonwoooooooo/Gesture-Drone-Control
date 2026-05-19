"""
parallel_planner.py
===================
Dispatch per-room A* jobs to a ``multiprocessing.Pool`` and stitch the
resulting path segments into a single world-space trajectory.

Architecture
------------
::

    room_sequence: [A, B, C, D]
    door_waypoints: {(A,B): p1, (B,C): p2, (C,D): p3}

    Task 0:  A* in room A  from start          → p1
    Task 1:  A* in room B  from p1             → p2
    Task 2:  A* in room C  from p2             → p3
    Task 3:  A* in room D  from p3             → goal

    All tasks submitted to Pool.map() simultaneously (true parallelism).
    Segments concatenated preserving the door waypoints.
"""

from __future__ import annotations

import os
import time
from multiprocessing import Pool
from typing import Optional

import numpy as np

from astar import _find_nearest_free, _worker_payload, path_length
from config import DOOR_WAYPOINTS, WORKER_COUNT
from voxel_io import GridMeta

# parallel_planner.py 상단에
_pool: Pool | None = None

def get_pool(n_workers: int) -> Pool:
    global _pool
    if _pool is None:
        _pool = Pool(processes=n_workers)
    return _pool
# ---------------------------------------------------------------------------
# Task construction
# ---------------------------------------------------------------------------

def _clamp_to_grid(gm: GridMeta, point: np.ndarray) -> np.ndarray:
    """
    If *point* lies outside *gm*'s world-space bounds, project it to the
    nearest grid centroid so A* always receives a valid coordinate.
    """
    lo = gm.origin
    hi = gm.origin + np.array(gm.shape) * gm.resolution
    clamped = np.clip(point, lo + gm.resolution, hi - gm.resolution)
    return clamped


def build_tasks(
    room_sequence: list[str],
    grids: dict[str, GridMeta],
    start_world: np.ndarray,
    goal_world: np.ndarray,
    door_waypoints: dict[frozenset, tuple[float, float, float]] | None = None,
    max_iter: int = 500_000,
    smooth: bool = True,
) -> list[tuple]:
    """
    Build the argument tuples for each per-room A* worker.

    Parameters
    ----------
    room_sequence:
        Ordered list ``[start_room, …, goal_room]`` from DFS/BFS.
    grids:
        Pre-loaded ``{room_id: GridMeta}`` mapping.
    start_world, goal_world:
        Global start and goal in world coordinates.
    door_waypoints:
        Overrides ``config.DOOR_WAYPOINTS`` when provided.
    """
    if door_waypoints is None:
        door_waypoints = DOOR_WAYPOINTS

    n = len(room_sequence)
    tasks: list[tuple] = []

    for i, room_id in enumerate(room_sequence):
        gm = grids[room_id]

        # Determine segment start
        if i == 0:
            seg_start = start_world.copy()
        else:
            prev_room = room_sequence[i - 1]
            key = frozenset({prev_room, room_id})
            if key in door_waypoints:
                seg_start = np.array(door_waypoints[key], dtype=float)
            else:
                # Fallback: centroid of the current room's grid
                seg_start = gm.origin + np.array(gm.shape) * gm.resolution / 2.0

        # Determine segment goal
        if i == n - 1:
            seg_goal = goal_world.copy()
        else:
            next_room = room_sequence[i + 1]
            key = frozenset({room_id, next_room})
            if key in door_waypoints:
                seg_goal = np.array(door_waypoints[key], dtype=float)
            else:
                seg_goal = gm.origin + np.array(gm.shape) * gm.resolution / 2.0

        tasks.append((room_id, gm, _clamp_to_grid(gm, seg_start), _clamp_to_grid(gm, seg_goal), max_iter, smooth))

    return tasks


# ---------------------------------------------------------------------------
# Sequential baseline (for benchmarking)
# ---------------------------------------------------------------------------

def run_sequential(tasks: list[tuple]) -> tuple[list[dict], float]:
    """
    Execute A* tasks one-by-one in the calling process.

    Returns
    -------
    results:
        List of per-room result dicts (same schema as parallel).
    wall_time_ms:
        Total wall-clock time in milliseconds.
    """
    t0 = time.perf_counter()
    results = [_worker_payload(t) for t in tasks]
    wall_ms = (time.perf_counter() - t0) * 1000.0
    return results, wall_ms


# ---------------------------------------------------------------------------
# Parallel execution
# ---------------------------------------------------------------------------

def run_parallel(tasks, n_workers=None):
    workers = n_workers or os.cpu_count()
    workers = min(workers, len(tasks))

    pool = get_pool(workers)  # 새로 만들지 않음

    t0 = time.perf_counter()
    results = pool.map(_worker_payload, tasks)  # with 블록 없음
    wall_ms = (time.perf_counter() - t0) * 1000.0
    return results, wall_ms

# ---------------------------------------------------------------------------
# Path stitching
# ---------------------------------------------------------------------------

def stitch_paths(
    results: list[dict],
    room_sequence: list[str],
) -> Optional[list[np.ndarray]]:
    """
    Concatenate per-room path segments into one continuous trajectory.

    The last waypoint of segment *i* and the first waypoint of segment
    *i+1* refer to the same door location, so duplicates are removed at
    the join.

    Returns ``None`` if any segment failed.
    """
    full_path: list[np.ndarray] = []
    for i, res in enumerate(results):
        if not res["success"] or res["path"] is None:
            return None
        seg: list[np.ndarray] = res["path"]
        if i == 0:
            full_path.extend(seg)
        else:
            # Drop the duplicate waypoint at the room boundary
            full_path.extend(seg[1:])
    return full_path if full_path else None


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def aggregate_stats(
    results: list[dict],
    wall_time_ms: float,
    mode: str,
) -> dict:
    """Compute aggregate statistics across all per-room results."""
    successful = [r for r in results if r["success"]]
    return {
        "mode":           mode,
        "total_rooms":    len(results),
        "success_rooms":  len(successful),
        "wall_time_ms":   wall_time_ms,
        "sum_time_ms":    sum(r["time_ms"] for r in results),
        "total_length_m": sum(r["length_m"] for r in successful),
        "total_expanded": sum(r["expanded"] for r in results),
        "total_waypoints": sum(r["waypoints"] for r in successful),
    }


# parallel_planner.py에 추가

def should_parallelize(tasks: list[tuple]) -> bool:
    """
    병렬화 가치가 있는지 판단.
    - 세그먼트가 2개 이하면 오버헤드만 남음
    - 가장 큰 그리드가 전체의 80% 이상이면 load imbalance
    """
    if len(tasks) < 3:
        return False

    # 각 task에서 GridMeta 꺼내기 (args[1])
    sizes = [t[1].grid.size for t in tasks]  # voxel 수
    if max(sizes) / sum(sizes) > 0.8:
        return False  # 한 방이 너무 지배적

    return True

def run_adaptive(tasks, n_workers=None):
    """
    Automatically choose sequential or parallel execution.

    Returns
    -------
    results : list
    wall_ms : float
    mode : str
    """

    if should_parallelize(tasks):

        results, wall_ms = run_parallel(
            tasks,
            n_workers=n_workers,
        )

        mode = "parallel"

    else:

        results, wall_ms = run_sequential(tasks)

        mode = "sequential"

    return results, wall_ms, mode

