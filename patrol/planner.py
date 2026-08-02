"""3D voxel A* / RRT* path planner (pure numpy).

Vendored and trimmed from the `chaewon` branch (`comparison/3D.py`) — only the
self-contained planning core, without its macOS-hardcoded input path, PLY
loaders, matplotlib plotting, benchmarking or CLI `main()`. The one substantive
change: the obstacle-inflation ("margin") step uses a vectorized
`scipy.ndimage.binary_dilation` instead of the original per-occupied-voxel Python
loop, which is far too slow at whole-building scale.

Everything works in a single WORLD-meter frame: `voxelize` builds an occupancy
grid from `(N,3)` world points (obstacle = every point); `astar` / `rrt_star`
take world-meter start/goal and return an ordered list of world-meter
`np.ndarray` waypoints (or `None`). The webapp feeds `asset.coord + asset.center`
(recovered world coords) straight in, and renders the returned waypoints after
subtracting `asset.center`.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class GridMeta:
    grid: np.ndarray  # (Nx,Ny,Nz) uint8; 0=free, 1=occupied
    x_min: float
    y_min: float
    z_min: float
    resolution: float

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.grid.shape

    @property
    def free_ratio(self) -> float:
        return float(np.mean(self.grid == 0))


# ---------------------------------------------------------------------------
# Voxelization
# ---------------------------------------------------------------------------

def voxelize(points: np.ndarray, resolution: float, margin: int, sample: int) -> GridMeta:
    """Occupancy grid from world-meter points. `margin` inflates obstacles by
    that many voxels (drone clearance); `sample` keeps every Nth point."""
    sampled = points[:: max(1, sample)]
    xyz_min = sampled.min(axis=0)
    xyz_max = sampled.max(axis=0)
    shape = np.ceil((xyz_max - xyz_min) / resolution).astype(int) + 3
    grid = np.zeros(tuple(shape), dtype=np.uint8)

    ijk = np.rint((sampled - xyz_min) / resolution).astype(int)
    valid = np.all((ijk >= 0) & (ijk < shape), axis=1)
    grid[ijk[valid, 0], ijk[valid, 1], ijk[valid, 2]] = 1

    if margin > 0:
        # Vectorized cubic dilation (replaces the original O(occupied) loop).
        try:
            from scipy.ndimage import binary_dilation

            struct = np.ones((2 * margin + 1,) * 3, dtype=bool)
            grid = binary_dilation(grid.astype(bool), structure=struct).astype(np.uint8)
        except Exception:
            # Fallback: separable max-pool dilation, no scipy dependency.
            occ = grid.astype(bool)
            for axis in range(3):
                acc = occ.copy()
                for s in range(1, margin + 1):
                    acc |= np.roll(occ, s, axis=axis)
                    acc |= np.roll(occ, -s, axis=axis)
                occ = acc
            grid = occ.astype(np.uint8)

    return GridMeta(grid, float(xyz_min[0]), float(xyz_min[1]), float(xyz_min[2]),
                    float(resolution))


# ---------------------------------------------------------------------------
# World <-> voxel + collision helpers
# ---------------------------------------------------------------------------

def world_to_voxel(gm: GridMeta, p) -> tuple[int, int, int]:
    x, y, z = p
    return (
        round((x - gm.x_min) / gm.resolution),
        round((y - gm.y_min) / gm.resolution),
        round((z - gm.z_min) / gm.resolution),
    )


def voxel_to_world(gm: GridMeta, v: tuple[int, int, int]) -> np.ndarray:
    ix, iy, iz = v
    return np.array(
        [gm.x_min + ix * gm.resolution, gm.y_min + iy * gm.resolution,
         gm.z_min + iz * gm.resolution],
        dtype=float,
    )


def grid_at(gm: GridMeta, ix: int, iy: int, iz: int) -> int:
    nx, ny, nz = gm.shape
    if ix < 0 or ix >= nx or iy < 0 or iy >= ny or iz < 0 or iz >= nz:
        return 1  # out of bounds = wall
    return int(gm.grid[ix, iy, iz])


def find_nearest_free(gm: GridMeta, v: tuple[int, int, int], max_radius: int = 20):
    """Snap a voxel to the nearest free voxel within `max_radius` (or None)."""
    ix, iy, iz = v
    if grid_at(gm, ix, iy, iz) == 0:
        return v
    for radius in range(1, max_radius + 1):
        best = None
        best_d = math.inf
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    if abs(dx) != radius and abs(dy) != radius and abs(dz) != radius:
                        continue
                    candidate = (ix + dx, iy + dy, iz + dz)
                    if grid_at(gm, *candidate) == 0:
                        d = dx * dx + dy * dy + dz * dz
                        if d < best_d:
                            best = candidate
                            best_d = d
        if best is not None:
            return best
    return None


def line_of_sight(gm: GridMeta, a: np.ndarray, b: np.ndarray, steps: int | None = None) -> bool:
    dist = float(np.linalg.norm(b - a))
    n = steps or max(2, math.ceil(dist / (gm.resolution * 0.7)))
    for t in np.linspace(0.0, 1.0, n + 1):
        p = a + (b - a) * t
        if grid_at(gm, *world_to_voxel(gm, p)) != 0:
            return False
    return True


def smooth_path(path: list[np.ndarray], gm: GridMeta) -> list[np.ndarray]:
    """Greedy string-pulling: drop intermediate waypoints with clear line of sight."""
    if len(path) < 3:
        return path
    result = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1:
            if line_of_sight(gm, path[i], path[j], steps=30):
                break
            j -= 1
        i = j
        result.append(path[i])
    return result


def path_length(path: list[np.ndarray] | None) -> float:
    if not path or len(path) < 2:
        return math.inf
    return float(sum(np.linalg.norm(path[i] - path[i - 1]) for i in range(1, len(path))))


# ---------------------------------------------------------------------------
# A*  (26-connected)
# ---------------------------------------------------------------------------

def astar(gm: GridMeta, start: np.ndarray, goal: np.ndarray, max_iters: int = 200_000):
    sv = find_nearest_free(gm, world_to_voxel(gm, start))
    ev = find_nearest_free(gm, world_to_voxel(gm, goal))
    if sv is None or ev is None:
        return None, {"expanded": 0, "reason": "start/goal not reachable to free space"}

    nx, ny, nz = gm.shape

    def encode(v):
        return v[0] * ny * nz + v[1] * nz + v[2]

    def decode(k):
        ix = k // (ny * nz)
        iy = (k % (ny * nz)) // nz
        iz = k % nz
        return int(ix), int(iy), int(iz)

    def heuristic(v):
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(v, ev)))

    dirs = [
        (dx, dy, dz, math.sqrt(dx * dx + dy * dy + dz * dz))
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if not (dx == dy == dz == 0)
    ]

    start_key = encode(sv)
    goal_key = encode(ev)
    came_from = {start_key: -1}
    g_score = {start_key: 0.0}
    heap = [(heuristic(sv), 0.0, sv)]
    expanded = 0

    while heap:
        _, g, cur = heapq.heappop(heap)
        key = encode(cur)
        if g > g_score.get(key, math.inf) + 1e-9:
            continue

        expanded += 1
        if expanded > max_iters:
            return None, {"expanded": expanded, "reason": "max_iters exceeded"}

        if key == goal_key:
            path = []
            while key != -1:
                path.append(voxel_to_world(gm, decode(key)))
                key = came_from[key]
            path.reverse()
            return path, {"expanded": expanded}

        ix, iy, iz = cur
        for dx, dy, dz, step in dirs:
            nxt = (ix + dx, iy + dy, iz + dz)
            if grid_at(gm, *nxt) != 0:
                continue
            nk = encode(nxt)
            ng = g + step
            if ng < g_score.get(nk, math.inf):
                g_score[nk] = ng
                came_from[nk] = key
                heapq.heappush(heap, (ng + heuristic(nxt), ng, nxt))

    return None, {"expanded": expanded, "reason": "no path"}


# ---------------------------------------------------------------------------
# RRT*  (continuous world-space sampling, collision-checked on the grid)
# ---------------------------------------------------------------------------

def rrt_star(
    gm: GridMeta,
    start_world: np.ndarray,
    goal_world: np.ndarray,
    rng: np.random.Generator,
    max_iter: int = 3000,
    step_len: float = 0.5,
    rewire_radius: float = 1.5,
    goal_bias: float = 0.1,
):
    sv = find_nearest_free(gm, world_to_voxel(gm, start_world))
    ev = find_nearest_free(gm, world_to_voxel(gm, goal_world))
    if sv is None or ev is None:
        return None, {"nodes": 0, "reason": "start/goal not reachable to free space"}

    start = voxel_to_world(gm, sv)
    goal = voxel_to_world(gm, ev)
    shape = np.array(gm.shape, dtype=float)
    lo = np.array([gm.x_min, gm.y_min, gm.z_min]) - 1.0
    hi = np.array([gm.x_min, gm.y_min, gm.z_min]) + shape * gm.resolution + 1.0

    nodes = [{"p": start, "parent": -1, "cost": 0.0}]
    best_goal_idx = -1
    best_goal_cost = math.inf

    def nearest(q):
        dists = [float(np.linalg.norm(n["p"] - q)) for n in nodes]
        return int(np.argmin(dists))

    def near(q):
        return [i for i, n in enumerate(nodes) if float(np.linalg.norm(n["p"] - q)) <= rewire_radius]

    for _ in range(max_iter):
        if rng.random() < goal_bias:
            q = goal
        else:
            q = rng.uniform(lo, hi)

        ni = nearest(q)
        base = nodes[ni]
        d = float(np.linalg.norm(q - base["p"]))
        if d == 0:
            continue
        new_p = base["p"] + (q - base["p"]) * min(step_len / d, 1.0)

        if not line_of_sight(gm, base["p"], new_p):
            continue

        near_ids = near(new_p)
        parent = ni
        best_cost = base["cost"] + float(np.linalg.norm(new_p - base["p"]))
        for idx in near_ids:
            candidate = nodes[idx]
            cost = candidate["cost"] + float(np.linalg.norm(new_p - candidate["p"]))
            if cost < best_cost and line_of_sight(gm, candidate["p"], new_p):
                parent = idx
                best_cost = cost

        new_idx = len(nodes)
        nodes.append({"p": new_p, "parent": parent, "cost": best_cost})

        for idx in near_ids:
            if idx == parent:
                continue
            candidate = nodes[idx]
            new_cost = best_cost + float(np.linalg.norm(candidate["p"] - new_p))
            if new_cost < candidate["cost"] and line_of_sight(gm, new_p, candidate["p"]):
                candidate["parent"] = new_idx
                candidate["cost"] = new_cost

        d_goal = float(np.linalg.norm(goal - new_p))
        if d_goal < step_len and line_of_sight(gm, new_p, goal):
            goal_cost = best_cost + d_goal
            if goal_cost < best_goal_cost:
                best_goal_cost = goal_cost
                best_goal_idx = len(nodes)
                nodes.append({"p": goal, "parent": new_idx, "cost": goal_cost})

    if best_goal_idx == -1:
        return None, {"nodes": len(nodes), "reason": "goal not connected"}

    path = []
    cur = best_goal_idx
    while cur != -1:
        path.append(nodes[cur]["p"])
        cur = nodes[cur]["parent"]
    path.reverse()
    return path, {"nodes": len(nodes)}


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def plan_path(
    points_world: np.ndarray,
    start_world,
    goal_world,
    *,
    algo: str = "astar",
    resolution: float = 0.15,
    margin: int = 1,
    sample: int = 10,
    smooth: bool = True,
    seed: int = 0,
    max_iters: int = 200_000,
    rrt_iter: int = 3000,
    rrt_step: float = 0.5,
    rrt_radius: float = 1.5,
    rrt_bias: float = 0.1,
    gm: Optional[GridMeta] = None,
):
    """Plan a world-meter path from `start_world` to `goal_world` through the
    occupancy grid of `points_world`.

    Returns `(waypoints | None, info, gm)`. `waypoints` is an ordered list of
    world-meter `np.ndarray([x,y,z])`. Pass a prebuilt `gm` to skip
    re-voxelizing the same scene across queries.
    """
    start_world = np.asarray(start_world, dtype=float)
    goal_world = np.asarray(goal_world, dtype=float)
    if gm is None:
        gm = voxelize(np.asarray(points_world, dtype=float), resolution, margin, sample)

    if algo == "rrt":
        rng = np.random.default_rng(seed)
        path, info = rrt_star(gm, start_world, goal_world, rng,
                              max_iter=rrt_iter, step_len=rrt_step,
                              rewire_radius=rrt_radius, goal_bias=rrt_bias)
    else:
        path, info = astar(gm, start_world, goal_world, max_iters=max_iters)

    if path is not None and smooth:
        path = smooth_path(path, gm)
    info = dict(info)
    info["algo"] = algo
    info["length_m"] = path_length(path)
    info["n_waypoints"] = len(path) if path is not None else 0
    return path, info, gm
