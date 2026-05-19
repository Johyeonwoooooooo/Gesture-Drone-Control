#!/usr/bin/env python3
"""
2D A* vs RRT* comparison using the SAME voxel NPY file.

This script:
1. Loads your 3D voxel/point-cloud NPY
2. Projects it into a 2D occupancy map (top-view)
3. Runs:
      - 2D A*
      - 2D RRT*
4. Compares performance
5. Saves graphs + path visualization

Run:
    python compare_2d_from_voxel.py

Optional:
    python compare_2d_from_voxel.py --show
"""

from __future__ import annotations

import argparse
import heapq
import math
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# =========================================================
# INPUT
# =========================================================

DEFAULT_INPUT = Path(
    "/Users/yoochaewon/Desktop/Qpor2mEya8F_voxel.npy"
)


# =========================================================
# GRID
# =========================================================

@dataclass
class Grid2D:
    grid: np.ndarray
    x_min: float
    y_min: float
    resolution: float

    @property
    def shape(self):
        return self.grid.shape


# =========================================================
# LOAD POINTS
# =========================================================

def load_points(path: Path) -> np.ndarray:

    arr = np.load(path)

    arr = np.asarray(arr, dtype=float)

    if arr.ndim == 1:
        arr = arr.reshape(-1, 3)

    if arr.shape[1] < 3:
        raise ValueError(
            "NPY must contain x,y,z"
        )

    return arr[:, :3]


# =========================================================
# 3D -> 2D projection
# =========================================================

def build_2d_map(
    points: np.ndarray,
    resolution=0.15,
    sample=10,
):

    pts = points[::max(1, sample)]

    xy = pts[:, :2]

    xy_min = xy.min(axis=0)
    xy_max = xy.max(axis=0)

    shape = (
        np.ceil((xy_max - xy_min) / resolution)
        .astype(int)
        + 3
    )

    grid = np.zeros(shape, dtype=np.uint8)

    ij = np.rint(
        (xy - xy_min) / resolution
    ).astype(int)

    valid = np.all(
        (ij >= 0) & (ij < shape),
        axis=1
    )

    grid[
        ij[valid, 0],
        ij[valid, 1]
    ] = 1

    return Grid2D(
        grid=grid,
        x_min=xy_min[0],
        y_min=xy_min[1],
        resolution=resolution,
    )


# =========================================================
# coordinate transforms
# =========================================================

def world_to_grid(gm: Grid2D, p):

    x, y = p

    return (
        round((x - gm.x_min) / gm.resolution),
        round((y - gm.y_min) / gm.resolution),
    )


def grid_to_world(gm: Grid2D, p):

    ix, iy = p

    return np.array([
        gm.x_min + ix * gm.resolution,
        gm.y_min + iy * gm.resolution,
    ])


# =========================================================
# occupancy
# =========================================================

def grid_at(gm: Grid2D, x, y):

    nx, ny = gm.shape

    if x < 0 or x >= nx:
        return 1

    if y < 0 or y >= ny:
        return 1

    return int(gm.grid[x, y])


# =========================================================
# line of sight
# =========================================================

def line_of_sight(gm, a, b, steps=40):

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    for t in np.linspace(0, 1, steps):

        p = a + (b - a) * t

        gx, gy = world_to_grid(gm, p)

        if grid_at(gm, gx, gy) != 0:
            return False

    return True


# =========================================================
# A*
# =========================================================

def heuristic(a, b):

    return math.hypot(
        a[0] - b[0],
        a[1] - b[1],
    )


def astar(gm, start_world, goal_world):

    start = world_to_grid(gm, start_world)
    goal = world_to_grid(gm, goal_world)

    dirs = [
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, math.sqrt(2)),
        (-1, 1, math.sqrt(2)),
        (1, -1, math.sqrt(2)),
        (1, 1, math.sqrt(2)),
    ]

    open_set = []

    heapq.heappush(
        open_set,
        (0, start)
    )

    came_from = {}

    g_score = {
        start: 0.0
    }

    expanded = 0

    while open_set:

        _, current = heapq.heappop(open_set)

        expanded += 1

        if current == goal:

            path = []

            while current in came_from:

                path.append(
                    grid_to_world(gm, current)
                )

                current = came_from[current]

            path.append(
                grid_to_world(gm, start)
            )

            path.reverse()

            return path, {
                "expanded": expanded
            }

        cx, cy = current

        for dx, dy, cost in dirs:

            nx = cx + dx
            ny = cy + dy

            if grid_at(gm, nx, ny) != 0:
                continue

            neighbor = (nx, ny)

            tentative = (
                g_score[current]
                + cost
            )

            if tentative < g_score.get(
                neighbor,
                math.inf
            ):

                came_from[neighbor] = current

                g_score[neighbor] = tentative

                f = tentative + heuristic(
                    neighbor,
                    goal
                )

                heapq.heappush(
                    open_set,
                    (f, neighbor)
                )

    return None, {
        "expanded": expanded
    }


# =========================================================
# RRT*
# =========================================================

def rrt_star(
    gm,
    start,
    goal,
    rng,
    max_iter=4000,
    step_len=0.6,
    radius=1.5,
    goal_bias=0.1,
):

    start = np.asarray(start, dtype=float)
    goal = np.asarray(goal, dtype=float)

    nodes = [{
        "p": start,
        "parent": -1,
        "cost": 0.0
    }]

    best_goal = -1
    best_cost = math.inf

    lo = np.array([
        gm.x_min,
        gm.y_min,
    ])

    hi = np.array([
        gm.x_min + gm.shape[0] * gm.resolution,
        gm.y_min + gm.shape[1] * gm.resolution,
    ])

    def nearest(q):

        dists = [
            np.linalg.norm(
                n["p"] - q
            )
            for n in nodes
        ]

        return int(np.argmin(dists))

    def near(q):

        return [
            i
            for i, n in enumerate(nodes)
            if np.linalg.norm(
                n["p"] - q
            ) <= radius
        ]

    for _ in range(max_iter):

        if rng.random() < goal_bias:
            q = goal
        else:
            q = rng.uniform(lo, hi)

        ni = nearest(q)

        base = nodes[ni]

        direction = q - base["p"]

        dist = np.linalg.norm(direction)

        if dist == 0:
            continue

        direction /= dist

        new_p = (
            base["p"]
            + direction
            * min(step_len, dist)
        )

        if not line_of_sight(
            gm,
            base["p"],
            new_p,
        ):
            continue

        near_ids = near(new_p)

        parent = ni

        best_parent_cost = (
            base["cost"]
            + np.linalg.norm(
                new_p - base["p"]
            )
        )

        for idx in near_ids:

            cand = nodes[idx]

            cost = (
                cand["cost"]
                + np.linalg.norm(
                    new_p - cand["p"]
                )
            )

            if (
                cost < best_parent_cost
                and line_of_sight(
                    gm,
                    cand["p"],
                    new_p,
                )
            ):
                parent = idx
                best_parent_cost = cost

        new_idx = len(nodes)

        nodes.append({
            "p": new_p,
            "parent": parent,
            "cost": best_parent_cost,
        })

        # rewire
        for idx in near_ids:

            if idx == parent:
                continue

            cand = nodes[idx]

            new_cost = (
                best_parent_cost
                + np.linalg.norm(
                    cand["p"] - new_p
                )
            )

            if (
                new_cost < cand["cost"]
                and line_of_sight(
                    gm,
                    new_p,
                    cand["p"]
                )
            ):
                cand["parent"] = new_idx
                cand["cost"] = new_cost

        # goal check
        if (
            np.linalg.norm(goal - new_p)
            < step_len
            and line_of_sight(
                gm,
                new_p,
                goal,
            )
        ):

            goal_cost = (
                best_parent_cost
                + np.linalg.norm(
                    goal - new_p
                )
            )

            if goal_cost < best_cost:

                best_cost = goal_cost

                best_goal = len(nodes)

                nodes.append({
                    "p": goal,
                    "parent": new_idx,
                    "cost": goal_cost,
                })

    if best_goal == -1:
        return None, {
            "nodes": len(nodes)
        }

    path = []

    cur = best_goal

    while cur != -1:

        path.append(
            nodes[cur]["p"]
        )

        cur = nodes[cur]["parent"]

    path.reverse()

    return path, {
        "nodes": len(nodes)
    }


# =========================================================
# metrics
# =========================================================

def path_length(path):

    if path is None:
        return math.inf

    total = 0.0

    for i in range(1, len(path)):

        total += np.linalg.norm(
            path[i] - path[i - 1]
        )

    return total


def run_once(name, func, *args):

    t0 = time.perf_counter()

    path, info = func(*args)

    elapsed = (
        time.perf_counter() - t0
    ) * 1000.0

    return {
        "algorithm": name,
        "success": path is not None,
        "time_ms": elapsed,
        "distance": path_length(path),
        "waypoints": len(path) if path else 0,
        "path": path,
        **info,
    }


# =========================================================
# plotting
# =========================================================

def plot_map_and_paths(
    gm,
    astar_path,
    rrt_path,
):

    fig, ax = plt.subplots(
        figsize=(8, 8)
    )

    ax.imshow(
        gm.grid.T,
        origin="lower",
        cmap="gray_r"
    )

    if astar_path:

        p = np.array([
            world_to_grid(gm, x)
            for x in astar_path
        ])

        ax.plot(
            p[:, 0],
            p[:, 1],
            linewidth=2,
            label="A*"
        )

    if rrt_path:

        p = np.array([
            world_to_grid(gm, x)
            for x in rrt_path
        ])

        ax.plot(
            p[:, 0],
            p[:, 1],
            linewidth=2,
            label="RRT*"
        )

    ax.legend()

    ax.set_title(
        "2D Projection Path Planning"
    )

    plt.tight_layout()

    plt.savefig(
        "2d_paths.png",
        dpi=180
    )

    print("Saved: 2d_paths.png")


def plot_metrics(results):

    labels = ["A*", "RRT*"]

    metrics = [
        ("time_ms", "Time (ms)"),
        ("distance", "Distance"),
        ("waypoints", "Waypoints"),
    ]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(14, 4)
    )

    for ax, (key, title) in zip(
        axes,
        metrics
    ):

        vals = []

        for label in labels:

            v = [
                r[key]
                for r in results
                if r["algorithm"] == label
            ]

            v = [
                x for x in v
                if np.isfinite(x)
            ]

            vals.append(np.mean(v))

        ax.bar(labels, vals)

        ax.set_title(title)

        ax.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        "2d_comparison.png",
        dpi=180
    )

    print("Saved: 2d_comparison.png")


# =========================================================
# MAIN
# =========================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--show",
        action="store_true"
    )

    args = parser.parse_args()

    points = load_points(
        args.input
    )

    gm = build_2d_map(points)

    print(
        f"Grid shape: {gm.shape}"
    )

    # SAME world coordinates
    start = np.array([
        -3.0,
        -0.5,
    ])

    goal = np.array([
        3.5,
        0.5,
    ])

    rng = np.random.default_rng(42)

    results = []

    last_astar = None
    last_rrt = None

    for _ in range(args.runs):

        a = run_once(
            "A*",
            astar,
            gm,
            start,
            goal,
        )

        r = run_once(
            "RRT*",
            rrt_star,
            gm,
            start,
            goal,
            rng,
        )

        results.append(a)
        results.append(r)

        last_astar = a["path"]
        last_rrt = r["path"]

    # summary
    print("\n===== SUMMARY =====")

    for algo in ["A*", "RRT*"]:

        rows = [
            r for r in results
            if r["algorithm"] == algo
        ]

        success = [
            r for r in rows
            if r["success"]
        ]

        print(f"\n{algo}")

        print(
            f"success: "
            f"{len(success)}/{len(rows)}"
        )

        print(
            f"avg time: "
            f"{np.mean([x['time_ms'] for x in success]):.2f} ms"
        )

        print(
            f"avg distance: "
            f"{np.mean([x['distance'] for x in success]):.2f}"
        )

    plot_metrics(results)

    plot_map_and_paths(
        gm,
        last_astar,
        last_rrt,
    )

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()