"""
Benchmark A*, RRT*, and the hybrid planner on an exported Unity scene grid.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from integrated_path_benchmark import (  # noqa: E402
    HybridAStarRRTPlanner,
    compute_path_length,
    compute_smoothness,
    smooth_path,
)
from unity_scene_map import UnitySceneGrid, load_unity_scene_grid  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DRONE_PATHFINDING_DIR = REPO_ROOT / "drone_pathfinding"
if str(DRONE_PATHFINDING_DIR) not in sys.path:
    sys.path.insert(0, str(DRONE_PATHFINDING_DIR))

from astar import AStarPlanner  # type: ignore  # noqa: E402
from rrt import RRTStarPlanner  # type: ignore  # noqa: E402


@dataclass
class SceneBenchmarkRow:
    map_type: str
    obstacle_density: float
    algorithm: str
    planner: str
    trial: int
    success: bool
    start_grid: Tuple[int, int]
    goal_grid: Tuple[int, int]
    start_world: Tuple[float, float]
    goal_world: Tuple[float, float]
    plan_time_sec: float
    path_length_world: float
    smoothness_deg: float
    waypoint_count: int
    optimal_ratio: float
    compute_time: float
    num_waypoints: int
    min_clearance: float
    smoothness: float
    start_goal_dist: float


def compute_min_clearance_grid(path: Sequence[Tuple[float, float]], scene_grid: UnitySceneGrid) -> float:
    if not path:
        return 0.0

    min_dist = float("inf")
    map_ = scene_grid.map_

    for point in path:
        for r in range(1, 10):
            found_obstacle = False
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    check = (point[0] + dx, point[1] + dy)
                    if map_.is_in_bounds(check) and not map_.is_free(check):
                        dist_cells = float(np.hypot(dx, dy))
                        min_dist = min(min_dist, dist_cells)
                        found_obstacle = True
            if found_obstacle:
                break

    if min_dist == float("inf"):
        return -1.0
    return min_dist * scene_grid.cell_size


def planner_factory(name: str):
    if name == "A*":
        return AStarPlanner()
    if name == "RRT*":
        return RRTStarPlanner(max_iter=5000, step_size=3.0, goal_sample_rate=0.12)
    if name == "Hybrid":
        return HybridAStarRRTPlanner(
            coarse_stride=5,
            max_iter=3000,
            step_size=3.0,
            goal_sample_rate=0.18,
            corridor_sample_rate=0.72,
            corridor_radius=5.5,
        )
    raise ValueError(name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def find_random_free_endpoint(scene_grid: UnitySceneGrid) -> Tuple[int, int]:
    while True:
        x = random.randint(0, scene_grid.map_.width - 1)
        y = random.randint(0, scene_grid.map_.height - 1)
        if scene_grid.map_.is_free((x, y)):
            return x, y


def grid_path_to_world(scene_grid: UnitySceneGrid, path: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    return [scene_grid.grid_to_world(x, y) for x, y in path]


def run_trial(
    scene_grid: UnitySceneGrid,
    planner_name: str,
    start_grid: Tuple[int, int],
    goal_grid: Tuple[int, int],
    trial: int,
) -> SceneBenchmarkRow:
    planner = planner_factory(planner_name)
    t0 = time.perf_counter()
    path = planner.plan(scene_grid.map_, start_grid, goal_grid)
    elapsed = time.perf_counter() - t0

    if not path:
        return SceneBenchmarkRow(
            map_type="unity_scene",
            obstacle_density=float(np.mean(scene_grid.map_.grid)),
            algorithm=planner_name,
            planner=planner_name,
            trial=trial,
            success=False,
            start_grid=start_grid,
            goal_grid=goal_grid,
            start_world=scene_grid.grid_to_world(*start_grid),
            goal_world=scene_grid.grid_to_world(*goal_grid),
            plan_time_sec=elapsed,
            path_length_world=0.0,
            smoothness_deg=0.0,
            waypoint_count=0,
            optimal_ratio=0.0,
            compute_time=elapsed,
            num_waypoints=0,
            min_clearance=0.0,
            smoothness=0.0,
            start_goal_dist=float(np.linalg.norm(np.array(scene_grid.grid_to_world(*goal_grid)) - np.array(scene_grid.grid_to_world(*start_grid)))),
        )

    path = smooth_path(scene_grid.map_, path)
    world_path = grid_path_to_world(scene_grid, path)
    start_world = scene_grid.grid_to_world(*start_grid)
    goal_world = scene_grid.grid_to_world(*goal_grid)
    path_length_world = compute_path_length(world_path)
    start_goal_dist = float(np.linalg.norm(np.array(goal_world) - np.array(start_world)))
    smoothness_deg = compute_smoothness(world_path)
    return SceneBenchmarkRow(
        map_type="unity_scene",
        obstacle_density=float(np.mean(scene_grid.map_.grid)),
        algorithm=planner_name,
        planner=planner_name,
        trial=trial,
        success=True,
        start_grid=start_grid,
        goal_grid=goal_grid,
        start_world=start_world,
        goal_world=goal_world,
        plan_time_sec=elapsed,
        path_length_world=path_length_world,
        smoothness_deg=smoothness_deg,
        waypoint_count=len(world_path),
        optimal_ratio=(path_length_world / start_goal_dist) if start_goal_dist > 0 else 1.0,
        compute_time=elapsed,
        num_waypoints=len(world_path),
        min_clearance=compute_min_clearance_grid(path, scene_grid),
        smoothness=smoothness_deg,
        start_goal_dist=start_goal_dist,
    )


def summarize(rows: Sequence[SceneBenchmarkRow]) -> Dict[str, Dict[str, float]]:
    planners = sorted({row.planner for row in rows})
    summary: Dict[str, Dict[str, float]] = {}
    for planner in planners:
        subset = [row for row in rows if row.planner == planner]
        success_rows = [row for row in subset if row.success]
        summary[planner] = {
            "success_rate": len(success_rows) / len(subset) * 100.0 if subset else 0.0,
            "avg_path_length": float(np.mean([row.path_length_world for row in success_rows])) if success_rows else 0.0,
            "avg_optimal_ratio": float(np.mean([row.optimal_ratio for row in success_rows])) if success_rows else 0.0,
            "avg_compute_time": float(np.mean([row.compute_time for row in subset])) if subset else 0.0,
            "avg_num_waypoints": float(np.mean([row.num_waypoints for row in success_rows])) if success_rows else 0.0,
            "avg_min_clearance": float(np.mean([row.min_clearance for row in success_rows])) if success_rows else 0.0,
            "avg_smoothness": float(np.mean([row.smoothness for row in success_rows])) if success_rows else 0.0,
            "avg_start_goal_dist": float(np.mean([row.start_goal_dist for row in subset])) if subset else 0.0,
        }
    return summary


def save_chart(rows: Sequence[SceneBenchmarkRow], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    summary = summarize(rows)
    planners = ["A*", "RRT*", "Hybrid"]
    colors = {"A*": "#2a9d8f", "RRT*": "#e76f51", "Hybrid": "#264653"}

    metrics = [
        ("success_rate", "Success Rate (%)"),
        ("avg_compute_time", "Computation Time (s)"),
        ("avg_optimal_ratio", "Optimal Ratio"),
        ("avg_min_clearance", "Min Clearance"),
        ("avg_smoothness", "Smoothness (deg)"),
        ("avg_num_waypoints", "Waypoint Count"),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    axes = axes.flatten()

    for ax, (metric_key, title) in zip(axes, metrics):
        values = [summary[p][metric_key] for p in planners]
        ax.bar(planners, values, color=[colors[p] for p in planners], alpha=0.9)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)

    for ax in axes[len(metrics):]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-json", required=True)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--min-grid-distance", type=float, default=25.0)
    parser.add_argument("--seed", type=int, default=20260512)
    parser.add_argument("--output-json", default=str(CURRENT_DIR / "unity_scene_benchmark_results.json"))
    parser.add_argument("--output-chart", default=str(CURRENT_DIR / "unity_scene_benchmark_chart.png"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scene_grid = load_unity_scene_grid(args.map_json)
    set_seed(args.seed)

    rows: List[SceneBenchmarkRow] = []
    planners = ["A*", "RRT*", "Hybrid"]

    for trial in range(args.trials):
        while True:
            start_grid = find_random_free_endpoint(scene_grid)
            goal_grid = find_random_free_endpoint(scene_grid)
            if np.linalg.norm(np.array(goal_grid) - np.array(start_grid)) >= args.min_grid_distance:
                break

        for planner_name in planners:
            rows.append(run_trial(scene_grid, planner_name, start_grid, goal_grid, trial))

    summary = summarize(rows)
    print(json.dumps(summary, indent=2))

    output_json = Path(args.output_json)
    output_json.write_text(json.dumps([asdict(row) for row in rows], indent=2), encoding="utf-8")
    save_chart(rows, Path(args.output_chart))

    print(f"\nSaved scene benchmark rows to {output_json}")
    print(f"Saved scene benchmark chart to {args.output_chart}")


if __name__ == "__main__":
    main()
