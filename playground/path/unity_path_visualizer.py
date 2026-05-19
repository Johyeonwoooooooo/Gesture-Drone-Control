"""
Create a 2D PNG showing the occupancy grid and the planned path used for
Unity autopilot runs.

Example:
    python playground/path/unity_path_visualizer.py ^
        --map-json playground/path/TEEsavR23oF_occupancy_grid_two.json ^
        --result-json playground/path/unity_autopilot_result_collision.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from integrated_path_benchmark import HybridAStarRRTPlanner, smooth_path  # noqa: E402
from unity_scene_map import UnitySceneGrid, load_unity_scene_grid  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DRONE_PATHFINDING_DIR = REPO_ROOT / "drone_pathfinding"
if str(DRONE_PATHFINDING_DIR) not in sys.path:
    sys.path.insert(0, str(DRONE_PATHFINDING_DIR))

from astar import AStarPlanner  # type: ignore  # noqa: E402
from rrt import RRTStarPlanner  # type: ignore  # noqa: E402


def planner_from_name(name: str):
    lowered = name.lower()
    if lowered == "astar":
        return AStarPlanner()
    if lowered == "rrt":
        return RRTStarPlanner(max_iter=5000, step_size=3.0, goal_sample_rate=0.12)
    if lowered == "hybrid":
        return HybridAStarRRTPlanner(
            coarse_stride=5,
            max_iter=3000,
            step_size=3.0,
            goal_sample_rate=0.18,
            corridor_sample_rate=0.72,
            corridor_radius=5.5,
        )
    raise ValueError(f"Unsupported planner: {name}")


def load_result(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def path_to_arrays(path: Sequence[Tuple[float, float]]) -> Tuple[np.ndarray, np.ndarray]:
    xs = np.array([p[0] for p in path], dtype=float)
    ys = np.array([p[1] for p in path], dtype=float)
    return xs, ys


def save_visualization(
    scene_grid: UnitySceneGrid,
    planner_name: str,
    start_world: Sequence[float],
    goal_world: Sequence[float],
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    start_grid = scene_grid.clamp_grid_point(scene_grid.world_to_grid(start_world[0], start_world[2]))
    goal_grid = scene_grid.clamp_grid_point(scene_grid.world_to_grid(goal_world[0], goal_world[2]))

    planner = planner_from_name(planner_name)
    path = planner.plan(scene_grid.map_, start_grid, goal_grid)
    if not path:
        raise RuntimeError("Could not reconstruct a path for visualization.")

    smoothed = smooth_path(scene_grid.map_, path)

    occupancy = np.array(scene_grid.map_.grid, dtype=np.int8)

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.imshow(occupancy, cmap="gray_r", origin="lower", interpolation="nearest")

    raw_x, raw_y = path_to_arrays(path)
    smooth_x, smooth_y = path_to_arrays(smoothed)

    ax.plot(raw_x, raw_y, color="#f4a261", linewidth=1.5, alpha=0.7, label="Raw path")
    ax.plot(smooth_x, smooth_y, color="#e63946", linewidth=2.5, label="Smoothed path")
    ax.scatter([start_grid[0]], [start_grid[1]], color="#2a9d8f", s=80, marker="o", label="Start")
    ax.scatter([goal_grid[0]], [goal_grid[1]], color="#1d3557", s=90, marker="*", label="Goal")

    ax.set_title(f"2D Planned Path on Occupancy Grid ({planner_name.upper()})")
    ax.set_xlabel("Grid X")
    ax.set_ylabel("Grid Y")
    ax.legend(loc="upper right")
    ax.grid(False)

    summary = (
        f"Start world: ({start_world[0]:.2f}, {start_world[2]:.2f})\n"
        f"Goal world: ({goal_world[0]:.2f}, {goal_world[2]:.2f})\n"
        f"Cell size: {scene_grid.cell_size:.2f}"
    )
    ax.text(
        0.02,
        0.02,
        summary,
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
        ha="left",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85},
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-json", required=True)
    parser.add_argument("--result-json", required=True)
    parser.add_argument("--output", default=str(CURRENT_DIR / "unity_path_visualization.png"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scene_grid = load_unity_scene_grid(args.map_json)
    result = load_result(args.result_json)

    output_path = Path(args.output)
    save_visualization(
        scene_grid=scene_grid,
        planner_name=result["planner"],
        start_world=result["start_world"],
        goal_world=result["goal_world"],
        output_path=output_path,
    )
    print(f"Saved path visualization to {output_path}")


if __name__ == "__main__":
    main()
