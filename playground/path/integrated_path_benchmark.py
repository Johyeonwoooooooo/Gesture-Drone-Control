"""
Integrated playground benchmark for A*, RRT*, and a hybrid planner.

This script connects the reusable planners from ``drone_pathfinding`` to a
simple playground simulation loop so all planners can be evaluated under the
same conditions.

Run:
    python playground/path/integrated_path_benchmark.py
"""

from __future__ import annotations

import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DRONE_PATHFINDING_DIR = REPO_ROOT / "drone_pathfinding"
if str(DRONE_PATHFINDING_DIR) not in sys.path:
    sys.path.insert(0, str(DRONE_PATHFINDING_DIR))

from astar import AStarPlanner  # type: ignore  # noqa: E402
from core import Controller, Map, Planner, Point  # type: ignore  # noqa: E402
from maps import GridMap2D  # type: ignore  # noqa: E402
from rrt import RRTStarPlanner  # type: ignore  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def compute_path_length(path: Sequence[Point]) -> float:
    if len(path) < 2:
        return 0.0
    return float(
        sum(math.dist(tuple(path[i]), tuple(path[i + 1])) for i in range(len(path) - 1))
    )


def compute_smoothness(path: Sequence[Point]) -> float:
    if len(path) < 3:
        return 0.0

    total_angle = 0.0
    for i in range(len(path) - 2):
        v1 = np.array(path[i + 1], dtype=float) - np.array(path[i], dtype=float)
        v2 = np.array(path[i + 2], dtype=float) - np.array(path[i + 1], dtype=float)
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 == 0.0 or n2 == 0.0:
            continue
        cos_angle = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
        total_angle += abs(math.degrees(math.acos(cos_angle)))
    return total_angle


def segment_collision_free(map_: Map, a: Point, b: Point, steps: int = 20) -> bool:
    for i in range(steps + 1):
        t = i / steps
        point = tuple(ai + (bi - ai) * t for ai, bi in zip(a, b))
        if not map_.is_free(point):
            return False
    return True


def smooth_path(map_: Map, path: Sequence[Point]) -> List[Point]:
    if len(path) <= 2:
        return list(path)

    smoothed = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1:
            if segment_collision_free(map_, path[i], path[j], steps=24):
                break
            j -= 1
        smoothed.append(path[j])
        i = j
    return smoothed


def distance_point_to_polyline(point: Point, polyline: Sequence[Point]) -> float:
    if not polyline:
        return float("inf")
    p = np.array(point, dtype=float)
    best = float("inf")
    for anchor in polyline:
        best = min(best, float(np.linalg.norm(p - np.array(anchor, dtype=float))))
    return best


def simulate_path_following(
    path: Sequence[Point],
    start: Point,
    goal: Point,
    dt: float = 0.1,
    max_steps: int = 2500,
    max_speed: float = 4.0,
) -> Dict[str, float]:
    if not path:
        return {"sim_steps": 0, "travel_distance": 0.0, "final_error": float("inf")}

    controller = Controller(
        waypoints=list(path),
        kp=1.5,
        ki=0.01,
        kd=0.45,
        arrival_threshold=0.8,
    )

    pos = np.array(start, dtype=float)
    travel_distance = 0.0

    for step in range(max_steps):
        if controller.is_done:
            return {
                "sim_steps": step,
                "travel_distance": travel_distance,
                "final_error": float(np.linalg.norm(pos - np.array(goal, dtype=float))),
            }

        velocity = controller.compute(tuple(pos), dt)
        speed = float(np.linalg.norm(velocity))
        if speed > max_speed:
            velocity = velocity / speed * max_speed

        prev = pos.copy()
        pos = pos + velocity * dt
        travel_distance += float(np.linalg.norm(pos - prev))

    return {
        "sim_steps": max_steps,
        "travel_distance": travel_distance,
        "final_error": float(np.linalg.norm(pos - np.array(goal, dtype=float))),
    }


class HybridAStarRRTPlanner(Planner):
    """
    Hybrid planner:
    1. Build a reliable coarse route with A*.
    2. Refine it with a guided RRT* search that samples near the A* corridor.
    """

    def __init__(
        self,
        coarse_stride: int = 5,
        max_iter: int = 3500,
        step_size: float = 3.0,
        goal_sample_rate: float = 0.15,
        corridor_sample_rate: float = 0.7,
        corridor_radius: float = 6.0,
    ) -> None:
        self.coarse_stride = coarse_stride
        self.max_iter = max_iter
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        self.corridor_sample_rate = corridor_sample_rate
        self.corridor_radius = corridor_radius
        self.astar = AStarPlanner()
        self.coarse_path: List[Point] = []
        self.tree_edges: List[Tuple[Point, Point]] = []

    def plan(self, map_: Map, start: Point, goal: Point) -> List[Point]:
        self.tree_edges = []
        coarse_path = self.astar.plan(map_, start, goal)
        self.coarse_path = coarse_path
        if not coarse_path:
            return []

        anchors = coarse_path[:: self.coarse_stride]
        if anchors[-1] != coarse_path[-1]:
            anchors.append(coarse_path[-1])

        bounds_min, bounds_max = map_.get_bounds()
        dims = map_.dimensions
        nodes: List[Tuple[Point, int | None, float]] = [(start, None, 0.0)]
        goal_threshold = self.step_size * 1.5

        for _ in range(self.max_iter):
            sample = self._sample(bounds_min, bounds_max, goal, anchors)
            nearest_idx = self._nearest(nodes, sample)
            nearest_point = nodes[nearest_idx][0]
            new_point = self._steer(nearest_point, sample)

            if not segment_collision_free(map_, nearest_point, new_point, steps=12):
                continue
            if distance_point_to_polyline(new_point, anchors) > self.corridor_radius * 1.5:
                continue

            radius = self._compute_radius(len(nodes), dims)
            near_indices = self._near(nodes, new_point, radius)

            best_parent = nearest_idx
            best_cost = nodes[nearest_idx][2] + math.dist(nearest_point, new_point)

            for idx in near_indices:
                node_point = nodes[idx][0]
                new_cost = nodes[idx][2] + math.dist(node_point, new_point)
                if new_cost < best_cost and segment_collision_free(
                    map_, node_point, new_point, steps=12
                ):
                    best_parent = idx
                    best_cost = new_cost

            new_idx = len(nodes)
            nodes.append((new_point, best_parent, best_cost))
            self.tree_edges.append((nodes[best_parent][0], new_point))

            for idx in near_indices:
                node_point = nodes[idx][0]
                rewire_cost = best_cost + math.dist(new_point, node_point)
                if rewire_cost < nodes[idx][2] and segment_collision_free(
                    map_, new_point, node_point, steps=12
                ):
                    nodes[idx] = (node_point, new_idx, rewire_cost)

            if math.dist(new_point, goal) < goal_threshold and segment_collision_free(
                map_, new_point, goal, steps=18
            ):
                nodes.append((goal, new_idx, best_cost + math.dist(new_point, goal)))
                return smooth_path(map_, self._extract_path(nodes, len(nodes) - 1))

        return smooth_path(map_, coarse_path)

    def _sample(
        self,
        bounds_min: Point,
        bounds_max: Point,
        goal: Point,
        anchors: Sequence[Point],
    ) -> Point:
        if random.random() < self.goal_sample_rate:
            return goal

        if random.random() < self.corridor_sample_rate and anchors:
            anchor = anchors[random.randrange(len(anchors))]
            sampled = []
            for dim, base in enumerate(anchor):
                jitter = random.uniform(-self.corridor_radius, self.corridor_radius)
                sampled.append(
                    min(bounds_max[dim], max(bounds_min[dim], base + jitter))
                )
            return tuple(sampled)

        return tuple(
            random.uniform(bounds_min[d], bounds_max[d]) for d in range(len(bounds_min))
        )

    @staticmethod
    def _nearest(nodes: Sequence[Tuple[Point, int | None, float]], point: Point) -> int:
        return min(range(len(nodes)), key=lambda i: math.dist(nodes[i][0], point))

    @staticmethod
    def _near(
        nodes: Sequence[Tuple[Point, int | None, float]],
        point: Point,
        radius: float,
    ) -> List[int]:
        return [i for i in range(len(nodes)) if math.dist(nodes[i][0], point) <= radius]

    def _steer(self, from_point: Point, to_point: Point) -> Point:
        dist = math.dist(from_point, to_point)
        if dist <= self.step_size:
            return to_point
        ratio = self.step_size / dist
        return tuple(f + (t - f) * ratio for f, t in zip(from_point, to_point))

    def _compute_radius(self, n: int, dims: int) -> float:
        if n <= 1:
            return self.step_size * 3
        return min(self.step_size * 3, 30.0 * (math.log(n + 1) / (n + 1)) ** (1.0 / dims))

    @staticmethod
    def _extract_path(
        nodes: Sequence[Tuple[Point, int | None, float]], goal_idx: int
    ) -> List[Point]:
        path: List[Point] = []
        idx: int | None = goal_idx
        while idx is not None:
            path.append(nodes[idx][0])
            idx = nodes[idx][1]
        path.reverse()
        return path


def make_indoor_playground_map() -> GridMap2D:
    map_ = GridMap2D(70, 70, resolution=0.3)
    map_.add_obstacle_rect(8, 0, 4, 38)
    map_.add_obstacle_rect(8, 44, 4, 26)
    map_.add_obstacle_rect(24, 12, 4, 58)
    map_.add_obstacle_rect(40, 0, 4, 28)
    map_.add_obstacle_rect(40, 34, 4, 36)
    map_.add_obstacle_rect(56, 10, 4, 60)
    map_.add_obstacle_circle(18, 10, 4)
    map_.add_obstacle_circle(18, 54, 5)
    map_.add_obstacle_circle(34, 34, 4)
    map_.add_obstacle_circle(50, 18, 4)
    map_.add_obstacle_circle(52, 52, 4)
    return map_


def make_cluttered_playground_map() -> GridMap2D:
    map_ = GridMap2D(70, 70, resolution=0.3)
    for cx, cy, r in [
        (14, 15, 5),
        (15, 42, 4),
        (24, 28, 5),
        (28, 54, 4),
        (38, 15, 5),
        (42, 37, 5),
        (53, 22, 4),
        (55, 49, 5),
    ]:
        map_.add_obstacle_circle(cx, cy, r)
    map_.add_obstacle_rect(31, 0, 3, 18)
    map_.add_obstacle_rect(31, 26, 3, 18)
    map_.add_obstacle_rect(31, 52, 3, 18)
    return map_


def make_narrow_passage_playground_map() -> GridMap2D:
    map_ = GridMap2D(70, 70, resolution=0.3)
    for wall_x, gap_y in [(14, 12), (28, 40), (42, 18), (56, 48)]:
        map_.add_obstacle_rect(wall_x, 0, 3, 70)
        map_.grid[gap_y : gap_y + 8, wall_x : wall_x + 3] = 0
    map_.add_obstacle_circle(22, 58, 4)
    map_.add_obstacle_circle(49, 10, 4)
    return map_


MAP_BUILDERS: List[Tuple[str, Callable[[], GridMap2D], Point, Point]] = [
    ("indoor", make_indoor_playground_map, (3, 3), (66, 64)),
    ("cluttered", make_cluttered_playground_map, (4, 6), (64, 61)),
    ("narrow_passage", make_narrow_passage_playground_map, (4, 4), (65, 63)),
]


@dataclass
class BenchmarkRow:
    map_name: str
    algorithm: str
    trial: int
    success: bool
    plan_time_sec: float
    path_length: float
    smoothness_deg: float
    waypoint_count: int
    sim_steps: int
    sim_travel_distance: float
    final_error: float


def run_single(
    planner_factory: Callable[[], Planner],
    algorithm_name: str,
    map_name: str,
    map_builder: Callable[[], GridMap2D],
    start: Point,
    goal: Point,
    trial: int,
    seed: int,
) -> BenchmarkRow:
    set_seed(seed)
    planner = planner_factory()
    map_ = map_builder()

    t0 = time.perf_counter()
    path = planner.plan(map_, start, goal)
    elapsed = time.perf_counter() - t0

    if not path:
        return BenchmarkRow(
            map_name=map_name,
            algorithm=algorithm_name,
            trial=trial,
            success=False,
            plan_time_sec=elapsed,
            path_length=0.0,
            smoothness_deg=0.0,
            waypoint_count=0,
            sim_steps=0,
            sim_travel_distance=0.0,
            final_error=float("inf"),
        )

    path = smooth_path(map_, path)
    sim = simulate_path_following(path, start=start, goal=goal)
    return BenchmarkRow(
        map_name=map_name,
        algorithm=algorithm_name,
        trial=trial,
        success=True,
        plan_time_sec=elapsed,
        path_length=compute_path_length(path),
        smoothness_deg=compute_smoothness(path),
        waypoint_count=len(path),
        sim_steps=int(sim["sim_steps"]),
        sim_travel_distance=float(sim["travel_distance"]),
        final_error=float(sim["final_error"]),
    )


def summarize(rows: Iterable[BenchmarkRow]) -> Dict[Tuple[str, str], Dict[str, float]]:
    grouped: Dict[Tuple[str, str], List[BenchmarkRow]] = {}
    for row in rows:
        grouped.setdefault((row.map_name, row.algorithm), []).append(row)

    summary: Dict[Tuple[str, str], Dict[str, float]] = {}
    for key, items in grouped.items():
        successes = [item for item in items if item.success]
        summary[key] = {
            "trials": float(len(items)),
            "success_rate": (len(successes) / len(items) * 100.0) if items else 0.0,
            "avg_plan_time_sec": float(np.mean([item.plan_time_sec for item in items])) if items else 0.0,
            "avg_path_length": float(np.mean([item.path_length for item in successes])) if successes else 0.0,
            "avg_smoothness_deg": float(np.mean([item.smoothness_deg for item in successes])) if successes else 0.0,
            "avg_waypoint_count": float(np.mean([item.waypoint_count for item in successes])) if successes else 0.0,
            "avg_sim_steps": float(np.mean([item.sim_steps for item in successes])) if successes else 0.0,
            "avg_final_error": float(np.mean([item.final_error for item in successes])) if successes else float("inf"),
        }
    return summary


def print_summary(rows: Sequence[BenchmarkRow]) -> None:
    summary = summarize(rows)
    print("\nIntegrated Playground Benchmark")
    print("=" * 96)
    print(
        f"{'Map':<18} {'Algo':<10} {'Succ':>7} {'Plan(s)':>10} "
        f"{'Path':>10} {'Smooth':>10} {'WPs':>8} {'SimStep':>9} {'FinalErr':>10}"
    )
    print("-" * 96)

    for map_name, _, _start, _goal in MAP_BUILDERS:
        for algorithm in ("A*", "RRT*", "Hybrid"):
            stats = summary[(map_name, algorithm)]
            print(
                f"{map_name:<18} {algorithm:<10} {stats['success_rate']:>6.1f}% "
                f"{stats['avg_plan_time_sec']:>10.4f} {stats['avg_path_length']:>10.2f} "
                f"{stats['avg_smoothness_deg']:>10.2f} {stats['avg_waypoint_count']:>8.2f} "
                f"{stats['avg_sim_steps']:>9.1f} {stats['avg_final_error']:>10.3f}"
            )


def save_results(rows: Sequence[BenchmarkRow], output_path: Path) -> None:
    payload = [asdict(row) for row in rows]
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved detailed results to {output_path}")


def save_charts(rows: Sequence[BenchmarkRow], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    summary = summarize(rows)
    maps = [map_name for map_name, _, _, _ in MAP_BUILDERS]
    algorithms = ["A*", "RRT*", "Hybrid"]
    colors = {"A*": "#2a9d8f", "RRT*": "#e76f51", "Hybrid": "#264653"}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Integrated Playground Benchmark", fontsize=16, fontweight="bold")

    x = np.arange(len(maps))
    width = 0.24

    metrics = [
        ("avg_plan_time_sec", "Planning Time (s)", axes[0, 0]),
        ("avg_path_length", "Path Length", axes[0, 1]),
        ("avg_smoothness_deg", "Smoothness (deg)", axes[1, 0]),
        ("avg_final_error", "Final Tracking Error", axes[1, 1]),
    ]

    for metric_key, title, ax in metrics:
        for idx, algorithm in enumerate(algorithms):
            values = [summary[(map_name, algorithm)][metric_key] for map_name in maps]
            ax.bar(
                x + (idx - 1) * width,
                values,
                width=width,
                label=algorithm,
                color=colors[algorithm],
                alpha=0.9,
            )
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(maps, rotation=15)
        ax.grid(True, axis="y", alpha=0.25)

    axes[0, 0].legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved chart image to {output_path}")


def main() -> None:
    planners: List[Tuple[str, Callable[[], Planner]]] = [
        ("A*", lambda: AStarPlanner()),
        ("RRT*", lambda: RRTStarPlanner(max_iter=5000, step_size=3.0, goal_sample_rate=0.12)),
        (
            "Hybrid",
            lambda: HybridAStarRRTPlanner(
                coarse_stride=5,
                max_iter=3000,
                step_size=3.0,
                goal_sample_rate=0.18,
                corridor_sample_rate=0.72,
                corridor_radius=5.5,
            ),
        ),
    ]

    rows: List[BenchmarkRow] = []
    trials_per_map = 8
    base_seed = 20260512

    for map_index, (map_name, map_builder, start, goal) in enumerate(MAP_BUILDERS):
        for trial in range(trials_per_map):
            for algo_index, (algorithm_name, planner_factory) in enumerate(planners):
                seed = base_seed + map_index * 100 + trial * 10 + algo_index
                row = run_single(
                    planner_factory=planner_factory,
                    algorithm_name=algorithm_name,
                    map_name=map_name,
                    map_builder=map_builder,
                    start=start,
                    goal=goal,
                    trial=trial,
                    seed=seed,
                )
                rows.append(row)

    print_summary(rows)
    save_results(rows, Path(__file__).with_name("integrated_benchmark_results.json"))
    save_charts(rows, Path(__file__).with_name("integrated_benchmark_charts.png"))


if __name__ == "__main__":
    main()
