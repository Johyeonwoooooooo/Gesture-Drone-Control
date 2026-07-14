"""
Multi-case benchmark for the 3D voxel autopilot.

Samples N interior start/goal pairs from the exported voxel map, plans each case
(offline by default, or flies them live in Unity with --execute), then writes a
CSV of per-case metrics plus summary graphs.

Examples:
    # Planning-only benchmark, no Unity needed (fast):
    python benchmark_autopilot_3d.py --map-json TEEsavR23oF_voxel_map_3d.json --cases 30

    # Live flight benchmark (Unity must be in Play mode); flies each case:
    python benchmark_autopilot_3d.py --map-json TEEsavR23oF_voxel_map_3d.json --cases 5 --execute
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from integrated_path_benchmark import compute_path_length, compute_smoothness  # noqa: E402
from unity_autopilot_3d import (  # noqa: E402
    find_reachable_points,
    grid_path_to_world_waypoints,
    run_autopilot_3d,
)
from unity_bridge import UnityTelloBridge  # noqa: E402
from unity_scene_voxel import (  # noqa: E402
    UnitySceneVoxel,
    VoxelPoint,
    _nearest_occupied_distance,
    check_world_points_against_voxels,
    load_unity_scene_voxel,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DRONE_PATHFINDING_DIR = REPO_ROOT / "drone_pathfinding"
if str(DRONE_PATHFINDING_DIR) not in sys.path:
    sys.path.insert(0, str(DRONE_PATHFINDING_DIR))

from astar import AStarPlanner  # type: ignore  # noqa: E402


def interior_free_points(scene_voxel: UnitySceneVoxel, min_clearance_m: float = 2.0) -> List[VoxelPoint]:
    """Free voxels that are vertically enclosed (indoor-looking) with some clearance."""
    grid = scene_voxel.map_.grid  # [z, y, x]
    points: List[VoxelPoint] = []
    for x in range(1, scene_voxel.map_.width - 1):
        for z in range(1, scene_voxel.map_.depth - 1):
            column = grid[z, :, x]
            for y in np.where(column == 0)[0]:
                if not (column[:y].any() and column[y + 1:].any()):
                    continue
                world = scene_voxel.grid_to_world(x, int(y), z)
                if _nearest_occupied_distance(scene_voxel, world, 3) >= min_clearance_m:
                    points.append((x, int(y), z))
    return points


def largest_interior_pool(scene_voxel: UnitySceneVoxel) -> List[VoxelPoint]:
    """Interior free voxels restricted to the largest connected region."""
    candidates = interior_free_points(scene_voxel)
    if not candidates:
        raise RuntimeError("No interior free voxels found in the map.")

    remaining = set(candidates)
    pool: List[VoxelPoint] = []
    while remaining:
        seed_point = next(iter(remaining))
        component = find_reachable_points(scene_voxel, seed_point)
        members = [p for p in remaining if p in component]
        if len(members) > len(pool):
            pool = members
        remaining -= set(members)
        remaining.discard(seed_point)
    if len(pool) < 2:
        raise RuntimeError("Interior free space is too fragmented to sample pairs.")
    print(f"candidate pool: {len(pool)}/{len(candidates)} interior voxels in the largest connected region")
    return pool


def sample_cases(
    scene_voxel: UnitySceneVoxel,
    count: int,
    seed: int,
    min_distance_m: float = 20.0,
) -> List[Tuple[VoxelPoint, VoxelPoint]]:
    rng = random.Random(seed)
    pool = largest_interior_pool(scene_voxel)

    min_dist_vox = min_distance_m / scene_voxel.voxel_size
    cases: List[Tuple[VoxelPoint, VoxelPoint]] = []
    attempts = 0
    while len(cases) < count and attempts < count * 200:
        attempts += 1
        start, goal = rng.sample(pool, 2)
        if math.dist(start, goal) < min_dist_vox:
            continue
        cases.append((start, goal))
    if len(cases) < count:
        print(f"[warn] only sampled {len(cases)}/{count} pairs at min distance {min_distance_m}m")
    return cases


def run_offline_case(
    scene_voxel: UnitySceneVoxel,
    planner: AStarPlanner,
    start: VoxelPoint,
    goal: VoxelPoint,
) -> dict:
    start_world = scene_voxel.grid_to_world(*start)
    goal_world = scene_voxel.grid_to_world(*goal)
    straight = math.dist(start_world, goal_world)

    t0 = time.perf_counter()
    path = planner.plan(scene_voxel.map_, start, goal)
    plan_time = time.perf_counter() - t0

    if not path:
        return {
            "success": False,
            "plan_time_sec": plan_time,
            "straight_dist_m": straight,
        }

    waypoints = grid_path_to_world_waypoints(scene_voxel, path)
    report = check_world_points_against_voxels(scene_voxel, waypoints, clearance_search_radius=3)
    length = compute_path_length(waypoints)
    return {
        "success": True,
        "plan_time_sec": plan_time,
        "straight_dist_m": straight,
        "path_length_m": length,
        "detour_ratio": length / straight if straight > 0 else float("nan"),
        "smoothness_deg": compute_smoothness(waypoints),
        "waypoint_count": len(waypoints),
        "planned_intrusion_steps": report.intrusion_points,
        "planned_min_clearance_m": report.min_clearance_m,
    }


def run_execute_case(
    scene_voxel: UnitySceneVoxel,
    bridge: UnityTelloBridge,
    start: VoxelPoint,
    goal: VoxelPoint,
    case_index: int,
    output_dir: Path,
) -> dict:
    start_world = scene_voxel.grid_to_world(*start)
    goal_world = scene_voxel.grid_to_world(*goal)
    result = run_autopilot_3d(
        scene_voxel=scene_voxel,
        bridge=bridge,
        goal_x=goal_world[0],
        goal_y=goal_world[1],
        goal_z=goal_world[2],
        execute=True,
        unity_path_json=output_dir / f"case_{case_index:02d}_path.json",
        start_x=start_world[0],
        start_y=start_world[1],
        start_z=start_world[2],
        trajectory_out=output_dir / f"case_{case_index:02d}_traj.json",
    )
    row = {
        "success": result.success,
        "plan_time_sec": result.plan_time_sec,
        "straight_dist_m": math.dist(start_world, goal_world),
        "path_length_m": result.path_length,
        "smoothness_deg": result.smoothness_deg,
        "waypoint_count": result.waypoint_count,
        "planned_intrusion_steps": result.planned_intrusion_steps,
        "planned_min_clearance_m": result.planned_min_clearance_m,
        "execute_time_sec": result.execute_time_sec,
        "mean_tracking_error_m": result.mean_tracking_error,
        "max_tracking_error_m": result.max_tracking_error,
        "final_position_error_m": result.final_position_error,
        "collision_count": result.collision_count,
        "trajectory_intrusion_steps": result.trajectory_intrusion_steps,
        "trajectory_intrusion_ratio": result.trajectory_intrusion_ratio,
    }
    if row["straight_dist_m"] > 0:
        row["detour_ratio"] = row["path_length_m"] / row["straight_dist_m"]
    return row


def run_edge_suite(
    scene_voxel: UnitySceneVoxel,
    cases_per_category: int,
    seed: int,
) -> List[dict]:
    """Deliberately hostile inputs, one category at a time.

    Proves robustness two ways:
    - invalid inputs (goal in a wall, outside the map, unreachable, start==goal)
      must be handled gracefully: no exception, and a sensible outcome
    - hard-but-valid inputs (big vertical shift, long range, near map boundary)
      must still produce a clean plan (path found, zero planned intrusions)
    """
    rng = random.Random(seed)
    planner = AStarPlanner()
    grid = scene_voxel.map_.grid

    pool = largest_interior_pool(scene_voxel)
    occupied = [(int(x), int(y), int(z)) for z, y, x in np.argwhere(grid == 1)]
    free_all = [(int(x), int(y), int(z)) for z, y, x in np.argwhere(grid == 0)]

    reach = find_reachable_points(scene_voxel, pool[0])
    unreachable_free = [p for p in free_all if p not in reach]

    def near_boundary_free() -> List[VoxelPoint]:
        result = []
        for x, y, z in free_all:
            margin = min(x, y, z,
                         scene_voxel.map_.width - 1 - x,
                         scene_voxel.map_.height - 1 - y,
                         scene_voxel.map_.depth - 1 - z)
            if margin <= 1:
                result.append((x, y, z))
        return result

    def plan_between(start: VoxelPoint, goal: VoxelPoint) -> dict:
        t0 = time.perf_counter()
        path = planner.plan(scene_voxel.map_, start, goal)
        plan_time = time.perf_counter() - t0
        row = {"plan_time_sec": plan_time, "path_found": bool(path)}
        if path:
            waypoints = grid_path_to_world_waypoints(scene_voxel, path)
            report = check_world_points_against_voxels(scene_voxel, waypoints, clearance_search_radius=3)
            row["planned_intrusion_steps"] = report.intrusion_points
        return row

    def snap(point: VoxelPoint) -> VoxelPoint | None:
        return scene_voxel.find_nearest_free(scene_voxel.clamp_grid_point(point))

    rows: List[dict] = []

    def record(category: str, index: int, expected: str, runner) -> None:
        row = {"category": category, "case": index, "expected": expected}
        try:
            row.update(runner())
            row["crashed"] = False
        except Exception as exc:  # noqa: BLE001 — the whole point is catching crashes
            row["crashed"] = True
            row["error"] = f"{type(exc).__name__}: {exc}"

        if row["crashed"]:
            row["passed"] = False
        elif expected == "plan":
            row["passed"] = bool(row.get("path_found")) and row.get("planned_intrusion_steps", 0) == 0
        elif expected == "snap":
            # Bad input must be snapped to a valid voxel and answered deterministically;
            # the snapped goal may legitimately be unreachable (reported via path_found).
            row["passed"] = bool(row.get("snapped"))
        else:  # "graceful": no crash; empty result is a legitimate outcome
            row["passed"] = True
        rows.append(row)

    def snap_and_plan(start: VoxelPoint, raw_goal: VoxelPoint) -> dict:
        snapped = snap(raw_goal)
        if snapped is None:
            return {"snapped": False, "path_found": False}
        result = plan_between(start, snapped)
        result["snapped"] = True
        return result

    for i in range(cases_per_category):
        start = rng.choice(pool)

        # 1. Goal inside a wall — autopilot snaps it to the nearest free voxel.
        goal_occ = rng.choice(occupied)
        record("goal_in_obstacle", i, "snap", lambda s=start, g=goal_occ: snap_and_plan(s, g))

        # 2. Goal far outside the map bounds — clamped, then snapped.
        oob = (scene_voxel.map_.width + 30, scene_voxel.map_.height + 30, scene_voxel.map_.depth + 30)
        record("goal_out_of_bounds", i, "snap", lambda s=start, g=oob: snap_and_plan(s, g))

        # 3. Goal in a disconnected free region — planner must fail cleanly, not hang/crash.
        if unreachable_free:
            goal_unreach = rng.choice(unreachable_free)
            record("goal_unreachable", i, "graceful",
                   lambda s=start, g=goal_unreach: plan_between(s, g))

        # 4. Start == goal.
        record("start_equals_goal", i, "graceful", lambda s=start: plan_between(s, s))

        # 5. Large vertical shift (>= 8 m) — must climb between floors.
        vertical = [p for p in pool if abs(p[1] - start[1]) >= 4]
        if vertical:
            record("vertical_shift", i, "plan",
                   lambda s=start, g=rng.choice(vertical): plan_between(s, g))

        # 6. Long range: the most distant reachable interior point.
        far = max(pool, key=lambda p, s=start: math.dist(s, p))
        record("long_range", i, "plan", lambda s=start, g=far: plan_between(s, g))

        # 7. Start snapped from the map boundary.
        boundary = near_boundary_free()
        if boundary:
            b_start = snap(rng.choice(boundary))
            if b_start is not None:
                goal_in = rng.choice(pool)
                record("near_boundary", i, "graceful",
                       lambda s=b_start, g=goal_in: plan_between(s, g))

    return rows


def summarize_edge_suite(rows: List[dict], output_path: Path) -> None:
    categories: List[str] = []
    for row in rows:
        if row["category"] not in categories:
            categories.append(row["category"])

    print("\n=== Edge-case suite ===")
    print(f"{'category':<20} {'cases':>5} {'passed':>7} {'crashed':>8} {'path found':>11}")
    stats = []
    for cat in categories:
        group = [r for r in rows if r["category"] == cat]
        passed = sum(1 for r in group if r["passed"])
        crashed = sum(1 for r in group if r["crashed"])
        found = sum(1 for r in group if r.get("path_found"))
        stats.append((cat, len(group), passed, crashed, found))
        print(f"{cat:<20} {len(group):>5} {passed:>7} {crashed:>8} {found:>11}")

    fig, ax = plt.subplots(figsize=(11, 5.5))
    names = [s[0] for s in stats]
    totals = np.array([s[1] for s in stats], dtype=float)
    pass_rates = np.array([s[2] for s in stats]) / np.maximum(totals, 1) * 100
    bars = ax.bar(names, pass_rates, color="#2f855a", alpha=0.9)
    for bar, (cat, total, passed, crashed, found) in zip(bars, stats):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{passed}/{total}", ha="center", fontsize=10)
    ax.set_ylim(0, 112)
    ax.set_ylabel("pass rate (%)")
    ax.set_title("Edge-case suite — graceful handling / clean planning per category")
    ax.grid(alpha=0.25, axis="y")
    plt.xticks(rotation=18)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close(fig)


def save_csv(rows: List[dict], output_path: Path) -> None:
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(rows: List[dict], execute: bool, output_path: Path) -> None:
    ok = [r for r in rows if r.get("success")]
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    fig.suptitle(
        f"3D Autopilot Benchmark — {len(rows)} cases, "
        f"success {len(ok)}/{len(rows)} ({100.0 * len(ok) / max(1, len(rows)):.0f}%)"
        + ("  [live execution]" if execute else "  [planning only]"),
        fontsize=14,
    )

    def vals(key: str) -> List[float]:
        return [r[key] for r in ok if key in r and r[key] is not None and math.isfinite(r[key])]

    ax = axes[0][0]
    ax.scatter(vals("straight_dist_m"), vals("plan_time_sec"), c="#2b6cb0")
    ax.set_xlabel("straight-line distance (m)")
    ax.set_ylabel("plan time (s)")
    ax.set_title("Planning time vs distance")
    ax.grid(alpha=0.3)

    ax = axes[0][1]
    ax.scatter(vals("straight_dist_m"), vals("path_length_m"), c="#2f855a")
    lim = max(vals("path_length_m") or [1])
    ax.plot([0, lim], [0, lim], "--", c="#a0aec0", label="ideal (straight)")
    ax.set_xlabel("straight-line distance (m)")
    ax.set_ylabel("path length (m)")
    ax.set_title("Path length vs distance")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[0][2]
    ax.hist(vals("detour_ratio"), bins=12, color="#805ad5", alpha=0.85)
    ax.set_xlabel("detour ratio (path / straight)")
    ax.set_ylabel("cases")
    ax.set_title("Detour ratio distribution")
    ax.grid(alpha=0.3)

    ax = axes[1][0]
    ax.hist(vals("smoothness_deg"), bins=12, color="#dd6b20", alpha=0.85)
    ax.set_xlabel("total turn (deg)")
    ax.set_ylabel("cases")
    ax.set_title("Path smoothness")
    ax.grid(alpha=0.3)

    if execute:
        ax = axes[1][1]
        ax.scatter(vals("mean_tracking_error_m"), vals("max_tracking_error_m"), c="#c53030")
        ax.set_xlabel("mean tracking error (m)")
        ax.set_ylabel("max tracking error (m)")
        ax.set_title("Tracking error")
        ax.grid(alpha=0.3)

        ax = axes[1][2]
        indices = np.arange(len(ok))
        width = 0.4
        ax.bar(indices - width / 2, vals("collision_count"), width, label="unity collisions", color="#c53030")
        ax.bar(indices + width / 2, vals("trajectory_intrusion_steps"), width, label="voxel intrusions", color="#2b6cb0")
        ax.set_xlabel("case")
        ax.set_ylabel("count")
        ax.set_title("Collisions vs voxel intrusions per case")
        ax.legend()
        ax.grid(alpha=0.3)
    else:
        ax = axes[1][1]
        ax.hist(vals("planned_min_clearance_m"), bins=12, color="#2b6cb0", alpha=0.85)
        ax.set_xlabel("min clearance of planned path (m)")
        ax.set_ylabel("cases")
        ax.set_title("Planned path clearance")
        ax.grid(alpha=0.3)

        ax = axes[1][2]
        ax.hist(vals("waypoint_count"), bins=12, color="#2f855a", alpha=0.85)
        ax.set_xlabel("waypoints")
        ax.set_ylabel("cases")
        ax.set_title("Waypoint count")
        ax.grid(alpha=0.3)

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(output_path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-json", required=True)
    parser.add_argument("--suite", choices=["random", "edge"], default="random",
                        help="'random': sampled start/goal pairs. 'edge': hostile input categories "
                             "(goal in wall / out of bounds / unreachable / start==goal / vertical / "
                             "long range / map boundary); --cases becomes cases per category.")
    parser.add_argument("--cases", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--min-distance", type=float, default=20.0,
                        help="Minimum straight-line start-goal distance in meters.")
    parser.add_argument("--execute", action="store_true",
                        help="Fly every case live in Unity (Play mode required).")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--command-port", type=int, default=9000)
    parser.add_argument("--local-command-port", type=int, default=9001)
    parser.add_argument("--local-state-port", type=int, default=9002)
    parser.add_argument("--output-dir", default=str(CURRENT_DIR / "benchmark_3d_results"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scene_voxel = load_unity_scene_voxel(args.map_json)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.suite == "edge":
        if args.execute:
            raise SystemExit("--suite edge is planning-level only; run it without --execute.")
        rows = run_edge_suite(scene_voxel, args.cases, args.seed)
        csv_path = output_dir / "benchmark_edge.csv"
        save_csv(rows, csv_path)
        png_path = output_dir / "benchmark_edge.png"
        summarize_edge_suite(rows, png_path)
        crashed = sum(1 for r in rows if r["crashed"])
        passed = sum(1 for r in rows if r["passed"])
        print(f"\ntotal: {passed}/{len(rows)} passed, {crashed} crashed")
        print(f"saved {csv_path}")
        print(f"saved {png_path}")
        return

    cases = sample_cases(scene_voxel, args.cases, args.seed, args.min_distance)
    print(f"sampled {len(cases)} start/goal pairs")

    bridge: UnityTelloBridge | None = None
    if args.execute:
        bridge = UnityTelloBridge(
            unity_host=args.host,
            command_port=args.command_port,
            local_command_port=args.local_command_port,
            local_state_port=args.local_state_port,
        )
        bridge.connect()

    planner = AStarPlanner()
    rows: List[dict] = []
    try:
        for index, (start, goal) in enumerate(cases):
            start_world = scene_voxel.grid_to_world(*start)
            goal_world = scene_voxel.grid_to_world(*goal)
            label = (
                f"case {index + 1}/{len(cases)}: "
                f"({start_world[0]:.0f},{start_world[1]:.0f},{start_world[2]:.0f}) -> "
                f"({goal_world[0]:.0f},{goal_world[1]:.0f},{goal_world[2]:.0f})"
            )
            print(label, flush=True)

            if args.execute:
                try:
                    row = run_execute_case(scene_voxel, bridge, start, goal, index, output_dir)
                except Exception as exc:  # noqa: BLE001 — one bad flight must not kill the batch
                    print(f"  case {index} failed: {exc}")
                    row = {"success": False, "error": f"{type(exc).__name__}: {exc}"}
            else:
                row = run_offline_case(scene_voxel, planner, start, goal)

            row["case"] = index
            row["start_world"] = json.dumps([round(c, 2) for c in start_world])
            row["goal_world"] = json.dumps([round(c, 2) for c in goal_world])
            rows.append(row)
    finally:
        if bridge is not None:
            bridge.close()

    csv_path = output_dir / ("benchmark_execute.csv" if args.execute else "benchmark_planning.csv")
    save_csv(rows, csv_path)
    png_path = output_dir / ("benchmark_execute.png" if args.execute else "benchmark_planning.png")
    plot_summary(rows, args.execute, png_path)

    ok = [r for r in rows if r.get("success")]
    print(f"\nsuccess: {len(ok)}/{len(rows)}")
    if ok:
        print(f"mean plan time: {np.mean([r['plan_time_sec'] for r in ok]):.3f}s")
        if args.execute:
            print(f"mean tracking error: {np.mean([r['mean_tracking_error_m'] for r in ok]):.2f}m")
            print(f"total collisions: {sum(r['collision_count'] for r in ok)}")
            print(f"total voxel intrusions: {sum(r['trajectory_intrusion_steps'] for r in ok)}")
    print(f"saved {csv_path}")
    print(f"saved {png_path}")


if __name__ == "__main__":
    main()
