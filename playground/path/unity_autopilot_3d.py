"""
3D autopilot runner for the Unity Tello simulator.

It plans on a Unity-exported voxel map, saves the planned 3D path for Unity
LineRenderer visualization, and can optionally execute the path.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from integrated_path_benchmark import compute_path_length, compute_smoothness  # noqa: E402
from unity_bridge import DroneState, UnityTelloBridge  # noqa: E402
from unity_scene_voxel import (  # noqa: E402
    IntrusionReport,
    UnitySceneVoxel,
    VoxelPoint,
    check_world_points_against_voxels,
    load_unity_scene_voxel,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DRONE_PATHFINDING_DIR = REPO_ROOT / "drone_pathfinding"
if str(DRONE_PATHFINDING_DIR) not in sys.path:
    sys.path.insert(0, str(DRONE_PATHFINDING_DIR))

from astar import AStarPlanner  # type: ignore  # noqa: E402
from core import Controller  # type: ignore  # noqa: E402


DEFAULT_UNITY_PATH_JSON = (
    REPO_ROOT / "playground" / "tello_simulator" / "Assets" / "Resources" / "planned_path_3d.json"
)
# Written after every executed flight so Unity's FlightReportRenderer can draw the
# flown trajectory and intrusion points directly in the scene.
DEFAULT_UNITY_TRAJ_JSON = (
    REPO_ROOT / "playground" / "tello_simulator" / "Assets" / "Resources" / "flight_trajectory_3d.json"
)


@dataclass
class Autopilot3DResult:
    planner: str
    success: bool
    execute_enabled: bool
    plan_time_sec: float
    execute_time_sec: float
    path_length: float
    smoothness_deg: float
    waypoint_count: int
    mean_tracking_error: float
    max_tracking_error: float
    final_position_error: float
    had_collision: bool
    collision_count: int
    rc_commands_sent: int
    goal_world: Tuple[float, float, float]
    start_world: Tuple[float, float, float]
    # Voxel-map intrusion check (independent of Unity physics).
    # "planned_*" comes from the planned waypoints, "trajectory_*" from the flown path.
    planned_intrusion_steps: int
    planned_intrusion_ratio: float
    planned_min_clearance_m: float
    trajectory_intrusion_steps: int
    trajectory_intrusion_ratio: float
    trajectory_min_clearance_m: float


def grid_path_to_world_waypoints(
    scene_voxel: UnitySceneVoxel,
    path: Sequence[VoxelPoint],
) -> List[Tuple[float, float, float]]:
    return [scene_voxel.grid_to_world(x, y, z) for x, y, z in path]


def count_local_obstacles(scene_voxel: UnitySceneVoxel, point: VoxelPoint, radius: int = 2) -> int:
    x, y, z = point
    hits = 0
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                cell = (x + dx, y + dy, z + dz)
                if scene_voxel.map_.is_in_bounds(cell) and not scene_voxel.map_.is_free(cell):
                    hits += 1
    return hits


def find_reachable_points(scene_voxel: UnitySceneVoxel, start: VoxelPoint) -> set[VoxelPoint]:
    queue = collections.deque([start])
    visited = {start}
    while queue:
        x, y, z = queue.popleft()
        for neighbor in scene_voxel.map_.get_neighbors((x, y, z)):
            neighbor = (int(neighbor[0]), int(neighbor[1]), int(neighbor[2]))
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return visited


def edge_margin_3d(scene_voxel: UnitySceneVoxel, point: VoxelPoint) -> int:
    x, y, z = point
    return min(
        x,
        y,
        z,
        scene_voxel.map_.width - 1 - x,
        scene_voxel.map_.height - 1 - y,
        scene_voxel.map_.depth - 1 - z,
    )


def find_z_extreme_goal_grid(
    scene_voxel: UnitySceneVoxel,
    start_grid: VoxelPoint,
    vertical_band: int = 2,
    min_obstacle_hits: int = 10,
) -> VoxelPoint:
    reachable = find_reachable_points(scene_voxel, start_grid)
    candidates: List[VoxelPoint] = []
    for point in reachable:
        if abs(point[1] - start_grid[1]) > vertical_band:
            continue
        if count_local_obstacles(scene_voxel, point, radius=2) < min_obstacle_hits:
            continue
        candidates.append(point)

    if not candidates:
        candidates = list(reachable)

    min_z_point = min(candidates, key=lambda p: p[2])
    max_z_point = max(candidates, key=lambda p: p[2])
    if abs(start_grid[2] - min_z_point[2]) >= abs(start_grid[2] - max_z_point[2]):
        return min_z_point
    return max_z_point


def find_indoor_upstairs_goal_grid(
    scene_voxel: UnitySceneVoxel,
    start_grid: VoxelPoint,
    min_height_gain_voxels: int = 1,
    min_edge_margin: int = 4,
    min_obstacle_hits: int = 18,
) -> VoxelPoint:
    reachable = find_reachable_points(scene_voxel, start_grid)
    candidates: List[Tuple[float, VoxelPoint]] = []

    for point in reachable:
        height_gain = point[1] - start_grid[1]
        if height_gain < min_height_gain_voxels:
            continue

        margin = edge_margin_3d(scene_voxel, point)
        if margin < min_edge_margin:
            continue

        obstacle_hits = count_local_obstacles(scene_voxel, point, radius=2)
        if obstacle_hits < min_obstacle_hits:
            continue

        dz = abs(point[2] - start_grid[2])
        dx = abs(point[0] - start_grid[0])
        dist = math.dist(start_grid, point)

        # Favor indoor-looking points that are farther in z and also clearly above the current level.
        score = (
            dist * 1.0
            + dz * 0.8
            + height_gain * 8.0
            + margin * 1.5
            + obstacle_hits * 0.15
            - dx * 0.15
        )
        candidates.append((score, point))

    if candidates:
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

    # Fallback 1: same building but farther from edges.
    interior_candidates: List[Tuple[float, VoxelPoint]] = []
    for point in reachable:
        margin = edge_margin_3d(scene_voxel, point)
        if margin < min_edge_margin:
            continue
        obstacle_hits = count_local_obstacles(scene_voxel, point, radius=2)
        if obstacle_hits < min_obstacle_hits:
            continue
        dz = abs(point[2] - start_grid[2])
        dist = math.dist(start_grid, point)
        score = dist + dz * 0.8 + margin * 1.5 + obstacle_hits * 0.15
        interior_candidates.append((score, point))

    if interior_candidates:
        interior_candidates.sort(key=lambda item: item[0], reverse=True)
        return interior_candidates[0][1]

    # Fallback 2: previous same-floor z-extreme heuristic.
    return find_z_extreme_goal_grid(scene_voxel, start_grid)


def world_velocity_to_rc(state: DroneState, velocity_world: np.ndarray, max_speed: float) -> Tuple[int, int, int, int]:
    yaw_rad = math.radians(state.yaw)
    world_x = float(velocity_world[0])
    world_y = float(velocity_world[1])
    world_z = float(velocity_world[2])

    local_x = math.cos(yaw_rad) * world_x - math.sin(yaw_rad) * world_z
    local_z = math.sin(yaw_rad) * world_x + math.cos(yaw_rad) * world_z

    scale = 100.0 / max_speed
    lr = int(max(-100, min(100, round(local_x * scale))))
    fb = int(max(-100, min(100, round(local_z * scale))))
    ud = int(max(-100, min(100, round(world_y * scale))))
    return lr, fb, ud, 0


def save_unity_path_preview(
    output_path: Path,
    planner: str,
    world_waypoints: Sequence[Tuple[float, float, float]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "planner": planner,
        "path_world": [{"x": p[0], "y": p[1], "z": p[2]} for p in world_waypoints],
        "start_world": {"x": world_waypoints[0][0], "y": world_waypoints[0][1], "z": world_waypoints[0][2]},
        "goal_world": {"x": world_waypoints[-1][0], "y": world_waypoints[-1][1], "z": world_waypoints[-1][2]},
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _json_safe(obj):
    """Recursively replace inf/-inf/nan with None so the output is valid JSON
    (Unity JsonUtility and strict parsers reject the default `Infinity` token)."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def save_trajectory(
    output_path: Path,
    planner: str,
    trajectory: Sequence[Tuple[float, float, float]],
    world_waypoints: Sequence[Tuple[float, float, float]],
    report: IntrusionReport,
    scene_voxel: UnitySceneVoxel | None = None,
) -> None:
    """Dump the flown trajectory + planned path for offline inspection/visualization."""
    intrusions: List[Tuple[float, float, float]] = []
    if scene_voxel is not None:
        for point in trajectory:
            grid = scene_voxel.world_to_grid(*point)
            if scene_voxel.map_.is_in_bounds(grid) and not scene_voxel.map_.is_free(grid):
                intrusions.append(point)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "planner": planner,
        "intrusion_steps": report.intrusion_points,
        "intrusion_ratio": report.intrusion_ratio,
        "min_clearance_m": report.min_clearance_m,
        "trajectory_world": [{"x": p[0], "y": p[1], "z": p[2]} for p in trajectory],
        "path_world": [{"x": p[0], "y": p[1], "z": p[2]} for p in world_waypoints],
        "intrusions_world": [{"x": p[0], "y": p[1], "z": p[2]} for p in intrusions],
    }
    output_path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")


def run_autopilot_3d(
    scene_voxel: UnitySceneVoxel,
    bridge: UnityTelloBridge | None,
    goal_x: float | None,
    goal_y: float | None,
    goal_z: float | None,
    execute: bool,
    unity_path_json: Path,
    start_x: float | None = None,
    start_y: float | None = None,
    start_z: float | None = None,
    trajectory_out: Path | None = None,
    max_speed: float = 4.0,
    dt: float = 0.05,
    arrival_threshold: float = 1.2,
    timeout_sec: float = 120.0,
) -> Autopilot3DResult:
    planner_name = "astar_3d"
    planner = AStarPlanner()

    start_specified = start_x is not None and start_y is not None and start_z is not None

    # A live bridge is needed to execute, and to read the current drone position
    # when an explicit start is not given. Pure planning + preview can run offline.
    if execute and bridge is None:
        raise RuntimeError("--execute requires a live Unity bridge.")
    if bridge is None and not start_specified:
        raise RuntimeError(
            "Provide --start-x/--start-y/--start-z, or run with a live Unity bridge "
            "so the current drone position can be used as the start."
        )

    if bridge is not None:
        init_reply = bridge.initialize_sdk()
        if init_reply not in {"ok", "timeout"}:
            raise RuntimeError(f"Simulator handshake failed: {init_reply}")

    if start_specified:
        start_world = (float(start_x), float(start_y), float(start_z))
    else:
        state = bridge.wait_for_state(timeout=3.0)
        if state is None:
            raise RuntimeError("No simulator state received. Check Unity play mode and state port.")
        start_world = (state.x, state.y, state.z)

    start_grid = scene_voxel.clamp_grid_point(scene_voxel.world_to_grid(*start_world))
    nearest_start = scene_voxel.find_nearest_free(start_grid)
    if nearest_start is None:
        raise RuntimeError("Could not find a nearby free 3D voxel around the start position.")
    start_grid = nearest_start

    if goal_x is None or goal_y is None or goal_z is None:
        min_height_gain_voxels = max(1, int(round(2.0 / scene_voxel.voxel_size)))
        goal_grid = find_indoor_upstairs_goal_grid(
            scene_voxel,
            start_grid,
            min_height_gain_voxels=min_height_gain_voxels,
        )
        goal_world = scene_voxel.grid_to_world(*goal_grid)
        goal_x, goal_y, goal_z = goal_world
    else:
        requested_goal = scene_voxel.clamp_grid_point(scene_voxel.world_to_grid(goal_x, goal_y, goal_z))
        nearest_goal = scene_voxel.find_nearest_free(requested_goal)
        if nearest_goal is None:
            raise RuntimeError("Could not find a nearby free 3D voxel around the requested goal.")
        goal_grid = nearest_goal
        goal_x, goal_y, goal_z = scene_voxel.grid_to_world(*goal_grid)

    t0 = time.perf_counter()
    path = planner.plan(scene_voxel.map_, start_grid, goal_grid)
    plan_time = time.perf_counter() - t0
    if not path:
        raise RuntimeError("3D A* could not find a path on the exported voxel map.")

    world_waypoints = grid_path_to_world_waypoints(scene_voxel, path)
    world_waypoints[0] = start_world
    world_waypoints[-1] = (goal_x, goal_y, goal_z)
    save_unity_path_preview(unity_path_json, planner_name, world_waypoints)

    # Verification (independent of Unity physics): does the *planned* path itself
    # pass through occupied voxels? Non-zero here points at the planner/coordinate
    # mapping (cause A); zero here but non-zero on the flown trajectory points at
    # tracking error / sim collision (cause B).
    planned_report = check_world_points_against_voxels(scene_voxel, world_waypoints)

    if not execute:
        return Autopilot3DResult(
            planner=planner_name,
            success=True,
            execute_enabled=False,
            plan_time_sec=plan_time,
            execute_time_sec=0.0,
            path_length=compute_path_length(world_waypoints),
            smoothness_deg=compute_smoothness(world_waypoints),
            waypoint_count=len(world_waypoints),
            mean_tracking_error=0.0,
            max_tracking_error=0.0,
            final_position_error=0.0,
            had_collision=False,
            collision_count=0,
            rc_commands_sent=0,
            goal_world=world_waypoints[-1],
            start_world=start_world,
            planned_intrusion_steps=planned_report.intrusion_points,
            planned_intrusion_ratio=planned_report.intrusion_ratio,
            planned_min_clearance_m=planned_report.min_clearance_m,
            trajectory_intrusion_steps=0,
            trajectory_intrusion_ratio=0.0,
            trajectory_min_clearance_m=float("inf"),
        )

    if start_specified:
        # Move the simulator drone to the requested start so the flight actually
        # begins there (planning start and physical start stay consistent).
        setpos_reply = bridge.set_position(*start_world)
        if setpos_reply not in {"ok", "timeout"}:
            raise RuntimeError(f"setpos failed: {setpos_reply}")
        time.sleep(0.2)

    takeoff_reply = bridge.takeoff()
    if takeoff_reply not in {"ok", "timeout"}:
        raise RuntimeError(f"Takeoff failed: {takeoff_reply}")

    time.sleep(1.0)
    state = bridge.wait_for_state(timeout=2.0)
    if state is None:
        raise RuntimeError("No state after takeoff.")

    world_waypoints[0] = (state.x, state.y, state.z)
    controller = Controller(
        waypoints=world_waypoints,
        kp=1.4,
        ki=0.01,
        kd=0.35,
        arrival_threshold=arrival_threshold,
    )

    tracking_errors: List[float] = []
    trajectory: List[Tuple[float, float, float]] = [(state.x, state.y, state.z)]
    rc_count = 0
    execute_start = time.perf_counter()
    success = False
    had_collision = False
    collision_count = 0

    while time.perf_counter() - execute_start < timeout_sec:
        state = bridge.wait_for_state(timeout=1.0)
        if state is None:
            continue

        had_collision = had_collision or state.had_collision
        collision_count = max(collision_count, state.collision_count)

        current_pos = np.array([state.x, state.y, state.z], dtype=float)
        trajectory.append((state.x, state.y, state.z))
        velocity = controller.compute(tuple(current_pos), dt=dt)
        speed = float(np.linalg.norm(velocity))
        if speed > max_speed:
            velocity = velocity / speed * max_speed

        rc = world_velocity_to_rc(state, velocity, max_speed=max_speed)
        bridge.send_rc(*rc)
        rc_count += 1

        target = np.array(controller.target, dtype=float)
        tracking_errors.append(float(np.linalg.norm(target - current_pos)))

        if controller.is_done:
            success = True
            break

        time.sleep(dt)

    bridge.send_rc(0, 0, 0, 0)
    time.sleep(0.2)
    final_state = bridge.wait_for_state(timeout=1.0) or state
    bridge.land()

    final_position_error = float(
        np.linalg.norm(
            np.array([final_state.x, final_state.y, final_state.z], dtype=float)
            - np.array(world_waypoints[-1], dtype=float)
        )
    )
    had_collision = had_collision or final_state.had_collision
    collision_count = max(collision_count, final_state.collision_count)

    trajectory.append((final_state.x, final_state.y, final_state.z))

    # Did the *flown* path pass through occupied voxels, regardless of whether
    # Unity's collision callback fired?
    trajectory_report = check_world_points_against_voxels(scene_voxel, trajectory)

    # Always drop a copy where Unity's FlightReportRenderer looks, so the flown
    # trajectory + intrusion points appear in the scene right after landing.
    save_trajectory(DEFAULT_UNITY_TRAJ_JSON, planner_name, trajectory, world_waypoints,
                    trajectory_report, scene_voxel)
    if trajectory_out is not None:
        save_trajectory(trajectory_out, planner_name, trajectory, world_waypoints,
                        trajectory_report, scene_voxel)

    return Autopilot3DResult(
        planner=planner_name,
        success=success,
        execute_enabled=True,
        plan_time_sec=plan_time,
        execute_time_sec=time.perf_counter() - execute_start,
        path_length=compute_path_length(world_waypoints),
        smoothness_deg=compute_smoothness(world_waypoints),
        waypoint_count=len(world_waypoints),
        mean_tracking_error=float(np.mean(tracking_errors)) if tracking_errors else float("inf"),
        max_tracking_error=float(np.max(tracking_errors)) if tracking_errors else float("inf"),
        final_position_error=final_position_error,
        had_collision=had_collision,
        collision_count=collision_count,
        rc_commands_sent=rc_count,
        goal_world=world_waypoints[-1],
        start_world=start_world,
        planned_intrusion_steps=planned_report.intrusion_points,
        planned_intrusion_ratio=planned_report.intrusion_ratio,
        planned_min_clearance_m=planned_report.min_clearance_m,
        trajectory_intrusion_steps=trajectory_report.intrusion_points,
        trajectory_intrusion_ratio=trajectory_report.intrusion_ratio,
        trajectory_min_clearance_m=trajectory_report.min_clearance_m,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-json", required=True, help="Path to the exported 3D voxel JSON.")
    parser.add_argument("--start-x", type=float, help="Explicit start X (Unity world). Requires --start-y/--start-z.")
    parser.add_argument("--start-y", type=float)
    parser.add_argument("--start-z", type=float)
    parser.add_argument("--goal-x", type=float)
    parser.add_argument("--goal-y", type=float)
    parser.add_argument("--goal-z", type=float)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--command-port", type=int, default=9000)
    parser.add_argument("--local-command-port", type=int, default=9001)
    parser.add_argument("--local-state-port", type=int, default=9002)
    parser.add_argument("--execute", action="store_true", help="Actually fly the planned 3D path in Unity.")
    parser.add_argument("--path-json", default=str(DEFAULT_UNITY_PATH_JSON))
    parser.add_argument("--output", default=str(CURRENT_DIR / "unity_autopilot_3d_result.json"))
    parser.add_argument(
        "--trajectory-out",
        default=None,
        help="When executing, save the flown trajectory + planned path to this JSON for inspection.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scene_voxel = load_unity_scene_voxel(args.map_json)

    start_specified = args.start_x is not None and args.start_y is not None and args.start_z is not None

    # Offline planning (no --execute) with an explicit start does not need Unity running.
    bridge: UnityTelloBridge | None = None
    if args.execute or not start_specified:
        bridge = UnityTelloBridge(
            unity_host=args.host,
            command_port=args.command_port,
            local_command_port=args.local_command_port,
            local_state_port=args.local_state_port,
        )
        bridge.connect()

    try:
        result = run_autopilot_3d(
            scene_voxel=scene_voxel,
            bridge=bridge,
            goal_x=args.goal_x,
            goal_y=args.goal_y,
            goal_z=args.goal_z,
            execute=args.execute,
            unity_path_json=Path(args.path_json),
            start_x=args.start_x,
            start_y=args.start_y,
            start_z=args.start_z,
            trajectory_out=Path(args.trajectory_out) if args.trajectory_out else None,
        )
    finally:
        if bridge is not None:
            bridge.close()

    output_path = Path(args.output)
    safe_result = _json_safe(asdict(result))
    output_path.write_text(json.dumps(safe_result, indent=2), encoding="utf-8")
    print(json.dumps(safe_result, indent=2))
    print(f"\nSaved 3D autopilot metrics to {output_path}")


if __name__ == "__main__":
    main()
