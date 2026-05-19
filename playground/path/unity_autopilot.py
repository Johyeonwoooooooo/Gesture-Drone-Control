"""
Autopilot runner for the Unity Tello simulator.

Usage example:
    python playground/path/unity_autopilot.py ^
        --map-json exported_scene.json ^
        --planner hybrid ^
        --goal-x 12.0 --goal-z 18.0
"""

from __future__ import annotations

import argparse
import json
import math
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
from unity_bridge import DroneState, UnityTelloBridge  # noqa: E402
from unity_scene_map import UnitySceneGrid, load_unity_scene_grid  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DRONE_PATHFINDING_DIR = REPO_ROOT / "drone_pathfinding"
if str(DRONE_PATHFINDING_DIR) not in sys.path:
    sys.path.insert(0, str(DRONE_PATHFINDING_DIR))

from astar import AStarPlanner  # type: ignore  # noqa: E402
from core import Controller  # type: ignore  # noqa: E402
from rrt import RRTStarPlanner  # type: ignore  # noqa: E402


@dataclass
class AutopilotResult:
    planner: str
    success: bool
    plan_time_sec: float
    execute_time_sec: float
    path_length: float
    smoothness_deg: float
    waypoint_count: int
    mean_tracking_error: float
    max_tracking_error: float
    final_position_error: float
    rc_commands_sent: int
    goal_world: Tuple[float, float, float]
    start_world: Tuple[float, float, float]


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


def grid_path_to_world_waypoints(
    scene_grid: UnitySceneGrid,
    path: Sequence[Tuple[float, float]],
    flight_height: float,
) -> List[Tuple[float, float, float]]:
    waypoints: List[Tuple[float, float, float]] = []
    for gx, gy in path:
        wx, wz = scene_grid.grid_to_world(gx, gy)
        waypoints.append((wx, flight_height, wz))
    return waypoints


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


def run_autopilot(
    scene_grid: UnitySceneGrid,
    planner_name: str,
    goal_x: float,
    goal_z: float,
    bridge: UnityTelloBridge,
    takeoff_height_offset: float = 1.0,
    max_speed: float = 4.0,
    dt: float = 0.05,
    arrival_threshold: float = 1.0,
    timeout_sec: float = 90.0,
) -> AutopilotResult:
    init_reply = bridge.initialize_sdk()
    if init_reply not in {"ok", "timeout"}:
        raise RuntimeError(f"Simulator handshake failed: {init_reply}")

    state = bridge.wait_for_state(timeout=3.0)
    if state is None:
        raise RuntimeError("No simulator state received. Check Unity play mode and state port.")

    takeoff_reply = bridge.takeoff()
    if takeoff_reply not in {"ok", "timeout"}:
        raise RuntimeError(f"Takeoff failed: {takeoff_reply}")

    time.sleep(1.0)
    state = bridge.wait_for_state(timeout=2.0)
    if state is None:
        raise RuntimeError("No state after takeoff.")

    start_world = (state.x, state.y, state.z)
    flight_height = max(scene_grid.flight_height, state.y + takeoff_height_offset)
    start_grid = scene_grid.clamp_grid_point(scene_grid.world_to_grid(state.x, state.z))
    goal_grid = scene_grid.clamp_grid_point(scene_grid.world_to_grid(goal_x, goal_z))

    planner = planner_from_name(planner_name)
    t0 = time.perf_counter()
    path = planner.plan(scene_grid.map_, start_grid, goal_grid)
    plan_time = time.perf_counter() - t0
    if not path:
        bridge.land()
        raise RuntimeError("Planner could not find a path on the exported scene grid.")

    path = smooth_path(scene_grid.map_, path)
    world_waypoints = grid_path_to_world_waypoints(scene_grid, path, flight_height)
    world_waypoints[0] = (state.x, flight_height, state.z)
    world_waypoints[-1] = (goal_x, flight_height, goal_z)

    controller = Controller(
        waypoints=world_waypoints,
        kp=1.4,
        ki=0.01,
        kd=0.35,
        arrival_threshold=arrival_threshold,
    )

    tracking_errors: List[float] = []
    rc_count = 0
    execute_start = time.perf_counter()
    success = False

    while time.perf_counter() - execute_start < timeout_sec:
        state = bridge.wait_for_state(timeout=1.0)
        if state is None:
            continue

        current_pos = np.array([state.x, state.y, state.z], dtype=float)
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

    return AutopilotResult(
        planner=planner_name,
        success=success,
        plan_time_sec=plan_time,
        execute_time_sec=time.perf_counter() - execute_start,
        path_length=compute_path_length(world_waypoints),
        smoothness_deg=compute_smoothness(world_waypoints),
        waypoint_count=len(world_waypoints),
        mean_tracking_error=float(np.mean(tracking_errors)) if tracking_errors else float("inf"),
        max_tracking_error=float(np.max(tracking_errors)) if tracking_errors else float("inf"),
        final_position_error=final_position_error,
        rc_commands_sent=rc_count,
        goal_world=world_waypoints[-1],
        start_world=start_world,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-json", required=True, help="Path to the exported occupancy grid JSON.")
    parser.add_argument("--planner", default="hybrid", choices=["astar", "rrt", "hybrid"])
    parser.add_argument("--goal-x", type=float, required=True)
    parser.add_argument("--goal-z", type=float, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--command-port", type=int, default=9000)
    parser.add_argument("--local-command-port", type=int, default=9001)
    parser.add_argument("--local-state-port", type=int, default=9002)
    parser.add_argument("--output", default=str(CURRENT_DIR / "unity_autopilot_result.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scene_grid = load_unity_scene_grid(args.map_json)
    bridge = UnityTelloBridge(
        unity_host=args.host,
        command_port=args.command_port,
        local_command_port=args.local_command_port,
        local_state_port=args.local_state_port,
    )
    bridge.connect()
    try:
        result = run_autopilot(
            scene_grid=scene_grid,
            planner_name=args.planner,
            goal_x=args.goal_x,
            goal_z=args.goal_z,
            bridge=bridge,
        )
    finally:
        bridge.close()

    output_path = Path(args.output)
    output_path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    print(json.dumps(asdict(result), indent=2))
    print(f"\nSaved autopilot metrics to {output_path}")


if __name__ == "__main__":
    main()
