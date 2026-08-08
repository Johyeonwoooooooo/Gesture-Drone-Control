"""
Waypoint-following flight over the Unity Tello simulator's UDP `rc` protocol.

PID controller and rc mapping are adapted from the jiyun-simul branch
(`drone_pathfinding/core.py` Controller and `playground/path/unity_autopilot_3d.py`
world_velocity_to_rc, both on THAT branch) but kept dependency-free here:
numpy + unity_bridge only.

All positions/velocities here are Unity world units ("u"). The house glb is
placed at scale 5, so 1 u = 0.2 m of real-house scale.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

try:
    from .unity_bridge import DroneState, UnityTelloBridge
except ImportError:  # run as a plain script from simulator/bridge/
    from unity_bridge import DroneState, UnityTelloBridge

# TelloSimulator.cs moveSpeed: rc=100 commands 15 u/s. Coupled to the C# field.
#
# The in-sim settings panel (SettingsPanel.cs, Tab) can now change moveSpeed at
# runtime, and it does NOT tell us — so at a panel ratio of r the drone actually
# flies r x the `max_speed` asked for below. follow_path closes on the reported
# position, so the flight still lands on its waypoints; only the u/s scale is
# nominal (and corners overshoot more). Keep the panel at 기본값 whenever the
# absolute speed matters, e.g. when timing a mission or tuning arrival_threshold.
UNITY_MOVE_SPEED = 15.0


@dataclass
class FollowResult:
    success: bool
    reason: str                  # arrived | timeout | state_lost | collision_abort
    rc_commands_sent: int
    collision_count: int
    final_error_u: float
    duration_s: float
    trajectory_unity: list = field(default_factory=list)


class WaypointPID:
    """PID over 3D position error with per-waypoint arrival advance."""

    def __init__(
        self,
        waypoints: Sequence[Sequence[float]],
        kp: float = 1.2,
        ki: float = 0.0,
        kd: float = 0.3,
        arrival_threshold: float = 0.8,
    ) -> None:
        self.waypoints = [np.asarray(w, dtype=float) for w in waypoints]
        self.current_idx = 0
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.arrival_threshold = arrival_threshold
        self._integral = np.zeros(3)
        self._prev_error = np.zeros(3)

    @property
    def target(self) -> np.ndarray:
        idx = min(self.current_idx, len(self.waypoints) - 1)
        return self.waypoints[idx]

    @property
    def is_done(self) -> bool:
        return self.current_idx >= len(self.waypoints)

    def compute(self, pos: np.ndarray, dt: float) -> np.ndarray:
        if self.is_done:
            return np.zeros(3)
        error = self.target - pos
        if np.linalg.norm(error) < self.arrival_threshold:
            self.current_idx += 1
            self._integral[:] = 0.0
            self._prev_error[:] = 0.0
            return self.compute(pos, dt)
        self._integral += error * dt
        derivative = (error - self._prev_error) / dt if dt > 0 else np.zeros(3)
        self._prev_error = error.copy()
        return self.kp * error + self.ki * self._integral + self.kd * derivative


# rc 의 yaw 성분은 곧 deg/s 다: TelloSimulator 가 yaw/100 을 targetYaw 로 받고
# rotationSpeed(100) 를 곱해 돌리므로 rc 50 이 50 deg/s 가 된다. 아래 상수들이
# 그 단위다.
FACE_TRAVEL_GAIN = 1.5          # 오차 1° 당 deg/s
FACE_TRAVEL_RC_LIMIT = 45       # deg/s. 스캔이 50 deg/s 라 그보다 살짝 아래
FACE_TRAVEL_DEADBAND_DEG = 6.0  # 이 안이면 안 돌린다 (제자리 떨림 방지)
FACE_TRAVEL_MIN_SPEED = 0.3     # u/s. 거의 정지 상태에서 기수가 헤매지 않게

# 이 각도 밖이면 **수평 병진을 멈추고 기수부터 맞춘다.** P 제어만으로는 코너를
# 못 따라간다 — 상한이 45 deg/s 라 90° 코너 하나를 도는 데 2초가 걸리는데 그동안
# 병진은 최고 속도로 계속 나가므로 코너마다 게걸음이 된다. 실측(ㄷ 자 경로,
# fake_unity_sim): 이동 시간의 41% 가 15° 밖, 23% 가 45° 밖, 최대 91°.
# 회전을 먼저 끝내면 그 구간이 사라진다. 대신 코너마다 1~2초 멈춘다.
#
# 수직(ud)은 막지 않는다. 고도는 기수와 무관하고, 같이 막으면 층 이동이
# 회전 대기에 묶인다.
FACE_TRAVEL_MOVE_TOLERANCE_DEG = 20.0


def world_velocity_to_rc(
    yaw_deg: float,
    v_world: np.ndarray,
    rc_limit: int = 30,
    face_travel: bool = True,
) -> tuple[int, int, int, int]:
    """Unity-world velocity (u/s) -> Tello rc, normalized by UNITY_MOVE_SPEED so
    the commanded rc reproduces v_world exactly (until rc_limit clips it).

    `face_travel` 이 켜져 있으면 기수를 진행 방향으로 돌리는 yaw 성분을 같이
    싣고, **크게 어긋나 있으면 수평 병진을 멈추고 회전을 먼저 끝낸다**
    (`FACE_TRAVEL_MOVE_TOLERANCE_DEG`). 예전엔 여기서 yaw 를 항상 0 으로
    돌려줘서 드론이 마지막 스캔이 끝난 각도 그대로 옆으로·뒤로 게걸음을 쳤고,
    yaw 를 실은 뒤에도 회전이 코너를 못 따라가 코너마다 게걸음이 남았다.
    3인칭 화면에서는 티가 안 나지만 탐지 카메라가 드론 1인칭이라 그대로
    드러난다 — 이동 내내 벽만 본다.

    경로 자체는 그대로다. 병진을 **줄이는** 게 아니라 정렬될 때까지 **미루는**
    것이라, 목표 속도 벡터는 손대지 않고 웨이포인트도 그대로 지난다."""
    yaw = math.radians(yaw_deg)
    wx, wy, wz = float(v_world[0]), float(v_world[1]), float(v_world[2])
    local_x = math.cos(yaw) * wx - math.sin(yaw) * wz
    local_z = math.sin(yaw) * wx + math.cos(yaw) * wz
    scale = 100.0 / UNITY_MOVE_SPEED

    def clip(v: float) -> int:
        return int(max(-rc_limit, min(rc_limit, round(v * scale))))

    yaw_rc = 0
    align_first = False
    if face_travel and math.hypot(wx, wz) >= FACE_TRAVEL_MIN_SPEED:
        # Unity 의 eulerAngles.y 와 같은 규약: 0° 가 +Z, +X 쪽으로 증가한다.
        desired_deg = math.degrees(math.atan2(wx, wz))
        error_deg = (desired_deg - yaw_deg + 180.0) % 360.0 - 180.0
        if abs(error_deg) > FACE_TRAVEL_DEADBAND_DEG:
            yaw_rc = int(max(-FACE_TRAVEL_RC_LIMIT,
                             min(FACE_TRAVEL_RC_LIMIT,
                                 round(error_deg * FACE_TRAVEL_GAIN))))
        align_first = abs(error_deg) > FACE_TRAVEL_MOVE_TOLERANCE_DEG

    if align_first:
        # 수평만 0. 고도는 계속 간다. 다음 틱에 PID 가 같은 속도 벡터를 다시
        # 주므로 오차가 허용치 안으로 들어오는 순간 병진이 이어진다.
        return 0, 0, clip(wy), yaw_rc

    return clip(local_x), clip(local_z), clip(wy), yaw_rc


def follow_path(
    bridge: UnityTelloBridge,
    waypoints_unity: Sequence[Sequence[float]],
    *,
    max_speed: float = 2.0,          # u/s (= 0.4 m/s at house scale 5)
    rc_limit: int = 30,
    dt: float = 0.05,
    arrival_threshold: float = 0.8,  # u
    timeout_sec: Optional[float] = None,
    abort_on_collision: bool = False,
    state_stall_sec: float = 5.0,
    on_state: Optional[Callable[[DroneState], None]] = None,
    on_status: Callable[[str], None] = print,
) -> FollowResult:
    """Drive rc commands until the last waypoint is reached (or a failure)."""
    waypoints = [np.asarray(w, dtype=float) for w in waypoints_unity]
    # kp 를 max_speed 에 묶는다. A* 웨이포인트가 0.15 m 격자라 다음 점까지의
    # 오차가 threshold 언저리를 못 벗어나고, 고정 kp 로는 kp*오차 가 곧 순항
    # 속도라 --sim-speed 를 올려도 안 빨라진다. 오차 = threshold 일 때 딱
    # max_speed 가 나오게 두면 순항은 max_speed 로 포화하고 도착 직전에만
    # 감속한다. 드론은 rc = 속도인 기구학 모델이라 관성 오버슈트는 없다.
    pid = WaypointPID(waypoints,
                      kp=max_speed / max(arrival_threshold, 1e-3),
                      arrival_threshold=arrival_threshold)

    path_len = float(
        sum(np.linalg.norm(b - a) for a, b in zip(waypoints[:-1], waypoints[1:]))
    )
    if timeout_sec is None:
        timeout_sec = min(600.0, max(60.0, 4.0 * path_len / max_speed))

    start_time = time.time()
    last_state_time = -1.0
    last_fresh_wall = start_time
    last_on_state_wall = 0.0
    baseline_collisions: Optional[int] = None
    collision_count = 0
    rc_sent = 0
    trajectory: list = []

    def stop_and_result(success: bool, reason: str, pos: Optional[np.ndarray]) -> FollowResult:
        try:
            bridge.send_rc(0, 0, 0, 0)
        except Exception:
            pass
        err = float(np.linalg.norm(waypoints[-1] - pos)) if pos is not None else float("inf")
        return FollowResult(
            success=success,
            reason=reason,
            rc_commands_sent=rc_sent,
            collision_count=collision_count,
            final_error_u=err,
            duration_s=time.time() - start_time,
            trajectory_unity=trajectory,
        )

    pos: Optional[np.ndarray] = None
    while True:
        now = time.time()
        if now - start_time > timeout_sec:
            on_status(f"flight timeout after {timeout_sec:.0f}s")
            return stop_and_result(False, "timeout", pos)

        state = bridge.get_latest_state()
        if state is None or state.time == last_state_time:
            if now - last_fresh_wall > state_stall_sec:
                on_status("simulator state lost")
                return stop_and_result(False, "state_lost", pos)
            time.sleep(dt)
            continue
        last_state_time = state.time
        last_fresh_wall = now

        pos = np.array([state.x, state.y, state.z], dtype=float)
        trajectory.append((state.x, state.y, state.z))

        if baseline_collisions is None:
            baseline_collisions = state.collision_count
        if state.collision_count - baseline_collisions > collision_count:
            collision_count = state.collision_count - baseline_collisions
            on_status(f"collision #{collision_count}")
            if abort_on_collision:
                return stop_and_result(False, "collision_abort", pos)

        if on_state is not None and now - last_on_state_wall >= 0.2:
            last_on_state_wall = now
            try:
                on_state(state)
            except Exception:
                pass

        velocity = pid.compute(pos, dt)
        if pid.is_done:
            return stop_and_result(True, "arrived", pos)

        speed = float(np.linalg.norm(velocity))
        if speed > max_speed:
            velocity = velocity * (max_speed / speed)
        lr, fb, ud, yaw_rc = world_velocity_to_rc(state.yaw, velocity, rc_limit)
        bridge.send_rc(lr, fb, ud, yaw_rc)
        rc_sent += 1
        time.sleep(dt)


def fly_mission(
    bridge: UnityTelloBridge,
    waypoints_unity: Sequence[Sequence[float]],
    *,
    setpos_start: bool = True,
    land_at_end: bool = True,
    on_state: Optional[Callable[[DroneState], None]] = None,
    on_status: Callable[[str], None] = print,
    **follow_kwargs,
) -> FollowResult:
    """setpos(wp0) -> takeoff -> follow -> land, with a safe-stop on any error."""
    waypoints = [np.asarray(w, dtype=float) for w in waypoints_unity]
    if len(waypoints) < 1:
        raise ValueError("empty waypoint list")

    try:
        reply = bridge.initialize_sdk()
        if reply == "timeout":
            on_status("simulator not reachable (command timeout)")
            return FollowResult(False, "state_lost", 0, 0, float("inf"), 0.0, [])

        if setpos_start:
            wp0 = waypoints[0]
            bridge.set_position(float(wp0[0]), float(wp0[1]), float(wp0[2]), 0.0)
        bridge.takeoff()

        state = bridge.wait_for_state(timeout=5.0)
        if state is None:
            on_status("no state stream from simulator (check UDP 9002 inbound)")
            return FollowResult(False, "state_lost", 0, 0, float("inf"), 0.0, [])
        # Takeoff lifts the drone above wp0; retarget the first waypoint to the
        # actual hover position so the PID doesn't fight the takeoff altitude.
        waypoints[0] = np.array([state.x, state.y, state.z], dtype=float)

        result = follow_path(
            bridge, waypoints, on_state=on_state, on_status=on_status, **follow_kwargs
        )
        return result
    finally:
        try:
            bridge.send_rc(0, 0, 0, 0)
            time.sleep(0.3)
            if land_at_end:
                bridge.land()
        except Exception:
            pass
