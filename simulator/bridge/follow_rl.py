"""follow_rl.py — A* 경로를 **강화학습 정책**으로 따라 난다 (PID 대체).

`follow_path.follow_path` 와 같은 자리에 들어가고 같은 `FollowResult` 를 준다.
다른 건 rc 를 만드는 방식뿐이다:

    follow_path : 웨이포인트 오차 → PID → 속도 → rc
    follow_rl   : 레이 14개 + 2.5 m 앞 carrot → SAC 정책 → 속도 → rc

**정책은 계획기가 아니다.** 학습 때 `--subgoal 2.5` 로 관측의 목표를 '최단경로
2.5 m 앞 지점(carrot)'으로 바꿔 놨기 때문에, 전역 경로는 여전히 밖에서 줘야
한다 (`planner.plan_path` 의 A* 를 그대로 쓴다). 정책이 하는 일은 그 경로를
따라가면서 주변을 보고 피하는 것 — 즉 추종기(controller)다.

관측 21차원은 학습 때(`geo_env._obs`)와 같아야 한다:

    [carrot 단위벡터 3][carrot 거리/집대각 1][레이 14][직전 행동 3]

레이는 학습과 같은 장애물 점군(방별 `coord.npy` 를 0.06 m 로 다운샘플)에 대한
KDTree 거리 판정이다. 값이 어긋나면 성공률이 조용히 떨어지므로 `--check` 가
`geo_env` 원본과 직접 대조한다 (conda env `tello` 에서).

    conda activate tello
    python simulator/bridge/follow_rl.py --check
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

try:
    from .follow_path import FollowResult, world_velocity_to_rc
    from .rl_policy import NumpyActor, DEFAULT_NPZ
except ImportError:   # run as a plain script from simulator/bridge/
    from follow_path import FollowResult, world_velocity_to_rc
    from rl_policy import NumpyActor, DEFAULT_NPZ

_REPO = Path(__file__).resolve().parents[2]
_OBSTACLE_CACHE = Path(__file__).resolve().parent / "geo_obstacles.npz"

# 학습 때 쓴 값 (train_geo/evaluate_geo 플래그와 geo_env 기본값).
# ★ 하나라도 다르면 관측 분포가 어긋나 성공률이 떨어진다. 바꾸지 말 것.
OBSTACLE_VOXEL = 0.06     # load_obstacles(voxel)
CLEARANCE = 0.12          # --clearance
RAY_MAX = 4.0             # --ray-max
RAY_STEP = 0.15           # geo_env ray_step 기본값
SUBGOAL_DIST = 2.5        # --subgoal (carrot 거리)
MAX_STEP = 0.30           # 정책 행동 1스텝이 움직이는 최대 거리 (m)
BOUNDS_MARGIN = 0.3       # geo_env 의 bounds 여유 (diag 계산용)


def ray_directions_horiz14() -> np.ndarray:
    """--rays horiz14: 수평 12방향(30°) + 위/아래."""
    dirs = [(np.cos(2 * np.pi * k / 12), np.sin(2 * np.pi * k / 12), 0.0)
            for k in range(12)]
    dirs += [(0.0, 0.0, 1.0), (0.0, 0.0, -1.0)]
    v = np.array(dirs, dtype=np.float64)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _voxel_down(coord: np.ndarray, v: float) -> np.ndarray:
    keys = np.floor(coord / v).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return coord[idx]


@dataclass
class GeoWorld:
    """정책이 보는 세계: 장애물 KDTree + 집 대각(거리 정규화용)."""
    coord: np.ndarray
    tree: object
    diag: float
    ray_dirs: np.ndarray

    @staticmethod
    def load(data_dir: str | Path, cache: Path = _OBSTACLE_CACHE) -> "GeoWorld":
        from scipy.spatial import cKDTree   # 로컬 PC 에도 scipy 는 있다 (torch 는 없음)

        data_dir = Path(data_dir)
        if cache.is_file():
            coord = np.load(cache)["coord"].astype(np.float64)
        else:
            rooms = sorted(p for p in data_dir.iterdir()
                           if p.is_dir() and (p / "coord.npy").is_file())
            coord = _voxel_down(
                np.vstack([np.load(r / "coord.npy").astype(np.float64) for r in rooms]),
                OBSTACLE_VOXEL)
            np.savez_compressed(cache, coord=coord.astype(np.float32))
            print(f"[rl] 장애물 점군 {len(coord):,}개 캐시: {cache.name}")
        lo, hi = coord.min(0) - BOUNDS_MARGIN, coord.max(0) + BOUNDS_MARGIN
        return GeoWorld(coord=coord, tree=cKDTree(coord),
                        diag=float(np.linalg.norm(hi - lo)),
                        ray_dirs=ray_directions_horiz14())

    def raycast(self, pos: np.ndarray) -> np.ndarray:
        """geo_env._raycast 와 같은 계산: ray_step 간격으로 전진하며 첫 충돌까지."""
        out = np.empty(len(self.ray_dirs), dtype=np.float32)
        for k, d in enumerate(self.ray_dirs):
            hit = RAY_MAX
            t = RAY_STEP
            while t <= RAY_MAX:
                if self.tree.query_ball_point(pos + d * t, CLEARANCE):
                    hit = t
                    break
                t += RAY_STEP
            out[k] = hit / RAY_MAX
        return out

    def observe(self, pos: np.ndarray, carrot: np.ndarray,
                prev_a: np.ndarray) -> np.ndarray:
        rel = np.asarray(carrot, float) - pos
        dist = float(np.linalg.norm(rel))
        return np.concatenate([
            (rel / (dist + 1e-8)).astype(np.float32),
            np.array([min(dist / self.diag, 1.0)], np.float32),
            self.raycast(pos),
            np.asarray(prev_a, np.float32),
        ])


def carrot_on_path(path: np.ndarray, pos: np.ndarray,
                   ahead: float = SUBGOAL_DIST) -> tuple[np.ndarray, float]:
    """경로 위에서 드론보다 `ahead` m 앞선 점, 그리고 **남은 경로 길이**.

    학습 때 carrot 은 측지 필드의 최급강하로 뽑았지만, 우리는 같은 역할을 하는
    최단경로(A*)를 이미 들고 있다 — 그 위를 걸어가는 게 같은 뜻이고 다익스트라
    격자를 한 번 더 깔 이유가 없다. 경로 끝이 더 가까우면 끝점을 준다.

    남은 길이를 같이 주는 이유: 진행 판정에 목표까지의 **직선거리**를 쓰면
    모퉁이를 도는 동안 값이 평평해져 멀쩡히 날고 있는데도 정체로 오판한다.
    """
    seg = path[1:] - path[:-1]
    seg_len = np.linalg.norm(seg, axis=1)
    ok = seg_len > 1e-9
    # 각 구간에 대한 수직투영으로 현재 진행 위치(구간 index + 비율)를 찾는다.
    t = np.zeros(len(seg))
    t[ok] = np.clip(((pos - path[:-1])[ok] * seg[ok]).sum(1) / seg_len[ok] ** 2, 0, 1)
    proj = path[:-1] + seg * t[:, None]
    i = int(np.argmin(np.linalg.norm(proj - pos, axis=1)))

    left = float(np.linalg.norm(path[i + 1] - proj[i]) + seg_len[i + 1:].sum())

    remain = ahead
    p = proj[i]
    for k in range(i, len(seg)):
        step = float(np.linalg.norm(path[k + 1] - p))
        if step >= remain:
            return p + (path[k + 1] - p) * (remain / max(step, 1e-9)), left
        remain -= step
        p = path[k + 1]
    return path[-1], left


def follow_rl(
    bridge,
    path_world: Sequence[Sequence[float]],
    sim_tf,
    world: GeoWorld,
    actor: NumpyActor,
    *,
    max_speed: float = 6.0,          # Unity u/s — follow_path 와 같은 단위
    rc_limit: int = 60,
    goal_radius: float = 0.5,        # m (학습 때 --goal-radius 0.5)
    timeout_sec: Optional[float] = None,
    abort_on_collision: bool = False,
    state_stall_sec: float = 5.0,
    stall_sec: float = 8.0,          # 이 시간 동안 목표에 못 다가가면 실패 (PID 폴백용)
    on_state: Optional[Callable] = None,
    on_status: Callable[[str], None] = print,
    should_abort: Optional[Callable[[], bool]] = None,
) -> FollowResult:
    """A* 경로(월드 m)를 정책으로 따라 난다. 실패하면 호출자가 PID 로 폴백한다.

    `should_abort` 는 follow_path 와 같은 뜻이고 같은 자리에서 확인한다. 여기서
    "aborted" 를 돌려주면 호출자(fly_leg)가 PID 폴백을 타지 않고 바로 끝낸다 —
    중단인데 폴백으로 다시 날면 안 되기 때문이다."""
    path = np.asarray(path_world, dtype=float)
    goal = path[-1]

    # 정책의 한 행동 = 최대 MAX_STEP(0.3 m) 이동. 그래서 '결정 주기'가 곧 속도다:
    # 0.3 m 를 dt 에 걸쳐 가면 0.3/dt m/s. 학습 분포를 지키려고 속도를 rc 로
    # 깎는 대신 주기를 늘린다. max_speed 는 Unity u/s 라 집 scale 5 로 나눈다.
    scale = float(np.linalg.norm(sim_tf.matrix[:, 0]))    # unity u per world m
    speed_world = max(0.1, float(max_speed) / scale)
    dt = MAX_STEP / speed_world

    path_len = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
    if timeout_sec is None:
        timeout_sec = min(600.0, max(60.0, 4.0 * path_len / speed_world))

    start_time = time.time()
    last_state_time = -1.0
    last_fresh_wall = start_time
    best_dist, best_wall = float("inf"), start_time
    baseline_collisions: Optional[int] = None
    collision_count = 0
    rc_sent = 0
    prev_a = np.zeros(3, dtype=np.float32)
    trajectory: list = []
    pos_unity: Optional[np.ndarray] = None

    def stop(success: bool, reason: str) -> FollowResult:
        try:
            bridge.send_rc(0, 0, 0, 0)
        except Exception:
            pass
        err = (float(np.linalg.norm(sim_tf.mosaic_to_unity(goal) - pos_unity))
               if pos_unity is not None else float("inf"))
        return FollowResult(success=success, reason=reason, rc_commands_sent=rc_sent,
                            collision_count=collision_count, final_error_u=err,
                            duration_s=time.time() - start_time,
                            trajectory_unity=trajectory)

    while True:
        now = time.time()
        if should_abort is not None and should_abort():
            on_status("중단 요청 — 비행 정지")
            return stop(False, "aborted")
        if now - start_time > timeout_sec:
            on_status(f"rl flight timeout after {timeout_sec:.0f}s")
            return stop(False, "timeout")

        state = bridge.get_latest_state()
        if state is None or state.time == last_state_time:
            if now - last_fresh_wall > state_stall_sec:
                on_status("simulator state lost")
                return stop(False, "state_lost")
            time.sleep(0.02)
            continue
        last_state_time = state.time
        last_fresh_wall = now

        pos_unity = np.array([state.x, state.y, state.z], dtype=float)
        trajectory.append(tuple(pos_unity))
        pos = sim_tf.unity_to_mosaic(pos_unity)

        if baseline_collisions is None:
            baseline_collisions = state.collision_count
        if state.collision_count - baseline_collisions > collision_count:
            collision_count = state.collision_count - baseline_collisions
            on_status(f"collision #{collision_count}")
            if abort_on_collision:
                return stop(False, "collision_abort")

        if float(np.linalg.norm(goal - pos)) <= goal_radius:
            return stop(True, "arrived")
        carrot, left = carrot_on_path(path, pos)
        if left < best_dist - 0.3:            # 경로 남은 길이로 진행을 잰다
            best_dist, best_wall = left, now
        elif now - best_wall > stall_sec:
            on_status(f"rl stalled {stall_sec:.0f}s, {left:.1f} m left")
            return stop(False, "rl_stalled")

        if on_state is not None:
            try:
                on_state(state)
            except Exception:
                pass

        action = actor(world.observe(pos, carrot, prev_a))
        prev_a = action.astype(np.float32)
        v_world = action * MAX_STEP / dt                      # m/s
        v_unity = np.asarray(v_world, float) @ sim_tf.matrix.T  # 회전+스케일만
        lr, fb, ud, yaw_rc = world_velocity_to_rc(state.yaw, v_unity, rc_limit)
        bridge.send_rc(lr, fb, ud, yaw_rc)
        rc_sent += 1
        time.sleep(dt)


# ------------------------------------------------------------------- self-test

def _check() -> None:
    """관측 재현 검증 — `geo_env` 원본과 같은 값이 나와야 한다 (env `tello`).

    레이 14개와 거리 정규화가 학습 때와 1:1 로 같은지가 전부다. 여기서 어긋나면
    정책은 조용히 못 날게 된다.
    """
    import sys
    sys.path.insert(0, str(_REPO / "playground" / "reinforce_learning"))
    sys.path.insert(0, str(_REPO / "playground" / "demo_house"))
    sys.path.insert(0, str(_REPO / "playground" / "auto_driving" / "draft"))
    from geo_env import DroneGeoEnv

    env = DroneGeoEnv(curriculum=False, priv_obs=True, ray_max=RAY_MAX,
                      ray_layout="horiz14", subgoal_dist=SUBGOAL_DIST,
                      clearance=CLEARANCE, guide_clearance=0.20, bump_penalty=2.0)
    world = GeoWorld.load(_REPO / "data" / "final_npy")

    assert len(world.coord) == len(env.coord), \
        f"장애물 점 개수가 다르다: {len(world.coord)} vs {len(env.coord)}"
    assert abs(world.diag - env.diag) < 1e-6, f"diag {world.diag} vs {env.diag}"

    rng = np.random.default_rng(0)
    idx = rng.choice(len(env.coord), 40, replace=False)
    worst = 0.0
    for p in env.coord[idx] + rng.normal(0, 0.6, (40, 3)):
        env.pos = p
        worst = max(worst, float(np.abs(world.raycast(p) - env._raycast(p)).max()))
    print(f"[rl] 레이 14 최대 오차 {worst:.2e} (40개 위치)")
    assert worst < 1e-6, "레이캐스트가 학습 때와 다르다"

    # carrot: A* 대신 직선 경로를 넣어 경로 위 2.5 m 앞이 맞는지 (기하 검증)
    line = np.array([[0, 0, 0], [10, 0, 0]], float)
    c, left = carrot_on_path(line, np.array([3.0, 0.4, 0.0]))
    assert np.allclose(c, [5.5, 0, 0], atol=1e-6), c
    assert abs(left - 7.0) < 1e-6, left
    assert np.allclose(carrot_on_path(line, np.array([9.0, 0, 0]))[0], [10, 0, 0])
    print("[rl] carrot OK")
    print("[rl] 관측 재현 OK: 정책이 학습 때와 같은 것을 본다")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true")
    ap.parse_args()
    _check()
