# -*- coding: utf-8 -*-
"""
rl_planner.py — 학습된 SAC 정책을 "경로 계획기"로 쓰는 공용 모듈
────────────────────────────────────────────────────────────────────
A*/RRT* 대신 정책을 결정론으로 굴려서 나온 궤적을 경로로 쓴다. 세 곳이
이걸 같이 쓴다:

  web_server.py                          웹 관제 콘솔의 POST /plan
  simulator/bridge/fly_rl_path.py        시뮬레이터에서 실제 비행
  3D-segmentation/webapp_llm_v2/server.py  LLM 파이프라인의 --algo rl

인터페이스는 그쪽 `planner.plan_path()` 와 맞춰 뒀다 — 둘 다 **월드미터**
좌표의 waypoint 순서열을 돌려주므로 갈아끼우기만 하면 된다.

    import rl_planner
    rl_planner.load()                       # 모델+환경 1회 로드 (약 6초)
    wps, info = rl_planner.plan(start, goal)
    if not info['success']:                 # 정책 미도달 -> A* 폴백
        ...

좌표계: demo_house / LitePT 와 같은 월드(Z-up, m) 프레임. 원본 점군(npy)
없이도 동작한다 — 장애물·측지격자·계단체인 캐시가 저장소에 커밋돼 있다.
"""
from __future__ import annotations

import os
import sys
import time
import threading

import numpy as np

_ROOT = os.path.dirname(os.path.abspath(__file__))
_RL = os.path.join(_ROOT, 'playground', 'reinforce_learning')
if _RL not in sys.path:
    sys.path.insert(0, _RL)

# 학습 당시 환경 설정 — 바꾸면 정책 성능이 무너진다.
# ★ clearance 는 반드시 명시할 것: 기본값(0.235)으로 env 를 만들면 0.12 용
#   geo_graph_cache 가 덮여서 다음 실행이 캐시를 1분간 재생성한다.
ENV_KW = dict(curriculum=False, ray_max=4.0, ray_layout='horiz14',
              subgoal_dist=2.5, bump_penalty=2.0, clearance=0.12)
MAX_STEPS = 700
MODEL = os.path.join(_RL, 'model_geo_best')

# 시뮬레이터 스폰 지점(TelloSimulator.spawnPosition (-22.51, 5.20, 5.22))을
# 월드 좌표로 되돌린 값. 시뮬레이터가 없을 때의 기본 출발점.
HOME_WORLD = (4.50, -1.04, -2.06)

_lock = threading.Lock()      # env 는 재진입 불가 -> 요청 직렬화
_model = None
_envs = {}                    # (people, shield) -> DroneGeoEnv
_priv = False
_engine = None


# ── 로드 ───────────────────────────────────────────────────────────
def load(model_path: str | None = None) -> str:
    """모델 + 기본 환경을 1회 로드하고 엔진 이름을 반환한다."""
    global _model, _priv, _engine
    if _model is not None:
        return _engine

    import gymnasium
    import asym_policy                       # noqa: F401  SAC.load 가 정책 클래스를 찾게
    from stable_baselines3 import SAC

    path = model_path or MODEL
    if path.endswith('.zip'):
        path = path[:-4]
    if not os.path.exists(path + '.zip'):
        raise FileNotFoundError(f'모델이 없습니다: {path}.zip')

    t0 = time.time()
    _model = SAC.load(path, device='auto')
    _priv = isinstance(_model.observation_space, gymnasium.spaces.Dict)
    _get_env(0, False)                       # 기본 환경 미리 생성(캐시 워밍)
    _engine = f'SAC · {os.path.basename(path)}'
    print(f'[rl_planner] 준비 완료 ({time.time() - t0:.1f}s) - {_engine}')
    return _engine


def ready() -> bool:
    return _model is not None


def engine() -> str:
    return _engine or 'not loaded'


def home() -> np.ndarray:
    return np.array(HOME_WORLD, dtype=float)


def _get_env(people: int, shield: bool):
    """(사람 수, 안전막) 조합별 환경을 지연 생성해 재사용한다."""
    key = (int(people), bool(shield))
    env = _envs.get(key)
    if env is None:
        from geo_env import DroneGeoEnv
        env = DroneGeoEnv(priv_obs=_priv, max_steps=MAX_STEPS,
                          people=key[0], shield=key[1], **ENV_KW)
        _envs[key] = env
    return env


# ── 경로 단축 ──────────────────────────────────────────────────────
def shortcut(points, env, radius: float | None = None):
    """정책 궤적(0.3m 간격)을 시야(line-of-sight) 단축으로 성기게 만든다.

    PID 추종기(follow_path)는 waypoint 를 하나씩 찍고 가므로, 0.3m 간격
    그대로 주면 stop-and-go 가 된다. 갈 수 있는 가장 먼 점으로 건너뛴다.
    판정 반경은 정책이 실제로 쓰는 여유반경과 같은 값(기본 0.12)이라
    단축된 경로도 정책 물리 기준으로 안전하다.
    """
    import rrt_star_drone as R
    pts = [np.asarray(p, dtype=float) for p in points]
    if len(pts) < 3:
        return pts
    r = float(radius if radius is not None else env._clr(pts[0]))
    out = [pts[0]]
    i = 0
    while i < len(pts) - 1:
        j = len(pts) - 1
        while j > i + 1 and not R.is_edge_free(pts[i], pts[j], env.tree, radius=r):
            j -= 1
        out.append(pts[j])
        i = j
    return out


# ── 경로 계획 ──────────────────────────────────────────────────────
def plan(start_world, goal_world, *, people: int = 0, shield: bool = False,
         smooth: bool = True):
    """`planner.plan_path()` 자리에 그대로 들어가는 RL 경로 계획.

    반환 `(waypoints, info)`:
      waypoints  월드미터 np.ndarray 리스트 (smooth=True 면 시야 단축본)
      info       algo/success/steps/bumps/dist/length_m/ms + 다음 두 가지
                 raw    정책이 실제로 지나간 조밀 궤적(0.3m 간격) — 그림용
                 start, goal  환경이 스냅한 실제 출발/도착 좌표

    ★ 정책은 성공률 0.98 수준이라 실패할 수 있다. `info['success']` 가
      False 면 궤적은 "가다 만" 것이니 호출부에서 A* 로 폴백할 것.
    """
    if _model is None:
        raise RuntimeError('rl_planner.load() 를 먼저 호출해야 합니다')

    start_world = np.asarray(start_world, dtype=float).reshape(3)
    goal_world = np.asarray(goal_world, dtype=float).reshape(3)

    t0 = time.time()
    with _lock:
        env = _get_env(people, shield)
        # 벽 위나 집 밖을 줘도 환경이 가장 가까운 도달 가능 셀로 스냅한다.
        obs, _ = env.reset(options={'start': start_world, 'goal': goal_world})
        snapped_start, snapped_goal = env.pos.copy(), env.goal.copy()
        term = trunc = False
        step_info = {}
        while not (term or trunc):
            action, _ = _model.predict(obs, deterministic=True)
            obs, _, term, trunc, step_info = env.step(action)
        raw = [np.asarray(p, dtype=float) for p in env.path]
        steps = int(env.steps)
        wps = shortcut(raw, env) if smooth and len(raw) > 2 else list(raw)
    ms = (time.time() - t0) * 1000.0

    def _len(seq):
        return float(sum(np.linalg.norm(b - a) for a, b in zip(seq[:-1], seq[1:]))) \
            if len(seq) > 1 else 0.0

    info = {
        'algo': 'rl',
        'engine': _engine,
        'success': bool(step_info.get('is_success', False)),
        'steps': steps,
        'bumps': int(step_info.get('bumps', 0)),
        'person_hits': int(step_info.get('person_hits', 0)),
        'dist': float(step_info.get('dist', float('nan'))),   # 목표까지 남은 거리
        'length_m': _len(wps),
        'raw_length_m': _len(raw),
        'n_waypoints': len(wps),
        'ms': ms,
        'raw': raw,
        'start': snapped_start,
        'goal': snapped_goal,
    }
    return wps, info


if __name__ == '__main__':                    # 간이 자가 점검
    load()
    for s, g in [(HOME_WORLD, (6.0, 1.0, 4.2)), ((6.0, 1.0, 4.2), (2.0, 0.0, -1.0))]:
        wps, info = plan(s, g)
        print(f"  {np.round(s, 2)} -> {np.round(g, 2)} : "
              f"{'성공' if info['success'] else '미도달'} | {info['steps']}스텝 | "
              f"조밀 {len(info['raw'])}점 {info['raw_length_m']:.2f}m -> "
              f"단축 {info['n_waypoints']}점 {info['length_m']:.2f}m | "
              f"{info['ms']:.0f}ms")
