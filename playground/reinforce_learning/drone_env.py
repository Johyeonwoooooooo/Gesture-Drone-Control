"""
drone_env.py
────────────
demo_house 점군 위에서 드론이 시작점→도착점으로 날아가는 것을 배우는
강화학습 '환경'(Gymnasium Environment).

핵심 개념(입문용 요약):
  - 강화학습은 '데이터셋'이 아니라 '환경'을 만든다. 에이전트(드론)가 이 환경 안에서
    수백만 번 직접 움직여 보면서 경험(데이터)을 스스로 만들어내고, 그걸로 학습한다.
  - 이 파일이 바로 그 '환경'. 다른 코드(rrt_star_drone / hierarchical_plan)에서 쓰던
    전역 장애물 점군 + KDTree 충돌검사를 그대로 재활용한다.

환경의 3요소:
  - 관측(state)  : 목표 방향 단위벡터(3) + 목표까지 거리(1) + 주변 레이캐스트 거리(N)
  - 행동(action) : 속도 벡터 (vx, vy, vz), 각 -1~1  (한 스텝에 최대 max_step 만큼 이동)
  - 보상(reward) : 목표에 가까워지면 +, 매 스텝 작은 -, 벽에 박으면 큰 -, 도착하면 큰 +

학습할 때 시작/도착점은 매 에피소드마다 '랜덤'으로 자동 생성된다(그래야 아무 두 점이나
잘 가는 정책이 됨). 특정 두 점으로 평가하려면 reset(options={'start':..,'goal':..}) 로 넘긴다.
"""

import os
import sys
import numpy as np

import gymnasium as gym
from gymnasium import spaces

# ── 기존 코드 재사용 ───────────────────────────────────────────
_BASE  = os.path.dirname(os.path.abspath(__file__))
_DEMO  = os.path.join(_BASE, '..', 'demo_house')
_DRAFT = os.path.join(_BASE, '..', 'auto_driving', 'draft')
sys.path.insert(0, _DEMO)
sys.path.insert(0, _DRAFT)

import rrt_star_drone as R                                   # noqa: E402  충돌검사/보조함수
from hierarchical_plan import load_graph, build_global_obstacles  # noqa: E402

from scipy.spatial import cKDTree                            # noqa: E402

_CACHE = os.path.join(_BASE, 'obstacles_cache.npz')


# ── 장애물(전역 점군 + KDTree) 준비 — 한 번만 만들고 캐시 ─────────
def load_obstacles(voxel=0.06, rebuild=False):
    """모든 방 npy 를 합친 전역 장애물 점군을 만들어 KDTree 와 함께 반환.
       처음 한 번은 22개 방 npy 를 읽느라 몇 초 걸리므로 .npz 로 캐시한다."""
    if (not rebuild) and os.path.exists(_CACHE):
        d = np.load(_CACHE)
        if float(d['voxel']) == voxel:
            return d['coord'].astype(np.float64)
    print(f"[환경] 전역 장애물 점군 구성 중... (voxel {voxel}m, 최초 1회만)")
    g = load_graph()
    coord = build_global_obstacles(g['rooms'], voxel)
    np.savez_compressed(_CACHE, coord=coord.astype(np.float32), voxel=voxel)
    print(f"[환경] 장애물 점 {len(coord):,}개 캐시 저장: {_CACHE}")
    return coord.astype(np.float64)


# 14방향 레이캐스트(거리센서) 방향 = 6축 + 8대각
def _ray_directions():
    dirs = [( 1,0,0),(-1,0,0),(0, 1,0),(0,-1,0),(0,0, 1),(0,0,-1)]
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                dirs.append((sx, sy, sz))
    v = np.array(dirs, dtype=np.float64)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


class DroneHouseEnv(gym.Env):
    """demo_house 안에서 시작점→도착점 자율비행을 배우는 환경."""
    metadata = {"render_modes": []}

    def __init__(self,
                 voxel=0.06,
                 max_step=0.30,        # 한 스텝 최대 이동거리(m)  ← RRT* STEP_SIZE 와 유사
                 clearance=R.OBSTACLE_RADIUS,  # 드론 여유반경(충돌검사용)
                 ray_max=2.0,          # 거리센서 최대 사거리(m)
                 ray_step=0.15,        # 레이캐스트 샘플 간격(m)
                 goal_radius=0.30,     # 이 거리 안에 들어오면 '도착'
                 max_steps=300,        # 한 에피소드 최대 스텝(시간초과 기준)
                 min_separation=1.5,   # 시작·도착 최소 거리(m)
                 progress_scale=5.0,   # 목표에 가까워진 거리 → 보상 환산 배율
                 step_penalty=0.02,    # 매 스텝 시간 페널티
                 collision_penalty=10.0,
                 goal_bonus=100.0,
                 timeout_penalty=0.0,  # 시간초과 시 페널티(배회 방지). 0이면 끔
                 sample_same_room=False,  # True면 시작·도착을 '같은 방'에서만 (쉬움)
                 max_goal_dist=None,   # 설정 시 시작-도착 거리 상한(m) — 가까운 목표만
                 obstacles=None):
        super().__init__()
        self.coord = load_obstacles(voxel) if obstacles is None else obstacles
        self.tree  = cKDTree(self.coord)

        m = 0.3
        self.bounds_lo = self.coord.min(0) - m
        self.bounds_hi = self.coord.max(0) + m
        self.diag = float(np.linalg.norm(self.bounds_hi - self.bounds_lo))

        # 시작/도착 샘플링용 방 bbox 목록 (랜덤점이 항상 '집 안'에 찍히도록)
        g = load_graph()
        self.rooms = [(np.array(r['bbox_min'], float), np.array(r['bbox_max'], float))
                      for r in g['rooms'].values()]

        self.max_step          = float(max_step)
        self.clearance         = float(clearance)
        self.ray_max           = float(ray_max)
        self.ray_step          = float(ray_step)
        self.goal_radius       = float(goal_radius)
        self.max_steps         = int(max_steps)
        self.min_separation    = float(min_separation)
        self.progress_scale    = float(progress_scale)
        self.step_penalty      = float(step_penalty)
        self.collision_penalty = float(collision_penalty)
        self.goal_bonus        = float(goal_bonus)
        self.timeout_penalty   = float(timeout_penalty)
        self.sample_same_room  = bool(sample_same_room)
        self.max_goal_dist     = None if max_goal_dist is None else float(max_goal_dist)

        self.ray_dirs = _ray_directions()
        n_obs = 3 + 1 + len(self.ray_dirs)            # 목표단위(3)+거리(1)+레이(14)
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(n_obs,), dtype=np.float32)
        self.action_space      = spaces.Box(-1.0, 1.0, shape=(3,),     dtype=np.float32)

        self.pos = None
        self.goal = None
        self.steps = 0
        self.path = []                                # 평가/시각화용 비행 궤적

    # ── 보조: 충돌 없는 랜덤 점(집 안) 하나 뽑기 ──────────────
    def _sample_free(self, room_idx=None, max_try=200):
        for _ in range(max_try):
            ri = room_idx if room_idx is not None else self.np_random.integers(len(self.rooms))
            lo, hi = self.rooms[ri]
            lo = lo + 0.3; hi = hi - 0.3
            hi = np.maximum(hi, lo + 1e-3)
            p = self.np_random.uniform(lo, hi)
            if R.is_point_free(p, self.tree, self.clearance):
                return p.astype(np.float64)
        return None

    # ── 보조: 시작·도착 쌍 뽑기 (난이도 옵션 반영) ──────────────
    def _sample_pair(self, tries=100):
        last = (None, None)
        for _ in range(tries):
            ri = self.np_random.integers(len(self.rooms)) if self.sample_same_room else None
            s = self._sample_free(ri)
            g = self._sample_free(ri)
            last = (s, g)
            if s is None or g is None:
                continue
            d = float(np.linalg.norm(g - s))
            if d < self.min_separation:
                continue
            if self.max_goal_dist is not None and d > self.max_goal_dist:
                continue
            return s, g
        return last

    # ── 관측 만들기 ──────────────────────────────────────────
    def _raycast(self, pos):
        out = np.empty(len(self.ray_dirs), dtype=np.float32)
        for k, d in enumerate(self.ray_dirs):
            hit = self.ray_max
            t = self.ray_step
            while t <= self.ray_max:
                if not R.is_point_free(pos + d * t, self.tree, self.clearance):
                    hit = t
                    break
                t += self.ray_step
            out[k] = hit / self.ray_max          # 0(코앞이 벽)~1(2m 안에 벽 없음)
        return out

    def _obs(self):
        rel = self.goal - self.pos
        dist = float(np.linalg.norm(rel))
        unit = rel / (dist + 1e-8)
        dnorm = min(dist / self.diag, 1.0)
        return np.concatenate([unit.astype(np.float32),
                               np.array([dnorm], np.float32),
                               self._raycast(self.pos)])

    def _resolve_point(self, pt):
        """평가용: 지정 좌표를 충돌 없는 가까운 점으로 보정(없으면 None)."""
        p = R.find_nearest_free(np.array(pt, float), self.tree,
                                [[self.bounds_lo[k], self.bounds_hi[k]] for k in range(3)],
                                radius=self.clearance)
        return p

    # ── Gymnasium API ────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        opt_s = options.get('start')
        opt_g = options.get('goal')

        if opt_s is not None or opt_g is not None:
            # 평가: 원하는 두 점 직접 지정(막힌 점이면 근처 빈 곳으로 보정)
            self.pos  = self._resolve_point(opt_s) if opt_s is not None else self._sample_free()
            self.goal = self._resolve_point(opt_g) if opt_g is not None else self._sample_free()
        else:
            # 학습: 난이도 옵션(같은 방/거리상한)을 반영한 랜덤 시작·도착 쌍
            self.pos, self.goal = self._sample_pair()

        if self.pos is None or self.goal is None:        # 극히 드문 실패 → 폴백
            self.pos = self.coord[0] + np.array([0.5, 0, 0.5])
            self.goal = self.pos + np.array([max(self.min_separation, 1.0), 0, 0])

        self.steps = 0
        self.path = [self.pos.copy()]
        return self._obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, np.float64), -1.0, 1.0)
        new = self.pos + action * self.max_step
        self.steps += 1

        prev_dist = float(np.linalg.norm(self.goal - self.pos))
        out_of_bounds = bool(np.any(new < self.bounds_lo) or np.any(new > self.bounds_hi))
        hit = (not out_of_bounds) and (not R.is_edge_free(self.pos, new, self.tree,
                                                          radius=self.clearance))

        terminated = False
        if out_of_bounds or hit:
            reward = -self.collision_penalty               # 벽 충돌/맵 이탈 → 큰 페널티, 종료
            terminated = True
        else:
            self.pos = new
            self.path.append(self.pos.copy())
            new_dist = float(np.linalg.norm(self.goal - self.pos))
            reward = (prev_dist - new_dist) * self.progress_scale - self.step_penalty
            if new_dist <= self.goal_radius:
                reward += self.goal_bonus                  # 도착 성공 → 큰 보너스, 종료
                terminated = True

        success = bool(terminated and reward > 0)          # 도착으로 종료된 경우만
        truncated = (self.steps >= self.max_steps)         # 시간초과
        if truncated and not terminated:
            reward -= self.timeout_penalty                 # 배회 방지(옵션, 기본 0)
        info = {"dist": float(np.linalg.norm(self.goal - self.pos)),
                "is_success": success}
        return self._obs(), float(reward), terminated, truncated, info
