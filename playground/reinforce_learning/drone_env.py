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
                 sample_same_floor=False, # True면 '같은 층'에서만(방은 달라도 됨) — 중간 난이도
                 sample_adjacent_rooms=False,  # True면 문 하나로 연결된 두 방에서 (2b)
                 sample_door_cross=False,      # True면 문 앞↔문 뒤 근접 지점만 (2a, 문 통과 집중훈련)
                 door_range=1.5,       # door_cross: 문 중심에서 이 반경 안에서 시작/도착을 뽑음
                 door_route_reward=True,  # 문 경유 에피소드(2a/2b)의 진척을 '문 경유 거리'로 계산.
                                          # 직선거리 진척은 벽 쪽으로 유인하는 문제가 있어, 문을
                                          # 지나기 전엔 (현위치→문)+(문→목표) 가 줄어야 +보상.
                 door_pass_radius=0.5, # 문 중심 이 반경 안에 들어오면 '문 통과' → 직선 진척으로 전환
                 easy_mix=0.0,         # 이 확률로 easy_mix_mode 난이도 에피소드를 섞음(망각 방지)
                 easy_mix_mode=None,   # 'same_room' | 'door_cross' | 'adjacent_rooms'
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
        self.rooms, floors = [], []
        for r in g['rooms'].values():
            self.rooms.append((np.array(r['bbox_min'], float), np.array(r['bbox_max'], float)))
            floors.append(int(r.get('floor', 0)))
        # 층 → 그 층에 속한 방 인덱스 목록 (같은 층 샘플링용)
        self.floor_rooms = {}
        for i, f in enumerate(floors):
            self.floor_rooms.setdefault(f, []).append(i)
        self.floor_keys = list(self.floor_rooms.keys())

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
        self.sample_same_floor = bool(sample_same_floor)
        self.sample_adjacent_rooms = bool(sample_adjacent_rooms)
        self.sample_door_cross = bool(sample_door_cross)
        self.door_range        = float(door_range)
        self.door_route_reward = bool(door_route_reward)
        self.door_pass_radius  = float(door_pass_radius)
        self.easy_mix          = float(easy_mix)
        self.easy_mix_mode     = easy_mix_mode
        self.max_goal_dist     = None if max_goal_dist is None else float(max_goal_dist)

        # 같은 층 문 엣지 (방A인덱스, 방B인덱스, 문 중심좌표) — 2a/2b 샘플링용.
        # 계단(층간) 엣지는 제외: 문 커리큘럼은 층 안에서만.
        # 수동 표시된 door_center 는 문틀/벽에 박혀 있는 경우가 있어(실측 5/20개가
        # 여유반경 미달 = 통과 불가) 개구부 중앙(주변 최대 여유 점)으로 스냅해서 쓴다.
        id2i = {rid: i for i, rid in enumerate(g['rooms'].keys())}
        self.door_edges, dropped = [], 0
        for e in g.get('edges', []):
            if e.get('door_center') is None:
                continue
            a, b = id2i.get(e['a']), id2i.get(e['b'])
            if a is None or b is None or floors[a] != floors[b]:
                continue
            raw = np.array(e['door_center'], float)
            dc = self._snap_door_center(raw)
            if dc is None:
                dc = self._snap_door_center(raw, half=0.5)
            if dc is None:                     # 0.5m 안에 통과 가능한 점 없음 → 제외
                dropped += 1
                continue
            self.door_edges.append((a, b, dc))
        if dropped:
            print(f"[환경] 통과 불가 문 {dropped}개 제외 (door_center 주변 0.5m에 빈 공간 없음)")

        needs_doors = (self.sample_door_cross or self.sample_adjacent_rooms
                       or self.easy_mix_mode in ('door_cross', 'adjacent_rooms'))
        if needs_doors and not self.door_edges:
            raise RuntimeError("rooms_graph.json 에 같은 층 문 엣지(door_center)가 없어 "
                               "door_cross/adjacent_rooms 샘플링을 쓸 수 없음")

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

    # ── 보조: 문 중심을 '주변에서 장애물과 가장 먼 점'(≈개구부 중앙)으로 보정 ──
    # 수동 표시 좌표가 문틀에 붙어 있으면 그 점을 드론이 점유할 수 없어 통과 불가가 된다.
    # ±half 박스를 격자 탐색해 최대 여유 점을 고른다(결정적). 여유 미달이면 None.
    def _snap_door_center(self, dc, half=0.3, step=0.06):
        ax = np.arange(-half, half + 1e-9, step)
        gx, gy, gz = np.meshgrid(ax, ax, ax, indexing='ij')
        cand = dc + np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
        d, _ = self.tree.query(cand)
        k = int(np.argmax(d))
        if d[k] < self.clearance + 0.03:
            return None
        return cand[k]

    # ── 보조: 문 중심 door_range 반경 안(방 bbox와 교집합)에서 빈 점 뽑기 ──
    def _sample_near_door(self, room_idx, center, max_try=60):
        lo, hi = self.rooms[room_idx]
        lo = np.maximum(lo + 0.3, center - self.door_range)
        hi = np.minimum(hi - 0.3, center + self.door_range)
        hi = np.maximum(hi, lo + 1e-3)
        for _ in range(max_try):
            p = self.np_random.uniform(lo, hi)
            if R.is_point_free(p, self.tree, self.clearance):
                return p.astype(np.float64)
        return None

    # ── 보조: 이번 에피소드 샘플링 모드 결정 (easy_mix 확률로 쉬운 모드 섞기) ──
    def _episode_mode(self):
        if self.sample_same_room:        mode = 'same_room'
        elif self.sample_door_cross:     mode = 'door_cross'
        elif self.sample_adjacent_rooms: mode = 'adjacent_rooms'
        elif self.sample_same_floor:     mode = 'same_floor'
        else:                            mode = 'all'
        if self.easy_mix_mode and self.np_random.random() < self.easy_mix:
            mode = self.easy_mix_mode
        return mode

    # ── 보조: 시작·도착 쌍 뽑기 (난이도 옵션 반영) ──────────────
    # 반환: (시작, 도착, 문중심 or None) — 문 경유 에피소드(2a/2b)면 그 문의 중심좌표
    def _sample_pair(self, tries=100):
        last = (None, None, None)
        for _ in range(tries):
            mode = self._episode_mode()
            door = None
            if mode == 'door_cross':                   # 문 앞 ↔ 문 뒤 (2a)
                ai, bi, dc = self.door_edges[int(self.np_random.integers(len(self.door_edges)))]
                if self.np_random.random() < 0.5:
                    ai, bi = bi, ai
                s = self._sample_near_door(ai, dc)
                g = self._sample_near_door(bi, dc)
                door = dc
            else:
                if mode == 'same_room':                # 같은 방 (제일 쉬움)
                    ra = rb = int(self.np_random.integers(len(self.rooms)))
                elif mode == 'adjacent_rooms':         # 문 하나로 연결된 두 방 (2b)
                    ra, rb, dc = self.door_edges[int(self.np_random.integers(len(self.door_edges)))]
                    if self.np_random.random() < 0.5:
                        ra, rb = rb, ra
                    door = dc
                elif mode == 'same_floor':             # 같은 층, 방은 달라도 됨 (중간)
                    f = self.floor_keys[int(self.np_random.integers(len(self.floor_keys)))]
                    idxs = self.floor_rooms[f]
                    ra = idxs[int(self.np_random.integers(len(idxs)))]
                    rb = idxs[int(self.np_random.integers(len(idxs)))]
                else:                                  # 집 전체 (제일 어려움, 층 넘기 포함)
                    ra = rb = None
                s = self._sample_free(ra)
                g = self._sample_free(rb)
            last = (s, g, door)
            if s is None or g is None:
                continue
            d = float(np.linalg.norm(g - s))
            if d < self.min_separation:
                continue
            if self.max_goal_dist is not None and d > self.max_goal_dist:
                continue
            return s, g, door
        return last

    # ── 보조: 진척 계산용 거리 (문 경유 보상) ────────────────────
    # 문 경유 에피소드에서 문을 지나기 전엔 (현위치→문)+(문→목표), 지난 후엔 직선거리.
    # 직선거리 진척은 목표가 옆방일 때 벽 쪽으로 유인하므로, 보상 기울기가 문을 가리키게 한다.
    def _route_dist(self, p):
        if self.door_wp is not None and not self.door_passed:
            return float(np.linalg.norm(self.door_wp - p) +
                         np.linalg.norm(self.goal - self.door_wp))
        return float(np.linalg.norm(self.goal - p))

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

        door = None
        if opt_s is not None or opt_g is not None:
            # 평가: 원하는 두 점 직접 지정(막힌 점이면 근처 빈 곳으로 보정)
            self.pos  = self._resolve_point(opt_s) if opt_s is not None else self._sample_free()
            self.goal = self._resolve_point(opt_g) if opt_g is not None else self._sample_free()
        else:
            # 학습: 난이도 옵션(같은 방/거리상한)을 반영한 랜덤 시작·도착 쌍
            self.pos, self.goal, door = self._sample_pair()

        if self.pos is None or self.goal is None:        # 극히 드문 실패 → 폴백
            self.pos = self.coord[0] + np.array([0.5, 0, 0.5])
            self.goal = self.pos + np.array([max(self.min_separation, 1.0), 0, 0])
            door = None

        # 문 경유 보상 상태 (문 경유 에피소드가 아니면 둘 다 비활성).
        # 시작부터 목표가 직선으로 보이면 문 경유가 불필요하므로 바로 직선 진척 모드.
        self.door_wp = door if (self.door_route_reward and door is not None) else None
        self.door_passed = bool(
            self.door_wp is not None and
            R.is_edge_free(self.pos, self.goal, self.tree, radius=self.clearance))

        self.steps = 0
        self.path = [self.pos.copy()]
        return self._obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, np.float64), -1.0, 1.0)
        new = self.pos + action * self.max_step
        self.steps += 1

        prev_route = self._route_dist(self.pos)
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
            # 문 통과 판정: 문 중심 근접 or 목표가 직선으로 보임(중심을 비껴 통과한 경우).
            # 후자가 없으면 문을 지나고도 보상이 문 쪽으로 되돌아오라고 유인하게 된다.
            if (self.door_wp is not None and not self.door_passed and
                    (float(np.linalg.norm(self.door_wp - self.pos)) <= self.door_pass_radius
                     or R.is_edge_free(self.pos, self.goal, self.tree, radius=self.clearance))):
                self.door_passed = True    # 이후 진척은 목표 직선거리 기준
            new_dist = float(np.linalg.norm(self.goal - self.pos))
            reward = (prev_route - self._route_dist(self.pos)) * self.progress_scale \
                     - self.step_penalty
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
