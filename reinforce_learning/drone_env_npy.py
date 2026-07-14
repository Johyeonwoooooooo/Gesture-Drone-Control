"""
drone_env_npy.py
────────────────
drone_env.py 와 동일한 환경이지만 demo_house 의존성을 제거하고
/shareHost/minyoy/project/data/npy 의 centers.pkl + coord.npy 로 직접 구동한다.

rrt_star_drone / hierarchical_plan 을 사용하지 않고 여기서 직접 구현:
  - is_point_free / is_edge_free / find_nearest_free
  - load_obstacles_npy()   : 전체 방 coord.npy 합치기
  - load_graph_npy()       : centers.pkl 에서 방 bbox 추출
"""

import os
import glob
import pickle

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scipy.spatial import cKDTree

# ── 경로 설정 ─────────────────────────────────────────────────
NPY_ROOT = '/shareHost/minyoy/project/data/npy'
_BASE    = os.path.dirname(os.path.abspath(__file__))
_CACHE   = os.path.join(_BASE, 'obstacles_npy_cache.npz')

OBSTACLE_RADIUS = 0.25   # 드론 여유반경(m) — rrt_star_drone.OBSTACLE_RADIUS 대체


# ── 충돌 검사 함수 (rrt_star_drone 대체) ──────────────────────
def is_point_free(p, tree: cKDTree, clearance: float) -> bool:
    d, _ = tree.query(p, workers=1)
    return float(d) >= clearance


def is_edge_free(p1, p2, tree: cKDTree, radius: float, steps: int = 10) -> bool:
    for i in range(steps + 1):
        t = i / steps
        if not is_point_free(p1 * (1 - t) + p2 * t, tree, radius):
            return False
    return True


def find_nearest_free(p, tree: cKDTree, bounds, radius: float, search_r: float = 1.0, n: int = 200):
    """p 주변 search_r 반경에서 충돌 없는 가장 가까운 점. 없으면 None."""
    rng = np.random.default_rng(42)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    cands = rng.uniform(-search_r, search_r, (n, 3)) + p
    cands = np.clip(cands, lo, hi)
    dists = np.linalg.norm(cands - p, axis=1)
    for idx in np.argsort(dists):
        if is_point_free(cands[idx], tree, radius):
            return cands[idx].astype(np.float64)
    return None


# ── 장애물 점군 로드 ───────────────────────────────────────────
def load_obstacles_npy(voxel: float = 0.06, rebuild: bool = False):
    if (not rebuild) and os.path.exists(_CACHE):
        d = np.load(_CACHE)
        if float(d['voxel']) == voxel:
            return d['coord'].astype(np.float64)

    print(f"[환경] 전역 장애물 점군 구성 중... (voxel {voxel}m, 최초 1회만)")
    pkls = sorted(glob.glob(os.path.join(NPY_ROOT, '*/centers.pkl')))
    all_coords = []
    for pkl_path in pkls:
        d = pickle.load(open(pkl_path, 'rb'))
        all_coords.append(d['coord'].astype(np.float32))

    coord = np.vstack(all_coords)

    # voxel 다운샘플
    voxel_idx = np.floor(coord / voxel).astype(np.int64)
    _, unique = np.unique(voxel_idx, axis=0, return_index=True)
    coord = coord[unique]

    np.savez_compressed(_CACHE, coord=coord.astype(np.float32), voxel=voxel)
    print(f"[환경] 장애물 점 {len(coord):,}개 캐시 저장: {_CACHE}")
    return coord.astype(np.float64)


# ── 방 그래프 로드 (centers.pkl → bbox) ───────────────────────
def load_graph_npy():
    """centers.pkl 에서 방 bbox 를 추출해 drone_env 가 기대하는 형식으로 반환.
    엣지(문 연결) 정보는 detections.json 에 없으므로 빈 리스트로 반환한다.
    → door_cross / adjacent_rooms 모드는 사용 불가.
    """
    pkls = sorted(glob.glob(os.path.join(NPY_ROOT, '*/centers.pkl')))
    rooms = {}
    for pkl_path in pkls:
        d = pickle.load(open(pkl_path, 'rb'))
        room_id = d['room']
        coord   = d['coord']
        lo = coord.min(0).tolist()
        hi = coord.max(0).tolist()
        # floor: 폴더명 끝 숫자 앞 세 자리(000/001/002)로 층 추정
        folder = os.path.basename(os.path.dirname(pkl_path))
        floor  = int(folder.split('_')[-2]) if '_' in folder else 0
        rooms[room_id] = {
            'bbox_min': lo,
            'bbox_max': hi,
            'floor':    floor,
        }
    return {'rooms': rooms, 'edges': []}


# ── 14방향 레이캐스트 방향 ────────────────────────────────────
def _ray_directions():
    dirs = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                dirs.append((sx, sy, sz))
    v = np.array(dirs, dtype=np.float64)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


# ── 환경 ──────────────────────────────────────────────────────
class DroneHouseEnv(gym.Env):
    """demo_house 의존성 없이 npy 점군으로 구동하는 드론 환경.
    drone_env.DroneHouseEnv 와 동일한 API."""
    metadata = {"render_modes": []}

    def __init__(self,
                 voxel=0.06,
                 max_step=0.30,
                 clearance=OBSTACLE_RADIUS,
                 ray_max=2.0,
                 ray_step=0.15,
                 goal_radius=0.30,
                 max_steps=300,
                 min_separation=1.5,
                 progress_scale=5.0,
                 step_penalty=0.02,
                 collision_penalty=10.0,
                 goal_bonus=100.0,
                 timeout_penalty=0.0,
                 sample_same_room=False,
                 sample_same_floor=False,
                 sample_adjacent_rooms=False,
                 sample_door_cross=False,
                 door_range=1.5,
                 door_route_reward=True,
                 door_pass_radius=0.5,
                 easy_mix=0.0,
                 easy_mix_mode=None,
                 max_goal_dist=None,
                 obstacles=None):
        super().__init__()

        self.coord = load_obstacles_npy(voxel) if obstacles is None else obstacles
        self.tree  = cKDTree(self.coord)

        m = 0.3
        self.bounds_lo = self.coord.min(0) - m
        self.bounds_hi = self.coord.max(0) + m
        self.diag = float(np.linalg.norm(self.bounds_hi - self.bounds_lo))

        g = load_graph_npy()
        self.rooms, floors = [], []
        for r in g['rooms'].values():
            self.rooms.append((np.array(r['bbox_min'], float),
                               np.array(r['bbox_max'], float)))
            floors.append(int(r.get('floor', 0)))

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

        # 문 엣지 없음 (detections.json 에 문 연결 정보 없음)
        self.door_edges = []
        needs_doors = (self.sample_door_cross or self.sample_adjacent_rooms
                       or self.easy_mix_mode in ('door_cross', 'adjacent_rooms'))
        if needs_doors:
            raise RuntimeError("door_cross / adjacent_rooms 모드는 문 엣지 정보가 필요합니다. "
                               "npy 기반 환경에서는 지원하지 않습니다.")

        self.ray_dirs = _ray_directions()
        n_obs = 3 + 1 + len(self.ray_dirs)
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(n_obs,), dtype=np.float32)
        self.action_space      = spaces.Box(-1.0, 1.0, shape=(3,),     dtype=np.float32)

        self.pos = None
        self.goal = None
        self.steps = 0
        self.path = []
        self.door_wp = None
        self.door_passed = True

    def _sample_free(self, room_idx=None, max_try=200):
        for _ in range(max_try):
            ri = room_idx if room_idx is not None else self.np_random.integers(len(self.rooms))
            lo, hi = self.rooms[ri]
            lo = lo + 0.3; hi = hi - 0.3
            hi = np.maximum(hi, lo + 1e-3)
            p = self.np_random.uniform(lo, hi)
            if is_point_free(p, self.tree, self.clearance):
                return p.astype(np.float64)
        return None

    def _episode_mode(self):
        if self.sample_same_room:    return 'same_room'
        elif self.sample_same_floor: return 'same_floor'
        else:                        return 'all'

    def _sample_pair(self, tries=100):
        last = (None, None, None)
        for _ in range(tries):
            mode = self._episode_mode()
            if mode == 'same_room':
                ra = rb = int(self.np_random.integers(len(self.rooms)))
            elif mode == 'same_floor':
                f = self.floor_keys[int(self.np_random.integers(len(self.floor_keys)))]
                idxs = self.floor_rooms[f]
                ra = idxs[int(self.np_random.integers(len(idxs)))]
                rb = idxs[int(self.np_random.integers(len(idxs)))]
            else:
                ra = rb = None
            s = self._sample_free(ra)
            g = self._sample_free(rb)
            last = (s, g, None)
            if s is None or g is None:
                continue
            d = float(np.linalg.norm(g - s))
            if d < self.min_separation:
                continue
            if self.max_goal_dist is not None and d > self.max_goal_dist:
                continue
            return s, g, None
        return last

    def _route_dist(self, p):
        return float(np.linalg.norm(self.goal - p))

    def _raycast(self, pos):
        out = np.empty(len(self.ray_dirs), dtype=np.float32)
        for k, d in enumerate(self.ray_dirs):
            hit = self.ray_max
            t = self.ray_step
            while t <= self.ray_max:
                if not is_point_free(pos + d * t, self.tree, self.clearance):
                    hit = t
                    break
                t += self.ray_step
            out[k] = hit / self.ray_max
        return out

    def _obs(self):
        rel  = self.goal - self.pos
        dist = float(np.linalg.norm(rel))
        unit = rel / (dist + 1e-8)
        dnorm = min(dist / self.diag, 1.0)
        return np.concatenate([unit.astype(np.float32),
                               np.array([dnorm], np.float32),
                               self._raycast(self.pos)])

    def _resolve_point(self, pt):
        bounds = [[self.bounds_lo[k], self.bounds_hi[k]] for k in range(3)]
        return find_nearest_free(np.array(pt, float), self.tree, bounds,
                                 radius=self.clearance)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        opt_s = options.get('start')
        opt_g = options.get('goal')

        if opt_s is not None or opt_g is not None:
            self.pos  = self._resolve_point(opt_s) if opt_s is not None else self._sample_free()
            self.goal = self._resolve_point(opt_g) if opt_g is not None else self._sample_free()
        else:
            self.pos, self.goal, _ = self._sample_pair()

        if self.pos is None or self.goal is None:
            self.pos  = self.coord[0].astype(np.float64) + np.array([0.5, 0, 0.5])
            self.goal = self.pos + np.array([max(self.min_separation, 1.0), 0, 0])

        self.door_wp     = None
        self.door_passed = True
        self.steps = 0
        self.path  = [self.pos.copy()]
        return self._obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, np.float64), -1.0, 1.0)
        new    = self.pos + action * self.max_step
        self.steps += 1

        prev_dist      = self._route_dist(self.pos)
        out_of_bounds  = bool(np.any(new < self.bounds_lo) or np.any(new > self.bounds_hi))
        hit = (not out_of_bounds) and (not is_edge_free(self.pos, new, self.tree,
                                                         radius=self.clearance))

        terminated = False
        if out_of_bounds or hit:
            reward     = -self.collision_penalty
            terminated = True
        else:
            self.pos = new
            self.path.append(self.pos.copy())
            new_dist = float(np.linalg.norm(self.goal - self.pos))
            reward   = (prev_dist - self._route_dist(self.pos)) * self.progress_scale \
                       - self.step_penalty
            if new_dist <= self.goal_radius:
                reward    += self.goal_bonus
                terminated = True

        success   = bool(terminated and reward > 0)
        truncated = (self.steps >= self.max_steps)
        if truncated and not terminated:
            reward -= self.timeout_penalty
        info = {"dist": float(np.linalg.norm(self.goal - self.pos)),
                "is_success": success}
        return self._obs(), float(reward), terminated, truncated, info


# load_obstacles 를 train_common_tqc 에서 호출할 수 있게 alias
load_obstacles = load_obstacles_npy
