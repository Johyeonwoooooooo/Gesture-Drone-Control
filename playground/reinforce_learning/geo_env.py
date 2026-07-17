"""
geo_env.py — v2 재설계: 측지거리(geodesic) 보상 + 자동 커리큘럼 환경
──────────────────────────────────────────────────────────────────
기존 drone_env(수동 커리큘럼 1→2a~2d, 문 BFS 경유 보상)의 문제를 RL 이론으로
정면 해결한 버전. 정책은 여전히 순수 reactive(센서만 봄)지만, '학습 신호'를 바꿨다.

핵심 아이디어 2개:

  1) 측지거리 potential-based shaping (Ng et al. 1999)
     직선거리 진척 보상은 목표가 옆방/딴층이면 벽으로 유인한다(2c 정체의 원인).
     대신 집을 0.3m 격자로 깔고 목표까지의 '측지거리'(벽을 돌아가는 최단거리,
     Dijkstra)를 포텐셜 φ 로 써서  r = (φ(s) - φ(s')) × scale  로 준다.
     → 보상 기울기가 항상 문/계단을 통과하는 실제 최단경로를 가리킨다.
     potential-based 라 최적 정책이 바뀌지 않는다는 것이 증명돼 있음.
     ★ φ(지도 지식)는 학습 중 보상 계산에만 쓰인다. 관측에는 안 들어가므로
       배포 시 정책은 지도 없이 난다 (reward is privileged, policy is not).

  2) 성공률 적응형 자동 커리큘럼
     목표 셀을 정하면 Dijkstra 필드가 공짜로 생기므로, '시작점'을 원하는
     측지거리 밴드 [d_max-3, d_max] 에서 골라 난이도를 연속 조절할 수 있다.
     최근 성공률 >60% → d_max +0.25m, <30% → -0.25m. 수동 단계(2a~2d) 불필요.

  계단: 기본 여유반경(0.235m)으론 물리적으로 통과 불가 → 계단 체인 근처 0.9m
  에서만 0.12m 로 완화(_clr). 측지 격자도 같은 규칙이라 필드가 계단을 통과한다.

관측(21) = 목표 방향 단위벡터(3) + 목표 직선거리(1) + 14방향 레이(14) + 직전 행동(3)
행동(3)  = 속도벡터, 스텝당 최대 0.3m
보상     = 측지 진척×5 − 0.02/스텝 − 충돌 10(종료) + 도착 100(종료) − 타임아웃 10
"""

import os
from collections import deque

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scipy.spatial import cKDTree
from scipy import sparse
from scipy.sparse.csgraph import dijkstra, connected_components

# drone_env 가 rrt_star_drone / hierarchical_plan 의 sys.path 를 세팅해 준다
from drone_env import load_obstacles, _ray_directions, _STAIR_CACHE, DroneHouseEnv
import rrt_star_drone as R
from hierarchical_plan import load_graph

_BASE = os.path.dirname(os.path.abspath(__file__))


# 수평 중심 레이 배치: 수평 12방향(30° 간격) + 위/아래 = 14개 (차원 동일).
# 기본 배치(축6+꼭짓점 대각8)는 대각선이 35° 기울어져 2~3m 안에 바닥/천장에 박힘 →
# 원거리에선 수평 4방향(90° 간격)만 남아 문이 레이 사이로 빠짐. 문은 수평면에 있으므로
# 수평 해상도에 몰빵하는 게 옳다 (상하 2개는 계단 구멍·천장 감지용으로 유지).
def _ray_directions_horiz14():
    dirs = []
    for k in range(12):
        a = 2.0 * np.pi * k / 12.0
        dirs.append((np.cos(a), np.sin(a), 0.0))
    dirs += [(0.0, 0.0, 1.0), (0.0, 0.0, -1.0)]
    v = np.array(dirs, dtype=np.float64)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


class DroneGeoEnv(gym.Env):
    """측지거리 shaping + 자동 커리큘럼으로 전체집(층 넘기 포함)을 한 번에 학습."""
    metadata = {"render_modes": []}

    def __init__(self,
                 voxel=0.06,
                 grid_v=0.30,          # 측지 격자 간격(m)
                 max_step=0.30,
                 clearance=R.OBSTACLE_RADIUS,
                 guide_clearance=0.20,  # ★ 안내(측지 격자/간선) 전용 최소 여유(m).
                                        # 물리(clearance)와 별도 — 물리 0.12 통일 후
                                        # 여유 0.13~0.15짜리 '키홀'(스캔 틈새 유령 지름길)이
                                        # 그래프에 편입돼 필드가 비행 불가 통로로 유도하는
                                        # 사고(0717) 방지. 계단 체인 stair_relax 안은
                                        # 기존대로 stair_clearance 로 완화(실계단은 통과).
                                        # 유효값 = max(clearance, guide_clearance) 라
                                        # 기본 물리(0.235)에선 영향 없음.
                 stair_clearance=0.12,
                 stair_relax=1.2,      # 계단 체인 근처 이 반경(m)에서 여유반경 완화.
                                       # 0.9는 계단 입구 턱에서 여유반경이 급변해
                                       # (0.235→0.12) 오라클도 걸림 — 1.2로 진입로 포함
                 ray_max=2.0,
                 ray_step=0.15,
                 goal_radius=0.5,
                 max_steps=400,
                 progress_scale=5.0,
                 step_penalty=0.02,
                 collision_penalty=10.0,
                 goal_bonus=100.0,
                 timeout_penalty=10.0,
                 bump_penalty=None,    # 설정 시 '범퍼 물리': 충돌해도 안 죽고 제자리 +
                                       # 이 페널티. 좁은 계단 통과를 연습할 수 있게 함
                                       # (즉사 규칙은 성공 경험 자체를 차단). Tello 는
                                       # 프로펠러 가드가 있어 현실적으로도 타당
                 ray_layout='axes14',  # 'axes14'(축6+대각8, 구버전) | 'horiz14'(수평12+상하2)
                 subgoal_dist=None,    # 설정(m) 시 관측의 목표를 '측지 내리막 subgoal_dist
                                       # 앞 지점(carrot)'으로 교체 — 계층(하이브리드) 모드.
                                       # 순수 센서 정책의 천장(~11m, 전역정보 부재) 해소용.
                                       # 도착/성공 판정과 보상은 여전히 최종 목표 기준.
                 priv_obs=False,       # True면 관측을 Dict{'sensor','priv'} 로 분리.
                                       # priv = 측지거리 φ + 측지 하강방향(특권 정보) —
                                       # asymmetric critic 용. actor 는 구조적으로 sensor 만 봄
                 curriculum=True,      # False = 평가용(전 난이도 균일 샘플)
                 d_init=4.0,           # 커리큘럼 시작 측지거리 상한(m)
                 d_band=3.0,           # 시작점을 [d_max-d_band, d_max] 에서 샘플
                 d_cap=45.0,           # d_max 상한 (집 전체 대각 측지거리보다 크게)
                 adapt_window=30,      # 성공률 이동창(에피소드)
                 goal_hold=8,          # 같은 목표(=같은 Dijkstra 필드)로 몇 에피소드 쓸지
                 explore_mix=0.2,      # 커리큘럼 중 이 확률로 '전 난이도 균일' 에피소드를
                                       # 섞음 — 밴드가 올라간 뒤 짧은 과제 망각 방지.
                                       # (이 에피소드는 난이도 적응 통계에서 제외)
                 stair_mix=0.0,        # 커리큘럼 중 이 확률로 '계단 과외' 에피소드
                                       # (계단 체인 한쪽 끝 → 반대쪽 끝, 짧은 층간 과제).
                                       # 계단 통과 경험을 강제로 쌓는다. 적응 통계 제외.
                 obstacles=None):
        super().__init__()
        self.coord = load_obstacles(voxel) if obstacles is None else obstacles
        self.tree  = cKDTree(self.coord)

        m = 0.3
        self.bounds_lo = self.coord.min(0) - m
        self.bounds_hi = self.coord.max(0) + m
        self.diag = float(np.linalg.norm(self.bounds_hi - self.bounds_lo))

        self.grid_v            = float(grid_v)
        self.max_step          = float(max_step)
        self.clearance         = float(clearance)
        self.guide_clearance   = max(float(clearance), float(guide_clearance))
        self.stair_clearance   = float(stair_clearance)
        self.ray_max           = float(ray_max)
        self.ray_step          = float(ray_step)
        self.goal_radius       = float(goal_radius)
        self.max_steps         = int(max_steps)
        self.progress_scale    = float(progress_scale)
        self.step_penalty      = float(step_penalty)
        self.collision_penalty = float(collision_penalty)
        self.goal_bonus        = float(goal_bonus)
        self.timeout_penalty   = float(timeout_penalty)
        self.curriculum        = bool(curriculum)
        self.d_max             = float(d_init)
        self.d_band            = float(d_band)
        self.d_cap             = float(d_cap)
        self.goal_hold         = int(goal_hold)
        self.explore_mix       = float(explore_mix)
        self.stair_mix         = float(stair_mix)
        self.subgoal_dist      = None if subgoal_dist is None else float(subgoal_dist)
        self.stair_relax       = float(stair_relax)
        self.bump_penalty      = None if bump_penalty is None else float(bump_penalty)

        # 계단 체인(grid A* 캐시, drone_env 가 만든 것 재사용 — 없으면 한 번 생성)
        if not os.path.exists(_STAIR_CACHE):
            print("[geo] 계단 체인 캐시가 없어 생성합니다 (최초 1회)...")
            DroneHouseEnv(obstacles=self.coord, sample_full_house=True)
        d = np.load(_STAIR_CACHE)
        chains = [d[f'chain_{k}'].astype(np.float64) for k in range(int(d['n']))]
        self._chains = chains                       # 검증된(구간 비행가능) 계단 웨이포인트
        self.stair_tree = cKDTree(np.vstack(chains)) if chains else None

        # 방 bbox (측지 격자를 '집 안'으로 제한 — 스캔 구멍으로 밖으로 새는 것 방지)
        g = load_graph()
        self.room_boxes = [(np.array(r['bbox_min'], float), np.array(r['bbox_max'], float))
                           for r in g['rooms'].values()]

        self._build_geo_grid()

        self.ray_dirs = (_ray_directions_horiz14() if ray_layout == 'horiz14'
                         else _ray_directions())
        self.priv_obs = bool(priv_obs)
        n_obs = 3 + 1 + len(self.ray_dirs) + 3       # 목표(4) + 레이(14) + 직전행동(3)
        sensor_space = spaces.Box(-1.0, 1.0, shape=(n_obs,), dtype=np.float32)
        if self.priv_obs:
            # priv = [φ정규화(1), 측지 하강 단위벡터(3)] — critic 전용 특권 정보
            self.observation_space = spaces.Dict({
                'sensor': sensor_space,
                'priv':   spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)})
        else:
            self.observation_space = sensor_space
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)

        self.pos = None
        self._carrot_prev = None
        self.goal = None
        self.prev_a = np.zeros(3)
        self.steps = 0
        self.path = []
        self._field = None            # 현재 목표의 측지거리 필드 (노드별, m)
        self._ep_on_goal = 0          # 현재 필드로 치른 에피소드 수
        self._succ = deque(maxlen=int(adapt_window))
        self._explore_ep = False      # 이번 에피소드가 explore_mix 샘플인지

    # ── 측지 격자: 유효셀(집 안 + 통과 가능) + 26이웃 그래프 ──────────
    def _build_geo_grid(self):
        v = self.grid_v
        dims = np.ceil((self.bounds_hi - self.bounds_lo) / v).astype(int) + 1
        self.grid_dims = dims
        ii = np.indices(dims).reshape(3, -1).T
        centers = self.bounds_lo + ii * v
        try:
            dist, _ = self.tree.query(centers, workers=-1)
        except TypeError:                                  # 구버전 scipy
            dist, _ = self.tree.query(centers)
        # ★ 안내 여유(guide_clearance): 계단 체인 근처는 stair_clearance 로 완화하고
        #   0.8m 에 걸쳐 guide_clearance 로 램프. 물리(_clr, clearance)보다 엄격해서
        #   '기하적으로 뚫려 있지만 사실상 비행 불가'인 키홀이 필드에 편입되지 않는다.
        if self.stair_tree is not None:
            try:
                ds, _ = self.stair_tree.query(centers, workers=-1)
            except TypeError:
                ds, _ = self.stair_tree.query(centers)
            t = np.clip((ds - self.stair_relax) / 0.8, 0.0, 1.0)
            cell_gc_all = self.stair_clearance + t * (self.guide_clearance
                                                      - self.stair_clearance)
        else:
            cell_gc_all = np.full(len(centers), self.guide_clearance)
        free = dist > cell_gc_all
        inroom = np.zeros(len(centers), bool)
        for blo, bhi in self.room_boxes:
            inroom |= np.all((centers >= blo - 0.1) & (centers <= bhi + 0.1), axis=1)
        valid = (free & inroom).reshape(tuple(dims))

        self.nid = -np.ones(tuple(dims), np.int32)
        nv = int(valid.sum())
        self.nid[valid] = np.arange(nv, dtype=np.int32)
        self.node_pos = centers[valid.ravel()]

        # 셀별 유효 '안내' 여유반경 (계단 램프 반영) — 간선 검증에 사용.
        # 물리(_clr)가 아니라 guide 램프를 쓴다 (간선도 키홀 기준으로 걸러야 함).
        pv = self.node_pos
        cell_clr = cell_gc_all[valid.ravel()]

        # 26이웃(대칭이라 13방향만 저장) 희소 그래프.
        # ★ 간선(셀→이웃셀 직선 구간)도 충돌 검증한다 — 셀 중심만 비었는지 보면
        #   얇은 벽 양쪽 셀이 '이웃'으로 붙어 측지 필드가 벽을 관통해 흐른다
        #   (여유반경을 줄일수록 심해짐 — carrot 이 벽 너머를 가리키는 사고의 원인).
        cache_key = (float(v), float(self.clearance), float(self.guide_clearance),
                     float(self.stair_clearance), float(self.stair_relax), nv)
        graph_cache = os.path.join(_BASE, 'geo_graph_cache.npz')
        rows = cols = wts = None
        if os.path.exists(graph_cache):
            g = np.load(graph_cache)
            if tuple(g['key']) == cache_key:
                rows, cols, wts = g['rows'], g['cols'], g['wts']
        if rows is None:
            print("[geo] 격자 간선 충돌 검증 중... (최초 1회, 캐시됨)")
            rows_l, cols_l, wts_l = [], [], []
            offs = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                    for dz in (-1, 0, 1) if (dx, dy, dz) > (0, 0, 0)]
            dropped = total = 0
            for o in offs:
                sa = tuple(slice(max(0, -o[k]), dims[k] - max(0, o[k])) for k in range(3))
                sb = tuple(slice(max(0, o[k]),  dims[k] - max(0, -o[k])) for k in range(3))
                A, B = self.nid[sa].ravel(), self.nid[sb].ravel()
                mask = (A >= 0) & (B >= 0)
                ia, ib = A[mask], B[mask]
                seg = np.array(o, float) * v
                L = float(np.linalg.norm(seg))
                r_edge = np.minimum(cell_clr[ia], cell_clr[ib])
                ok = np.ones(len(ia), bool)
                K = max(3, int(L / (self.stair_clearance * 0.5)))
                for t in np.linspace(0.0, 1.0, K)[1:-1]:      # 끝점(셀 중심)은 이미 검증됨
                    pts = pv[ia[ok]] + t * seg
                    try:
                        dd, _ = self.tree.query(pts, workers=-1)
                    except TypeError:
                        dd, _ = self.tree.query(pts)
                    sub = ok.nonzero()[0]
                    ok[sub[dd <= r_edge[ok]]] = False
                total += len(ia); dropped += int((~ok).sum())
                rows_l.append(ia[ok]); cols_l.append(ib[ok])
                wts_l.append(np.full(int(ok.sum()), L))
            rows = np.concatenate(rows_l); cols = np.concatenate(cols_l)
            wts = np.concatenate(wts_l)
            np.savez_compressed(graph_cache, key=np.array(cache_key),
                                rows=rows, cols=cols, wts=wts)
            print(f"[geo] 간선 {total:,}개 중 벽 관통 {dropped:,}개 제거 후 캐시 저장")
        self.csr = sparse.csr_matrix((wts, (rows, cols)), shape=(nv, nv))

        # 주 연결성분(층 3개가 계단으로 이어진 덩어리)에서만 목표를 뽑는다
        _, labels = connected_components(self.csr, directed=False)
        main = np.argmax(np.bincount(labels))
        self.main_nodes = np.where(labels == main)[0]
        frac = len(self.main_nodes) / max(nv, 1)
        print(f"[geo] 측지 격자 {dims} 유효셀 {nv:,}개, 주성분 {frac:.0%}")

    # ── 위치별 유효 여유반경 (계단 근처만 완화 — drone_env 와 동일 발상) ──
    # 경계에서 0.12→0.235 로 급변하면 '보이지 않는 벽'이 생겨 계단 진입로에서
    # 급사한다(실패 분석에서 사망 지점이 경계와 일치) → 0.8m 에 걸쳐 선형 전환.
    def _clr(self, p):
        if self.stair_tree is None:
            return self.clearance
        d = float(self.stair_tree.query(p)[0])
        if d <= self.stair_relax:
            return self.stair_clearance
        if d >= self.stair_relax + 0.8:
            return self.clearance
        t = (d - self.stair_relax) / 0.8
        return self.stair_clearance + t * (self.clearance - self.stair_clearance)

    # ── 포텐셜 φ(p): 그 위치 셀의 측지거리 (막힌 셀이면 주변 셀 최소값) ──
    def _phi(self, p):
        c = np.clip(np.round((p - self.bounds_lo) / self.grid_v).astype(int),
                    0, self.grid_dims - 1)
        n = self.nid[c[0], c[1], c[2]]
        if n >= 0 and np.isfinite(self._field[n]):
            return float(self._field[n])
        for r in (1, 2):
            lo = np.maximum(c - r, 0)
            hi = np.minimum(c + r, self.grid_dims - 1)
            sub = self.nid[lo[0]:hi[0]+1, lo[1]:hi[1]+1, lo[2]:hi[2]+1].ravel()
            sub = sub[sub >= 0]
            if len(sub):
                vals = self._field[sub]
                vals = vals[np.isfinite(vals)]
                if len(vals):
                    return float(vals.min())
        return float(np.linalg.norm(self.goal - p))    # 측지 불가 지점 폴백(드묾)

    # ── 새 목표 + Dijkstra 필드 ──────────────────────────────
    def _pick_goal(self, goal_pos=None):
        if goal_pos is not None:                        # 평가용 지정 목표
            d = np.linalg.norm(self.node_pos[self.main_nodes] - np.asarray(goal_pos), axis=1)
            gn = int(self.main_nodes[int(np.argmin(d))])
        else:
            gn = int(self.main_nodes[int(self.np_random.integers(len(self.main_nodes)))])
        self._field = dijkstra(self.csr, directed=False, indices=gn)
        self.goal = self.node_pos[gn].copy()
        self._ep_on_goal = 0

    # ── 시작점: 커리큘럼 밴드 [d_max-d_band, d_max] 측지거리에서 샘플 ──
    # explore_mix 확률로는 전 난이도 균일 샘플(짧은 과제 망각 방지 + 밴드 밖 노출).
    def _pick_start(self):
        f = self._field
        self._explore_ep = bool(self.curriculum and
                                self.np_random.random() < self.explore_mix)
        if self.curriculum and not self._explore_ep:
            lo_d = max(1.0, self.d_max - self.d_band)
            cand = np.where((f >= lo_d) & (f <= self.d_max))[0]
        else:
            cand = np.where(np.isfinite(f) & (f >= 1.5))[0]
        if len(cand) == 0:
            cand = np.where(np.isfinite(f) & (f >= 1.0))[0]
        n = int(cand[int(self.np_random.integers(len(cand)))])
        p = self.node_pos[n] + self.np_random.uniform(-0.1, 0.1, 3)
        if not R.is_point_free(p, self.tree, self._clr(p)):
            p = self.node_pos[n].copy()
        return p

    # ── 계단 과외 과제: 검증된 계단 체인 한쪽 끝 근처 → 반대쪽 끝 (짧은 층간 과제) ──
    # 정책이 계단 통과 성공 경험을 강제로 쌓게 한다. 목표와 시작을 함께 정한다.
    def _pick_stair_task(self):
        ch = self._chains[int(self.np_random.integers(len(self._chains)))]
        a, b = ch[0], ch[-1]
        if self.np_random.random() < 0.5:
            a, b = b, a
        self._pick_goal(goal_pos=b)
        self._ep_on_goal = self.goal_hold           # 다음 일반 에피소드는 새 목표로
        d = np.linalg.norm(self.node_pos - a, axis=1)
        cand = np.where((d < 2.0) & np.isfinite(self._field))[0]
        if len(cand) == 0:
            cand = np.where(np.isfinite(self._field) & (self._field >= 1.0))[0]
        n = int(cand[int(self.np_random.integers(len(cand)))])
        p = self.node_pos[n] + self.np_random.uniform(-0.1, 0.1, 3)
        if not R.is_point_free(p, self.tree, self._clr(p)):
            p = self.node_pos[n].copy()
        return p

    # ── 보조: carrot — 측지 필드 내리막을 subgoal_dist 만큼 걸어간 지점 ──
    # 현재 셀에서 이웃 중 φ 최소 셀로 반복 이동(격자 최급강하). 누적 이동거리가
    # subgoal_dist 에 닿거나 목표 셀에 도달하면 그 지점을 반환. 목표가 그보다
    # 가까우면 최종 목표를 그대로 쓴다.
    def _carrot(self):
        # 격자 간선이 전부 충돌 검증돼 있으므로(벽 관통 간선 제거) 계단 포함
        # 전 구간을 이 격자 레일 하나로 안내한다 (계단 전용 추종기는 격자↔체인
        # 전환 진동·제자리걸음 등 문제가 더 많아 제거).
        c = np.clip(np.round((self.pos - self.bounds_lo) / self.grid_v).astype(int),
                    0, self.grid_dims - 1)
        n = self.nid[c[0], c[1], c[2]]
        if n < 0 or not np.isfinite(self._field[n]):
            best = None                              # 막힌 셀이면 주변 유효셀에서 출발
            for r in (1, 2):
                lo = np.maximum(c - r, 0)
                hi = np.minimum(c + r, self.grid_dims - 1)
                sub = self.nid[lo[0]:hi[0]+1, lo[1]:hi[1]+1, lo[2]:hi[2]+1].ravel()
                sub = sub[sub >= 0]
                sub = sub[np.isfinite(self._field[sub])]
                if len(sub):
                    # ★ '가장 가까운' 셀 — φ 최소 셀은 벽 건너편일 수 있다(목표에
                    #   더 가까우니까). 복귀 목표가 벽 뒤가 되면 영구히 낀다.
                    dd = np.linalg.norm(self.node_pos[sub] - self.pos, axis=1)
                    best = int(sub[int(np.argmin(dd))])
                    break
            if best is None:
                return self.goal
            n = best
            c = np.round((self.node_pos[n] - self.bounds_lo) / self.grid_v).astype(int)
        # 내리막을 걷되, 드론에서 '직선으로 보이는' 마지막 지점까지만 (LOS 클램프).
        # 모퉁이 너머를 가리키면 직진 추종이 벽을 스치므로, carrot 은 항상 가시권 안에 둔다.
        n0 = n                                       # 출발 셀 (복귀용)
        walked = 0.0
        vis_n = None
        first_dn = None                              # 첫 내리막 이웃 (레일 크롤용)
        for _ in range(int(self.subgoal_dist / self.grid_v) + 4):
            if self._field[n] <= self.grid_v:        # 목표 셀 도달
                self._carrot_prev = self.goal.copy()
                return self.goal
            best_n, best_f, best_step = n, self._field[n], 0.0
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        if not (dx or dy or dz):
                            continue
                        x, y, z = c[0] + dx, c[1] + dy, c[2] + dz
                        if not (0 <= x < self.grid_dims[0] and
                                0 <= y < self.grid_dims[1] and
                                0 <= z < self.grid_dims[2]):
                            continue
                        m = self.nid[x, y, z]
                        if m >= 0 and self._field[m] < best_f:
                            best_n, best_f = int(m), float(self._field[m])
                            best_step = float(np.linalg.norm([dx, dy, dz])) * self.grid_v
                            best_c = (x, y, z)
            if best_n == n:                           # 내리막 없음(이론상 목표뿐)
                break
            n, c = best_n, np.array(best_c)
            if first_dn is None:
                first_dn = n
            p = self.node_pos[n]
            clr = min(self._clr(self.pos), self._clr(p))
            if R.is_edge_free(self.pos, p, self.tree, radius=clr):
                vis_n = n                             # 마지막 가시 지점 갱신
            elif vis_n is not None:
                break                                 # 시야가 끊기면 거기까지
            walked += best_step
            if walked >= self.subgoal_dist:
                break
        if vis_n is not None:
            self._carrot_prev = self.node_pos[vis_n].copy()
            return self._carrot_prev
        # 내리막 어느 지점도 안 보임 = 드론이 셀 중심에서 벗어나 구석에 낀 상태.
        # ★ 히스테리시스: 직전에 보였던 carrot 이 아직 측지적으로 충분히 '앞'이고
        #   멀지 않으면 그것을 유지한다. 셀 중심 폴백은 드론 위치와 겹쳐 관측 목표가
        #   0 이 되는 고착점(내리막 carrot 과 매 스텝 번갈아 나오는 리밋사이클)이었음.
        if self._carrot_prev is not None and \
                self._phi(self._carrot_prev) < self._phi(self.pos) - 0.3 and \
                float(np.linalg.norm(self._carrot_prev - self.pos)) <= self.subgoal_dist * 2:
            return self._carrot_prev
        # ★ 레일 크롤: 최급강하 체인(argmin)이 안 보여도, '드론의 현재 위치에서
        #   직선으로 갈 수 있는' 내리막 이웃이 있으면 그중 φ 최소로 한 칸 전진.
        #   argmin 하나만 보는 위 워크와 달리 내리막 이웃 전부를 스캔한다.
        #   검사 기준점은 셀 중심이 아니라 self.pos — 중심 기준으로 고르면 드론이
        #   10cm만 비껴 있어도 못 가는 목표를 주게 되어 무한 범프한다.
        if first_dn is not None:
            c0 = np.round((self.node_pos[n0] - self.bounds_lo) / self.grid_v).astype(int)
            cands = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        if not (dx or dy or dz):
                            continue
                        x, y, z = c0[0] + dx, c0[1] + dy, c0[2] + dz
                        if not (0 <= x < self.grid_dims[0] and
                                0 <= y < self.grid_dims[1] and
                                0 <= z < self.grid_dims[2]):
                            continue
                        m = self.nid[x, y, z]
                        if m >= 0 and self._field[m] < self._field[n0]:
                            cands.append(int(m))
            cands.sort(key=lambda m: self._field[m])
            for m in cands:
                pm = self.node_pos[m]
                if R.is_edge_free(self.pos, pm, self.tree,
                                  radius=min(self._clr(self.pos), self._clr(pm))):
                    self._carrot_prev = pm.copy()
                    return self._carrot_prev
        # 어느 내리막도 현 위치에선 안 보임 → 검증된 레일(셀 중심)로 복귀.
        # 정확히 중심에 서면 Dijkstra 전임자 간선(검증됨)이 반드시 뚫려 있어
        # 다음 쿼리에서 위 크롤이 성공한다.
        # 검증된 레일(셀 중심)로 복귀시키면 다음 셀은 보인다(간선 검증됨).
        return self.node_pos[n0]

    # ── 관측 ────────────────────────────────────────────────
    def _raycast(self, pos):
        clr = self._clr(pos)
        out = np.empty(len(self.ray_dirs), dtype=np.float32)
        for k, d in enumerate(self.ray_dirs):
            hit = self.ray_max
            t = self.ray_step
            while t <= self.ray_max:
                if not R.is_point_free(pos + d * t, self.tree, clr):
                    hit = t
                    break
                t += self.ray_step
            out[k] = hit / self.ray_max
        return out

    def _obs(self):
        # 서브골 모드: 정책 눈에는 '측지 내리막 carrot'이 목표처럼 보인다.
        target = self._carrot() if self.subgoal_dist is not None else self.goal
        rel = target - self.pos
        dist = float(np.linalg.norm(rel))
        unit = rel / (dist + 1e-8)
        dnorm = min(dist / self.diag, 1.0)
        sensor = np.concatenate([unit.astype(np.float32),
                                 np.array([dnorm], np.float32),
                                 self._raycast(self.pos),
                                 self.prev_a.astype(np.float32)])
        if not self.priv_obs:
            return sensor
        return {'sensor': sensor, 'priv': self._priv()}

    # ── 특권 관측 (critic 전용): φ + 측지 하강방향 (중심차분) ──────────
    def _priv(self):
        phi = self._phi(self.pos)
        g = np.zeros(3)
        for k in range(3):
            e = np.zeros(3); e[k] = self.grid_v
            g[k] = self._phi(self.pos + e) - self._phi(self.pos - e)
        n = float(np.linalg.norm(g))
        desc = (-g / n) if n > 1e-9 else np.zeros(3)
        return np.concatenate([[np.clip(phi / 40.0, 0.0, 1.0)],
                               desc]).astype(np.float32)

    # ── Gymnasium API ────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        stair_ep = (self.stair_mix > 0 and self.curriculum and bool(self._chains) and
                    options.get('goal') is None and options.get('start') is None and
                    self.np_random.random() < self.stair_mix)
        if stair_ep:                                    # 계단 과외: 목표+시작 동시 설정
            self.pos = self._pick_stair_task()
        elif options.get('goal') is not None:           # 평가: 목표 직접 지정
            self._pick_goal(goal_pos=options['goal'])
        elif seed is not None or self._field is None or self._ep_on_goal >= self.goal_hold:
            # seed 지정 리셋은 항상 새 목표 — 같은 seed = 같은 에피소드 보장(재현성).
            # 학습(무seed) 리셋만 goal_hold 동안 필드를 재사용해 Dijkstra 비용을 아낀다.
            self._pick_goal()
        self._ep_on_goal += 1

        if stair_ep:
            self._explore_ep = True                     # 난이도 적응 통계에서 제외
        elif options.get('start') is not None:
            self._explore_ep = False                    # 지정 시작점은 적응 통계 제외 대상 아님
            p = np.asarray(options['start'], float)
            if not R.is_point_free(p, self.tree, self._clr(p)):
                # 찍은 점이 벽 위(점군 표면)면 가장 가까운 도달 가능 셀로 스냅
                nn = self.main_nodes[int(np.argmin(np.linalg.norm(
                    self.node_pos[self.main_nodes] - p, axis=1)))]
                p = self.node_pos[int(nn)].copy()
            self.pos = p
        else:
            self.pos = self._pick_start()

        self.prev_a = np.zeros(3)
        self.steps = 0
        self._bumps = 0
        self._carrot_prev = None
        self.path = [self.pos.copy()]
        self._phi_prev = self._phi(self.pos)
        return self._obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, np.float64), -1.0, 1.0)
        new = self.pos + action * self.max_step
        self.steps += 1

        out_of_bounds = bool(np.any(new < self.bounds_lo) or np.any(new > self.bounds_hi))
        clr = min(self._clr(self.pos), self._clr(new))
        hit = (not out_of_bounds) and (not R.is_edge_free(self.pos, new, self.tree,
                                                          radius=clr))
        terminated = False
        success = False
        if out_of_bounds or hit:
            if self.bump_penalty is not None:
                # 범퍼 물리 + 슬라이드: 부딪히면 행동의 축 성분 중 갈 수 있는 가장 큰
                # 성분만 적용(프로펠러 가드가 벽을 타듯). 완전 제자리 정지는 결정론
                # 정책이 같은 막힌 행동을 무한 반복하는 고착점이었음. 페널티는 그대로
                # 부과되므로 벽 타기는 자유비행보다 항상 손해.
                self._bumps += 1
                self.prev_a = action
                slid = None
                if not out_of_bounds:
                    disp = action * self.max_step
                    for k in np.argsort(-np.abs(disp)):
                        if abs(disp[k]) < 1e-9:
                            continue
                        cand = self.pos.copy()
                        cand[k] += disp[k]
                        if np.any(cand < self.bounds_lo) or np.any(cand > self.bounds_hi):
                            continue
                        if R.is_edge_free(self.pos, cand, self.tree,
                                          radius=min(self._clr(self.pos), self._clr(cand))):
                            slid = cand
                            break
                if slid is None:
                    reward = -self.bump_penalty
                else:
                    self.pos = slid
                    self.path.append(self.pos.copy())
                    phi = self._phi(self.pos)
                    reward = (self._phi_prev - phi) * self.progress_scale \
                        - self.step_penalty - self.bump_penalty
                    self._phi_prev = phi
                    if float(np.linalg.norm(self.goal - self.pos)) <= self.goal_radius:
                        reward += self.goal_bonus
                        terminated = True
                        success = True
            else:
                reward = -self.collision_penalty
                terminated = True
        else:
            self.pos = new
            self.path.append(self.pos.copy())
            self.prev_a = action
            phi = self._phi(self.pos)
            reward = (self._phi_prev - phi) * self.progress_scale - self.step_penalty
            self._phi_prev = phi
            if float(np.linalg.norm(self.goal - self.pos)) <= self.goal_radius:
                reward += self.goal_bonus
                terminated = True
                success = True

        truncated = (self.steps >= self.max_steps)
        if truncated and not terminated:
            reward -= self.timeout_penalty

        if (terminated or truncated) and not self._explore_ep:  # 자동 커리큘럼 적응
            # explore(전 난이도) 에피소드는 통계에서 제외 — 밴드 난이도만 반영
            self._succ.append(success)
            if self.curriculum and len(self._succ) == self._succ.maxlen:
                sr = float(np.mean(self._succ))
                if sr > 0.6:
                    self.d_max = min(self.d_max + 0.25, self.d_cap)
                elif sr < 0.3:
                    self.d_max = max(2.0, self.d_max - 0.25)

        info = {"dist": float(np.linalg.norm(self.goal - self.pos)),
                "is_success": success, "d_max": self.d_max, "bumps": self._bumps}
        return self._obs(), float(reward), terminated, truncated, info
