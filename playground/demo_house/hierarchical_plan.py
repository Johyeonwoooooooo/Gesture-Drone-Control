"""
hierarchical_plan.py
────────────────────
rooms_graph.json 을 이용한 계층적 경로계획.

  [상위] 방 그래프(edges) 에 A*  → 지나갈 방 시퀀스
  [하위] 시퀀스의 연속 방마다, 그 방의 npy 를 장애물로 RRT* 로
         다음 문(edge.door_center, = 직접 찍은 점에서 나온 목표점)까지 경로
  계단(stairs) 구간은 두 방 점군을 합쳐서 푼다.
  마지막에 모든 층을 좌→우로 나란히 펼쳐 전체 경로를 그린다.

사용:
  python hierarchical_plan.py 002 019           # 002 방 → 019 방
  python hierarchical_plan.py 002 019 --max-iter 5000
  python hierarchical_plan.py --list            # 방 목록만 출력
"""

import os
import sys
import json
import heapq
import argparse
import numpy as np
from scipy.spatial import cKDTree

_BASE  = os.path.dirname(os.path.abspath(__file__))
_GRAPH = os.path.join(_BASE, 'rooms_graph.json')
_NPY   = os.path.join(_BASE, 'npy')

# 기존 RRT* 재사용 (rrt_star_drone.py 는 auto_driving/draft 로 이동됨)
sys.path.insert(0, os.path.join(_BASE, '..', 'auto_driving', 'draft'))
import rrt_star_drone as R   # noqa: E402

_KO_FONT_PATHS = [r"C:\Windows\Fonts\malgun.ttf", r"C:\Windows\Fonts\NanumGothic.ttf"]


# ──────────────────────────────────────────
def load_graph():
    return json.load(open(_GRAPH, encoding='utf-8'))


def load_coord(rid, rooms):
    return np.load(os.path.join(_NPY, rooms[rid]['npy'], 'coord.npy')).astype(np.float64)


def voxel_down(coord, v=0.05):
    """장애물 KDTree 속도용 다운샘플 (간격 v < 드론반경 이라 충돌검사 안전)."""
    if v <= 0 or len(coord) == 0:
        return coord
    keys = np.floor(coord / v).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return coord[idx]


# ──────────────────────────────────────────
# [상위] 방 그래프 A*
# ──────────────────────────────────────────
def room_astar(rooms, edges, start, goal):
    adj = {r: [] for r in rooms}
    door = {}
    for e in edges:
        a, b = e['a'], e['b']
        w = float(np.linalg.norm(np.array(rooms[a]['center']) - np.array(rooms[b]['center'])))
        adj[a].append((b, w)); adj[b].append((a, w))
        door[(a, b)] = door[(b, a)] = e

    pq = [(0.0, start, [start])]
    seen = set()
    while pq:
        c, u, path = heapq.heappop(pq)
        if u == goal:
            return path, door
        if u in seen:
            continue
        seen.add(u)
        for v, w in adj[u]:
            if v not in seen:
                heapq.heappush(pq, (c + w, v, path + [v]))
    return None, door


# ──────────────────────────────────────────
# [하위] 한 구간 RRT*
# ──────────────────────────────────────────
def stair_points(ra, rb):
    """계단 edge: 두 방의 통로 점 중 가장 가까운 쌍 = (아래방 점, 위방 점).
       이 둘을 경유점으로 써서 수직 통로(계단)를 타고 오르내린다."""
    best = None
    for x in ra.get('passages', []):
        for y in rb.get('passages', []):
            d = float(np.linalg.norm(np.array(x) - np.array(y)))
            if best is None or d < best[0]:
                best = (d, list(map(float, x)), list(map(float, y)))
    if best is None:
        return None
    return best[1], best[2]   # (점 in ra, 점 in rb)


def door_points(ra, rb, edge):
    """door edge(A-B): 내가 직접 찍은 통로 점들만 경유점으로 (자동 중점 door_center 안 씀).
       한 방의 passages 는 '각 점 = 다른 이웃 방으로 가는 출입구' 이므로,
       이 edge 에 해당하는 점을 골라 A→B 순서로 돌려준다.
         - 양쪽 다 점 있음 → 가장 가까운 쌍 [A점, B점] (그 문에 해당하는 두 출입구)
         - 한쪽만 있음     → 그 점 [점] (상대 방 center 에 가장 가까운 출입구)
         - 둘 다 없음       → [edge door_center] (폴백)"""
    pa = [np.array(p, float) for p in ra.get('passages', [])]
    pb = [np.array(p, float) for p in rb.get('passages', [])]
    if pa and pb:
        best = None
        for x in pa:
            for y in pb:
                d = float(np.linalg.norm(x - y))
                if best is None or d < best[0]:
                    best = (d, x, y)
        return [list(map(float, best[1])), list(map(float, best[2]))]
    if pa:
        cb = np.array(rb['center'], float)
        x = min(pa, key=lambda p: float(np.linalg.norm(p - cb)))
        return [list(map(float, x))]
    if pb:
        ca = np.array(ra['center'], float)
        y = min(pb, key=lambda p: float(np.linalg.norm(p - ca)))
        return [list(map(float, y))]
    return [list(map(float, edge['door_center']))]


def find_vertical_opening(tree, cxy, z_lo, z_hi, xy_bounds, radius=R.OBSTACLE_RADIUS,
                          step=0.2, zstep=0.2):
    """cxy 주변(단, xy_bounds=아래방 footprint 안)에서 z_lo~z_hi 전체가 비어있는
       수직 컬럼 xy 를 찾는다 = 층을 관통하는 실제 계단 구멍. cxy 에 가장 가까운 것 반환.
       (건물 바깥의 빈 공간을 통로로 오인하지 않도록 footprint 로 제한)"""
    (xlo, xhi), (ylo, yhi) = xy_bounds
    zs = np.arange(z_lo, z_hi + 1e-6, zstep)
    best = None
    for x in np.arange(xlo, xhi + 1e-6, step):
        for y in np.arange(ylo, yhi + 1e-6, step):
            if all(R.is_point_free(np.array([x, y, z]), tree, radius=radius) for z in zs):
                d = (x - cxy[0]) ** 2 + (y - cxy[1]) ** 2
                if best is None or d < best[0]:
                    best = (d, (float(x), float(y)))
    return best[1] if best else None


def build_global_obstacles(rooms, voxel):
    """모든 방 점군을 합친 전역 장애물 (벽을 전부 포함 → 벽 뚫기 방지)."""
    clouds = [np.load(os.path.join(_NPY, rooms[r]['npy'], 'coord.npy')).astype(np.float64)
              for r in rooms]
    return voxel_down(np.vstack(clouds), voxel)


def plan_segment(tree, gbounds, start, goal, max_iter, pad=2.5,
                 radius=R.OBSTACLE_RADIUS, step_size=R.STEP_SIZE):
    """전역 장애물 tree 위에서 start→goal RRT*.
       샘플링은 두 점 주변 박스(±pad)로 제한(장애물은 전역이라 벽 못 뚫음).
       계단 등 좁은 구간은 radius(여유반경)↓, step_size↓ 로 통과."""
    s = R.find_nearest_free(np.array(start, float), tree, gbounds, radius=radius)
    g = R.find_nearest_free(np.array(goal,  float), tree, gbounds, radius=radius)
    if s is None or g is None:
        return None
    lo = np.minimum(s, g) - pad
    hi = np.maximum(s, g) + pad
    bounds = [[float(lo[k]), float(hi[k])] for k in range(3)]
    planner = R.RRTStar(s, g, tree, bounds, max_iter=max_iter,
                        step_size=step_size, obstacle_radius=radius)
    raw = planner.plan(verbose=False)
    if raw is None:
        return None
    return R.smooth_path(raw, tree, obs_radius=radius)


def grid_astar(tree, start, goal, lo, hi, radius, v=0.15, max_expand=300000):
    """3D 격자 A* (완전 탐색). lo~hi 박스 안에서 start→goal 충돌 없는 경로를 반드시 찾는다(있으면).
       좁은 계단 통로처럼 RRT* 가 약한 곳에 사용. 월드좌표 경로(N,3) 반환, 없으면 None."""
    import heapq
    from itertools import product
    start = np.asarray(start, float); goal = np.asarray(goal, float)
    lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    dims = (np.floor((hi - lo) / v).astype(int) + 1)
    dims = tuple(int(d) for d in dims)
    cache = {}

    def free(idx):
        c = cache.get(idx)
        if c is None:
            c = R.is_point_free(lo + np.array(idx) * v, tree, radius=radius)
            cache[idx] = c
        return c

    def to_idx(p):
        return tuple(int(np.clip(round((p[k] - lo[k]) / v), 0, dims[k] - 1)) for k in range(3))

    def snap(p):
        b = to_idx(p)
        if free(b):
            return b
        for rr in range(1, 12):                    # 막혀있으면 근처 free 격자로
            for d in product(range(-rr, rr + 1), repeat=3):
                idx = (b[0] + d[0], b[1] + d[1], b[2] + d[2])
                if all(0 <= idx[k] < dims[k] for k in range(3)) and free(idx):
                    return idx
        return None

    s = snap(start); gidx = snap(goal)
    if s is None or gidx is None:
        return None
    nbrs = [d for d in product((-1, 0, 1), repeat=3) if any(d)]

    def h(a):
        return ((a[0] - gidx[0]) ** 2 + (a[1] - gidx[1]) ** 2 + (a[2] - gidx[2]) ** 2) ** 0.5

    openh = [(h(s), 0.0, s)]; came = {}; gs = {s: 0.0}; closed = set(); cnt = 0
    while openh and cnt < max_expand:
        _, gc, cur = heapq.heappop(openh)
        if cur in closed:
            continue
        closed.add(cur); cnt += 1
        if cur == gidx:
            path = [cur]
            while cur in came:
                cur = came[cur]; path.append(cur)
            path.reverse()
            return np.array([lo + np.array(i) * v for i in path])
        for d in nbrs:
            nb = (cur[0] + d[0], cur[1] + d[1], cur[2] + d[2])
            if not all(0 <= nb[k] < dims[k] for k in range(3)):
                continue
            if nb in closed or not free(nb):
                continue
            nc = gc + (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]) ** 0.5
            if nc < gs.get(nb, 1e18):
                gs[nb] = nc; came[nb] = cur
                heapq.heappush(openh, (nc + h(nb), nc, nb))
    return None


# ──────────────────────────────────────────
def plan(start_room, goal_room, max_iter=4000, voxel=0.06, stair_radius=0.12):
    g = load_graph(); rooms = g['rooms']; edges = g['edges']
    if start_room not in rooms or goal_room not in rooms:
        print(f"[오류] 방 id 확인. 사용 가능: {', '.join(rooms)}"); return None

    seq, door = room_astar(rooms, edges, start_room, goal_room)
    if seq is None:
        print(f"[경로 없음] {start_room} → {goal_room} 는 그래프상 연결되어 있지 않습니다."); return None
    print(f"\n[상위] 방 경로: {' -> '.join(seq)}")

    print("\n[장애물] 전체 집 점군 구성 중...")
    coord = build_global_obstacles(rooms, voxel)
    tree = cKDTree(coord)
    gbounds = [[coord[:, k].min() - 0.3, coord[:, k].max() + 0.3] for k in range(3)]
    print(f"  장애물 점 {len(coord):,}개 (voxel {voxel}m) — 모든 방 벽 포함")

    # 경유점 + 각 구간 계획 방식.  plans[k] = wps[k]->wps[k+1] 를 어떻게 풀지
    #   'rrt'        : 같은 층 일반 구간 (RRT*)
    #   'grid'+lower : 계단 — 찍은 두 점(pa,pb) 사이를 격자 A* 로 (아래방 footprint 안에서)
    wps = [list(rooms[start_room]['center'])]
    labels, plans = [], []
    for i in range(len(seq) - 1):
        a, b = seq[i], seq[i + 1]
        e = door[(a, b)]
        if e['type'] == 'stairs':
            sp = stair_points(rooms[a], rooms[b])
            if sp:
                pa, pb = sp                      # 사용자가 찍은 두 점 (무조건 사용)
                lower = a if rooms[a]['floor'] <= rooms[b]['floor'] else b
                wps.append(pa); labels.append(f"{a} 계단점 도착"); plans.append({'kind': 'rrt'})
                wps.append(pb); labels.append(f"{a}→{b} 계단 통과(점 사용)")
                plans.append({'kind': 'grid', 'lower': lower})
                continue
            wps.append(list(e['door_center'])); labels.append(f"door({a}-{b}) stairs")
            plans.append({'kind': 'rrt'})
        else:
            pts = door_points(rooms[a], rooms[b], e)   # 직접 찍은 통로 점만 경유
            for j, p in enumerate(pts):
                wps.append(p)
                labels.append(f"door({a}-{b}) 통로점{j+1}" if len(pts) > 1 else f"door({a}-{b})")
                plans.append({'kind': 'rrt'})
    wps.append(list(rooms[goal_room]['center'])); labels.append(f"{goal_room} 목표")
    plans.append({'kind': 'rrt'})

    print("\n[하위] 구간별 경로계획:")
    segments = []
    for k in range(len(wps) - 1):
        pl = plans[k]
        if pl['kind'] == 'grid':                 # 계단: 찍은 두 점 사이 격자 A*
            lr = rooms[pl['lower']]
            pa, pb = np.array(wps[k]), np.array(wps[k + 1])
            mxy = 1.5                              # 통과 루트가 방 footprint 밖으로 새는 경우 대비
            lo = [min(lr['bbox_min'][0], pa[0], pb[0]) - mxy,
                  min(lr['bbox_min'][1], pa[1], pb[1]) - mxy,
                  min(pa[2], pb[2]) - 0.6]
            hi = [max(lr['bbox_max'][0], pa[0], pb[0]) + mxy,
                  max(lr['bbox_max'][1], pa[1], pb[1]) + mxy,
                  max(pa[2], pb[2]) + 0.6]
            gp = grid_astar(tree, pa, pb, lo, hi, stair_radius, max_expand=500000)
            if gp is not None and len(gp) >= 2:
                path = R.smooth_path(gp, tree, obs_radius=stair_radius)
                ok = True
            else:
                path = np.array([wps[k], wps[k + 1]]); ok = False
        else:                                    # 일반 구간: 직선 우선, 막히면 RRT*
            a = np.array(wps[k], float); b = np.array(wps[k + 1], float)
            if R.is_edge_free(a, b, tree, radius=R.OBSTACLE_RADIUS):
                path = np.array([a, b]); ok = True       # 두 통로점 사이가 트였으면 직선
            else:
                path = plan_segment(tree, gbounds, wps[k], wps[k + 1], max_iter, pad=2.5)
                ok = path is not None
                if ok:
                    path = np.asarray(path, float)
                    path[0] = a; path[-1] = b             # 끝점은 정확히 찍은 통로점으로
                else:
                    path = np.array([a, b])
        segments.append({'path': np.asarray(path),
                         'type': 'stairs' if pl['kind'] == 'grid' else 'seg',
                         'label': labels[k]})
        tag = 'OK' if ok else ('격자 경로없음' if pl['kind'] == 'grid' else '직선대체')
        print(f"  → {labels[k]} {np.round(wps[k+1],2)}  {tag} ({len(path)}pt)")

    # 전체 경로 = 각 구간을 순서대로 이어붙임.
    # ★ 구간을 가로지르는 전역 스무딩은 하지 않는다 — 그렇게 하면 line-of-sight 단축이
    #   내가 찍은 통로점을 지름길로 잘라내 버린다. 각 구간은 이미 시작/끝이 통로점이고
    #   내부만 (직선 or RRT*) 이라, 찍은 점들이 고정 경유점으로 그대로 보존된다.
    full = []
    for s in segments:
        for p in s['path']:
            full.append([float(p[0]), float(p[1]), float(p[2])])
    # 연속 중복 제거 (구간 경계에서 같은 통로점이 두 번 들어감)
    fp = [full[0]]
    for p in full[1:]:
        if np.linalg.norm(np.array(p) - np.array(fp[-1])) > 1e-6:
            fp.append(p)
    full_path = np.array(fp)

    total = float(np.linalg.norm(np.diff(full_path, axis=0), axis=1).sum())
    print(f"\n총 경로 길이 ≈ {total:.2f} m (스무딩 후 {len(full_path)}점)")
    return g, seq, segments, full_path


# ──────────────────────────────────────────
# 3D 맵에서 경로 시각화
# ──────────────────────────────────────────
_FLOOR_COL = {0: [0.40, 0.55, 0.90], 1: [0.40, 0.75, 0.50], 2: [0.90, 0.62, 0.35]}


def _sphere(c, r, col):
    import open3d as o3d
    m = o3d.geometry.TriangleMesh.create_sphere(radius=r, resolution=12)
    m.translate(np.asarray(c, float)); m.paint_uniform_color(col); m.compute_vertex_normals()
    return m


def visualize_3d(g, seq, segments, full_path, voxel=0.06):
    import open3d as o3d
    rooms = g['rooms']; ids = list(rooms)
    onpath = set(seq)

    geoms = []
    for rid in ids:
        coord = np.load(os.path.join(_NPY, rooms[rid]['npy'], 'coord.npy')).astype(np.float64)
        coord = voxel_down(coord, max(voxel, 0.06))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coord)
        col = _FLOOR_COL.get(rooms[rid]['floor'], [0.6, 0.6, 0.6]) if rid in onpath else [0.55, 0.55, 0.58]
        pcd.paint_uniform_color(col)
        geoms.append(pcd)

    allpts = np.asarray(full_path)        # 스무딩된 최종 경로

    # 경로 라인 + 점(굵게 보이도록 작은 구)
    if len(allpts) >= 2:
        lines = [[i, i + 1] for i in range(len(allpts) - 1)]
        ls = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(allpts),
            lines=o3d.utility.Vector2iVector(np.asarray(lines)))
        ls.colors = o3d.utility.Vector3dVector(np.tile([1.0, 0.0, 0.0], (len(lines), 1)))
        geoms.append(ls)
        # 경로 점은 0.4m 간격으로만 찍어 깔끔하게 (촘촘한 격자점 뭉침 방지)
        last = allpts[0]; geoms.append(_sphere(last, 0.05, [1.0, 0.0, 0.0]))
        for p in allpts[1:]:
            if np.linalg.norm(p - last) >= 0.4:
                geoms.append(_sphere(p, 0.05, [1.0, 0.0, 0.0])); last = p

    # 마커: 시작(초록), 목표(빨강), 직접 찍은 계단점만 주황 (자동 문 중심은 표시 안 함)
    geoms.append(_sphere(allpts[0], 0.24, [0.1, 0.9, 0.1]))
    geoms.append(_sphere(allpts[-1], 0.24, [0.9, 0.1, 0.1]))
    for s in segments:
        if s['type'] == 'stairs':
            geoms.append(_sphere(s['path'][0], 0.18, [1.0, 0.55, 0.0]))
            geoms.append(_sphere(s['path'][-1], 0.18, [1.0, 0.55, 0.0]))

    print(f"\n[3D] 경로 {seq[0]} → {seq[-1]}  (빨강=경로, 초록=시작, 빨강=목표, 주황=내가 찍은 계단점)")
    print("  마우스: 회전/줌/이동.  창 닫으면 종료.")
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"경로 {seq[0]} -> {seq[-1]}  ({' -> '.join(seq)})",
                      width=1400, height=900)
    for ge in geoms:
        vis.add_geometry(ge)
    opt = vis.get_render_option()
    opt.background_color = np.array([0.09, 0.09, 0.11]); opt.point_size = 2.0
    vis.run()
    vis.destroy_window()


# ──────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('start', nargs='?', help='시작 방 id (예: 002)')
    ap.add_argument('goal',  nargs='?', help='목표 방 id (예: 019)')
    ap.add_argument('--max-iter', type=int, default=6000)
    ap.add_argument('--voxel', type=float, default=0.05, help='장애물 다운샘플(m), 0이면 원본')
    ap.add_argument('--stair-radius', type=float, default=0.12,
                    help='계단 통과 시 드론 여유반경(m). 기본 0.12 (좁은 계단 통과용)')
    ap.add_argument('--list', action='store_true', help='방 목록만 출력')
    args = ap.parse_args()

    g = load_graph()
    if args.list or not args.start or not args.goal:
        print("방 목록:")
        for rid, r in g['rooms'].items():
            print(f"  {rid}  floor {r['floor']}  label={r['label']}")
        if not (args.start and args.goal):
            print("\n사용: python hierarchical_plan.py <시작방> <목표방>   예) 002 019")
        return

    res = plan(args.start, args.goal, max_iter=args.max_iter, voxel=args.voxel,
               stair_radius=args.stair_radius)
    if res is None:
        return
    g, seq, segments, full_path = res
    print("\n[시각화] 3D 맵에서 경로를 표시합니다...")
    visualize_3d(g, seq, segments, full_path, voxel=args.voxel)


if __name__ == '__main__':
    main()
