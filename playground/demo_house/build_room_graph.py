"""
build_room_graph.py
───────────────────
demo_house/npy 의 Matterport region(방) npy 들로부터 계층적 경로계획용 room graph 생성.

핵심: edge(문/통로) 좌표는 단순 bbox 겹침이 아니라
  1) 같은 층 + 경계가 가까운 방 쌍에 대해
  2) 두 방 경계면의 navigable 높이대(floor+0.4~1.6m)에서 드론 반경만큼 비어있는(free) 점을 찾고
  3) 그 점이 벽을 실제로 관통(crossing)하는지 검증한 뒤
  4) 가장 넓은 free 통로(클러스터)의 중심을 door_center 로 찍는다.
→ door_center 는 항상 드론이 실제로 통과할 수 있는 free 공간 위에 위치한다.

층간(계단) 연결은 기하학적으로 모호하므로 '후보'로만 내보내고 verified=False 로 둔다.
(뷰어 단계에서 수동 확정)

출력: demo_house/rooms_graph.json
"""

import os
import json
import numpy as np
from scipy.spatial import cKDTree

# ──────────────────────────────────────────
# 파라미터 (rrt_star_drone.py 와 동일 기준)
# ──────────────────────────────────────────
DRONE_HALF_SIZE = 0.10
DRONE_RADIUS    = DRONE_HALF_SIZE * np.sqrt(2) * 1.3   # ~0.185 m
SAFETY_MARGIN   = 0.05
OBSTACLE_RADIUS = DRONE_RADIUS + SAFETY_MARGIN          # ~0.235 m  → 문은 폭 0.47m↑ 필요

NAV_LOW, NAV_HIGH = 0.40, 1.60   # 바닥 기준 navigable 높이대 (m)
NAV_STEP          = 0.15         # 높이 샘플 간격
GRID_STEP         = 0.10         # 경계면 가로 샘플 간격
WALL_SLAB         = 0.10         # 벽 두께 대응: 경계 ± 이 값 평면들 샘플
CROSS_H           = 0.50         # 벽 관통 확인용 편도 거리 (총 1.0m 가 free 여야 통로)
CROSS_N           = 5            # 관통 세그먼트 샘플 수
MIN_DOOR_WIDTH    = 0.45         # 이보다 좁은 free 클러스터는 노이즈로 폐기
MAX_PAIR_GAP      = 0.60         # 경계 간격이 이보다 크면 인접 아님

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_NPY_DIR  = os.path.join(_BASE_DIR, 'npy')


# ──────────────────────────────────────────
# 방 메타 로드
# ──────────────────────────────────────────
def load_rooms():
    rooms = {}
    for folder in sorted(os.listdir(_NPY_DIR)):
        cpath = os.path.join(_NPY_DIR, folder, 'coord.npy')
        if not os.path.exists(cpath):
            continue
        coord = np.load(cpath).astype(np.float32)
        parts = folder.split('_')          # [00809, Qpor2mEya8F, 000, 002]
        rid   = parts[-1]                  # '002' (층 넘어도 고유)
        floor = int(parts[-2])
        rooms[rid] = {
            'npy':      folder,
            'floor':    floor,
            'bbox_min': coord.min(0),
            'bbox_max': coord.max(0),
            'center':   ((coord.min(0) + coord.max(0)) / 2.0),
            'n_points': len(coord),
            'tree':     cKDTree(coord),     # 충돌검사용 (저장 X)
            'label':    None,
        }
        print(f"  loaded {rid:>3}  floor {floor}  pts {len(coord):>9,}")
    return rooms


# ──────────────────────────────────────────
# free / 관통 검사 (두 방 트리 OR)
# ──────────────────────────────────────────
def is_free(p, treeA, treeB, r=OBSTACLE_RADIUS):
    return (not treeA.query_ball_point(p, r)) and (not treeB.query_ball_point(p, r))


def crossing_free(center, axis, treeA, treeB):
    """door 후보가 sep축 벽을 실제로 관통하는지: center 양쪽 ±CROSS_H 가 모두 free."""
    d = np.zeros(3); d[axis] = 1.0
    for t in np.linspace(-CROSS_H, CROSS_H, CROSS_N):
        if not is_free(center + t * d, treeA, treeB):
            return False
    return True


# ──────────────────────────────────────────
# 두 방 사이 문/통로 검출
# ──────────────────────────────────────────
def find_door(A, B):
    amn, amx = A['bbox_min'], A['bbox_max']
    bmn, bmx = B['bbox_min'], B['bbox_max']

    # 축별 겹침( >0 겹침, <0 간격 )
    lo = np.maximum(amn, bmn)
    hi = np.minimum(amx, bmx)
    overlap = hi - lo                       # [ox, oy, oz]

    sep_axis  = int(np.argmin(overlap[:2])) # 더 빡빡한 축 = 벽이 그 축에 수직
    span_axis = 1 - sep_axis

    if overlap[sep_axis] < -MAX_PAIR_GAP:   # 너무 멀다
        return None
    if overlap[span_axis] < MIN_DOOR_WIDTH: # 마주보는 폭이 없다 (대각선)
        return None

    span_lo, span_hi = lo[span_axis], hi[span_axis]
    floor_z    = float(max(amn[2], bmn[2]))                   # 두 방 공통 바닥
    z_lo, z_hi = floor_z + NAV_LOW, floor_z + NAV_HIGH

    # sep축 스캔 범위: 벽 위치를 가정하지 않고 경계 구간 전체를 훑는다
    # (겹치면 그 구간, 떨어져 있으면 그 사이 간격)
    sep_a, sep_b = sorted([lo[sep_axis], hi[sep_axis]])
    sep_lo = sep_a - WALL_SLAB
    sep_hi = sep_b + WALL_SLAB

    treeA, treeB = A['tree'], B['tree']

    # (가로 × 높이) 각 칸에서 sep축을 훑어 free + 벽 관통(crossing)하는 통로를 찾는다
    cand = []
    spans = np.arange(span_lo, span_hi + 1e-6, GRID_STEP)
    zs    = np.arange(z_lo, z_hi + 1e-6, NAV_STEP)
    seps  = np.arange(sep_lo, sep_hi + 1e-6, GRID_STEP)
    for s in spans:
        for z in zs:
            for sep in seps:
                p = np.empty(3)
                p[sep_axis]  = sep
                p[span_axis] = s
                p[2]         = z
                if is_free(p, treeA, treeB) and crossing_free(p, sep_axis, treeA, treeB):
                    cand.append((s, z, sep))   # 실제 통로가 뚫린 sep 위치 기록
                    break                      # 이 (span,z) 는 통로로 확정
    if not cand:
        return None

    cand = np.array(cand)   # (N, [span, z, sep])

    # (span,z) 평면에서 연결요소 클러스터링 → 가장 넓은 통로 선택
    pts2d = cand[:, :2]
    ctree = cKDTree(pts2d)
    n = len(pts2d)
    seen = np.zeros(n, bool)
    best = None
    link = GRID_STEP * 1.8
    for i in range(n):
        if seen[i]:
            continue
        # BFS
        stack, comp = [i], []
        seen[i] = True
        while stack:
            k = stack.pop()
            comp.append(k)
            for j in ctree.query_ball_point(pts2d[k], link):
                if not seen[j]:
                    seen[j] = True
                    stack.append(j)
        comp = np.array(comp)
        width = pts2d[comp, 0].max() - pts2d[comp, 0].min()
        if width < MIN_DOOR_WIDTH:
            continue
        if best is None or len(comp) > len(best['comp']):
            best = {'comp': comp, 'width': width}

    if best is None:
        return None

    comp = best['comp']
    # door_center = 클러스터 중심에 가장 가까운 실제 검증된 후보 (free + 관통 보장)
    centroid = cand[comp].mean(0)
    k = comp[np.argmin(np.linalg.norm(cand[comp, :2] - centroid[:2], axis=1))]
    s, z, sep = cand[k]
    dc = np.empty(3)
    dc[sep_axis]  = sep            # 실제 통로가 뚫린 위치
    dc[span_axis] = s
    dc[2]         = z
    return {
        'door_center': [round(float(v), 3) for v in dc],
        'door_width':  round(float(best['width']), 3),
        'height_band': [round(float(cand[comp, 1].min()), 2),
                        round(float(cand[comp, 1].max()), 2)],
        'sep_axis':    'xyz'[sep_axis],
    }


# ──────────────────────────────────────────
# 층간(계단) 연결 후보 — XY 겹치는 인접 층 방 쌍
# ──────────────────────────────────────────
def interfloor_candidates(rooms):
    out = []
    ids = list(rooms)
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            A, B = rooms[ids[i]], rooms[ids[j]]
            if abs(A['floor'] - B['floor']) != 1:
                continue
            lo = np.maximum(A['bbox_min'][:2], B['bbox_min'][:2])
            hi = np.minimum(A['bbox_max'][:2], B['bbox_max'][:2])
            ov = hi - lo
            if ov[0] <= 0.5 or ov[1] <= 0.5:
                continue
            area = float(ov[0] * ov[1])
            cx, cy = (lo + hi) / 2.0
            zmid = (min(A['bbox_max'][2], B['bbox_max'][2]) +
                    max(A['bbox_min'][2], B['bbox_min'][2])) / 2.0
            out.append({
                'a': ids[i], 'b': ids[j], 'type': 'stairs',
                'xy_overlap_area': round(area, 2),
                'hint_center': [round(float(cx), 3), round(float(cy), 3), round(float(zmid), 3)],
                'verified': False,
            })
    out.sort(key=lambda e: -e['xy_overlap_area'])
    return out


# ──────────────────────────────────────────
# 메인
# ──────────────────────────────────────────
def main():
    print("[1/3] 방 로딩 + KDTree 구성...")
    rooms = load_rooms()
    ids = list(rooms)

    print("\n[2/3] 같은 층 문/통로 검출...")
    edges = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            A, B = rooms[ids[i]], rooms[ids[j]]
            if A['floor'] != B['floor']:
                continue
            # 경계가 가까운 쌍만
            gap = np.maximum(A['bbox_min'] - B['bbox_max'],
                             B['bbox_min'] - A['bbox_max'])[:2].max()
            if gap > MAX_PAIR_GAP:
                continue
            door = find_door(A, B)
            if door:
                edges.append({'a': ids[i], 'b': ids[j], 'type': 'door',
                              'floor': A['floor'], 'verified': True, **door})
                print(f"  door  {ids[i]:>3} <-> {ids[j]:>3}  "
                      f"@ {door['door_center']}  width={door['door_width']}m")

    print("\n[3/3] 층간(계단) 후보...")
    stairs = interfloor_candidates(rooms)
    for s in stairs:
        print(f"  stair?{s['a']:>3} <-> {s['b']:>3}  overlap={s['xy_overlap_area']}m^2  (미검증)")

    # 직렬화 (tree 제거)
    rooms_out = {}
    for rid, r in rooms.items():
        rooms_out[rid] = {
            'npy':      r['npy'],
            'floor':    r['floor'],
            'bbox_min': [round(float(v), 3) for v in r['bbox_min']],
            'bbox_max': [round(float(v), 3) for v in r['bbox_max']],
            'center':   [round(float(v), 3) for v in r['center']],
            'n_points': r['n_points'],
            'label':    r['label'],
        }

    graph = {
        'scene':  rooms[ids[0]]['npy'].rsplit('_', 2)[0],
        'params': {
            'obstacle_radius': round(OBSTACLE_RADIUS, 3),
            'navigable_band':  [NAV_LOW, NAV_HIGH],
            'min_door_width':  MIN_DOOR_WIDTH,
        },
        'rooms':  rooms_out,
        'edges':  edges,
        'interfloor_candidates': stairs,
    }

    out_path = os.path.join(_BASE_DIR, 'rooms_graph.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*55}")
    print(f"  방 {len(rooms_out)}개 | 문 edge {len(edges)}개 | 계단 후보 {len(stairs)}개")
    print(f"  저장: {out_path}")
    print(f"{'='*55}")


if __name__ == '__main__':
    main()
