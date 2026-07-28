"""
inspect_graph.py
────────────────
rooms_graph.json 의 매핑 결과를 Open3D 로 시각 검증한다.

  · 방 점군 = 방마다 다른 색 (segmentation 확인용). --rgb 면 원본 색.
  · 문(edge)   = door_center 에 빨간 구  +  두 방 중심을 잇는 초록 선.
                 → 빨간 구가 실제 벽 틈(통로)에 놓였는지 눈으로 확인.
  · 계단 후보  = hint_center 에 노란 구  +  주황 점선 느낌의 선 (verified=False).
  · 방 id / 문 폭 = 3D 라벨로 표시.

사용:
  python inspect_graph.py                    # 전체, 층별 계열색(차분)
  python inspect_graph.py --color-by room    # 방별 파스텔색
  python inspect_graph.py --color-by rgb     # 방 원본 색
  python inspect_graph.py --floor 2          # 2층만
  python inspect_graph.py --no-interfloor    # 계단 후보 숨김
  python inspect_graph.py --simple           # 라벨 없는 호환 뷰어(draw_geometries)
  python inspect_graph.py --dry              # 창 없이 구성만 검증
"""

import os
import json
import argparse
import colorsys
import numpy as np
import open3d as o3d

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_NPY_DIR  = os.path.join(_BASE_DIR, 'npy')
_GRAPH    = os.path.join(_BASE_DIR, 'rooms_graph.json')


# ──────────────────────────────────────────
# 유틸
# ──────────────────────────────────────────
# 층별 기준 색상(hue). 같은 층은 한 계열, 방은 밝기로만 구분 → 덜 어지럽다.
_FLOOR_HUE = {0: 0.58, 1: 0.33, 2: 0.08}   # 0:파랑계열 1:초록계열 2:주황계열


def floor_colors(rooms, ids):
    """방을 '층 계열색 + 방별 밝기'로 칠한다 (차분)."""
    by_floor = {}
    for rid in ids:
        by_floor.setdefault(rooms[rid]['floor'], []).append(rid)
    cmap = {}
    for fl, members in by_floor.items():
        hue = _FLOOR_HUE.get(fl, (fl * 0.27) % 1.0)
        m = max(len(members) - 1, 1)
        for k, rid in enumerate(members):
            v = 0.62 + 0.30 * (k / m)          # 밝기 0.62~0.92
            cmap[rid] = colorsys.hsv_to_rgb(hue, 0.42, v)
    return cmap


def room_colors(ids):
    """방마다 다른 파스텔색 (채도 낮춰 덜 자극적)."""
    cmap = {}
    for i, rid in enumerate(ids):
        h = (i * 0.61803398875) % 1.0
        cmap[rid] = colorsys.hsv_to_rgb(h, 0.40, 0.90)
    return cmap


def sphere(center, r, color):
    m = o3d.geometry.TriangleMesh.create_sphere(radius=r, resolution=12)
    m.translate(np.asarray(center, float))
    m.paint_uniform_color(color)
    m.compute_vertex_normals()
    return m


def line_set(segments, color):
    """segments: [(p0,p1), ...] → LineSet."""
    pts, lines = [], []
    for p0, p1 in segments:
        i = len(pts)
        pts.append(p0); pts.append(p1)
        lines.append([i, i + 1])
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.asarray(pts, float)),
        lines=o3d.utility.Vector2iVector(np.asarray(lines, int)),
    )
    ls.colors = o3d.utility.Vector3dVector(np.tile(color, (len(lines), 1)))
    return ls


# ──────────────────────────────────────────
# 지오메트리 구성
# ──────────────────────────────────────────
def build(graph, args):
    rooms = graph['rooms']
    ids   = [r for r in rooms if args.floor is None or rooms[r]['floor'] == args.floor]
    cmap  = room_colors(ids) if args.color_by == 'room' else floor_colors(rooms, ids)

    room_geoms = []   # (name, geometry)
    labels     = []   # (pos, text)

    print(f"[로드] 방 {len(ids)}개  (voxel={args.voxel}m, color_by={args.color_by})")
    for rid in ids:
        r = rooms[rid]
        coord = np.load(os.path.join(_NPY_DIR, r['npy'], 'coord.npy')).astype(np.float64)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coord)
        if args.color_by == 'rgb':
            col = np.load(os.path.join(_NPY_DIR, r['npy'], 'color.npy')).astype(np.float64) / 255.0
            pcd.colors = o3d.utility.Vector3dVector(col)
        else:
            pcd.paint_uniform_color(cmap[rid])
        if args.voxel > 0:
            pcd = pcd.voxel_down_sample(args.voxel)
        name = f"f{r['floor']}/room_{rid}" + ("" if r['label'] is None else f"_{r['label']}")
        room_geoms.append((name, pcd))
        labels.append((r['center'], f"{rid}" + ("" if r['label'] is None else f":{r['label']}")))
        print(f"  room {rid}  floor {r['floor']}  pts {len(pcd.points):>7,}")

    # 문 (verified door)
    door_geoms, door_segs = [], []
    for e in graph['edges']:
        if args.floor is not None and e.get('floor') != args.floor:
            continue
        if e['a'] not in rooms or e['b'] not in rooms:
            continue
        dc = e['door_center']
        door_geoms.append((f"door_{e['a']}-{e['b']}", sphere(dc, 0.13, [0.95, 0.1, 0.1])))
        labels.append((dc, f"{e['a']}-{e['b']} {e['door_width']}m"))
        door_segs.append((rooms[e['a']]['center'], dc))
        door_segs.append((dc, rooms[e['b']]['center']))

    # 계단 후보 (interfloor, unverified)
    stair_geoms, stair_segs = [], []
    if not args.no_interfloor and args.floor is None:
        for c in graph.get('interfloor_candidates', []):
            hc = c['hint_center']
            stair_geoms.append((f"stair?_{c['a']}-{c['b']}", sphere(hc, 0.16, [0.95, 0.85, 0.1])))
            stair_segs.append((rooms[c['a']]['center'], rooms[c['b']]['center']))

    line_geoms = []
    if door_segs:
        line_geoms.append(("door_links", line_set(door_segs, [0.1, 0.9, 0.2])))
    if stair_segs:
        line_geoms.append(("stair_links", line_set(stair_segs, [0.95, 0.6, 0.05])))

    print(f"[구성] 방 {len(room_geoms)} | 문 {len(door_geoms)} | 계단후보 {len(stair_geoms)} | 라벨 {len(labels)}")
    return room_geoms, door_geoms, stair_geoms, line_geoms, labels


# ──────────────────────────────────────────
# 렌더링
# ──────────────────────────────────────────
def render_rich(geoms_groups, labels, point_size):
    """O3DVisualizer: 이름 토글 패널 + 3D 라벨."""
    import open3d.visualization.gui as gui
    room_g, door_g, stair_g, line_g, _ = geoms_groups
    app = gui.Application.instance
    app.initialize()
    vis = o3d.visualization.O3DVisualizer("Room Graph 검증", 1400, 950)
    vis.show_settings = True
    for name, g in room_g + door_g + stair_g + line_g:
        vis.add_geometry(name, g)
    for pos, text in labels:
        vis.add_3d_label(np.asarray(pos, float), text)
    vis.point_size = int(round(point_size))   # 정수만 허용
    try:
        vis.set_background([0.10, 0.10, 0.12, 1.0], None)   # 어두운 배경 → 대비↑
    except Exception:
        pass
    vis.reset_camera_to_default()
    app.add_window(vis)
    app.run()


def render_simple(geoms_groups, point_size):
    """호환 뷰어: draw_geometries (라벨 없음)."""
    room_g, door_g, stair_g, line_g, _ = geoms_groups
    geos = [g for _, g in room_g + door_g + stair_g + line_g]
    o3d.visualization.draw_geometries(
        geos, window_name="Room Graph 검증 (simple)",
        width=1400, height=950, point_show_normal=False)


# ──────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--voxel', type=float, default=0.06, help='다운샘플 voxel 크기(m), 0이면 원본')
    ap.add_argument('--color-by', choices=['floor', 'room', 'rgb'], default='floor',
                    help='floor=층별 계열색(기본, 차분) | room=방별 파스텔 | rgb=원본색')
    ap.add_argument('--floor', type=int, default=None, help='특정 층만 표시')
    ap.add_argument('--no-interfloor', action='store_true', help='계단 후보 숨김')
    ap.add_argument('--simple', action='store_true', help='draw_geometries 호환 뷰어')
    ap.add_argument('--point-size', type=float, default=3.0)
    ap.add_argument('--dry', action='store_true', help='창 없이 구성만 검증')
    args = ap.parse_args()

    with open(_GRAPH, encoding='utf-8') as f:
        graph = json.load(f)

    room_g, door_g, stair_g, line_g, labels = build(graph, args)
    groups = (room_g, door_g, stair_g, line_g, labels)

    if args.dry:
        print("[dry] 구성 검증 완료 — 창을 띄우지 않음.")
        return

    if args.simple:
        render_simple(groups, args.point_size)
    else:
        try:
            render_rich(groups, labels, args.point_size)
        except Exception as e:
            print(f"[경고] O3DVisualizer 실패 ({e}) → simple 뷰어로 대체")
            render_simple(groups, args.point_size)


if __name__ == '__main__':
    main()
