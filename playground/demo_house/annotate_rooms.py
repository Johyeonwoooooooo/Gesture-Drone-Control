"""
annotate_rooms.py
─────────────────
방을 직접 보면서 수동으로 라벨/연결을 정하는 도구.

순서:
  [1] 각 방마다 3D 창(그 방 npy)을 띄워 WASD 로 날아다니며 문/통로에 점을 찍는다.
         · W/S 전후 · A/D 좌우 · Space/C 상하 · 마우스 드래그=시점
         · E = 현재 위치에 통로 점 추가 (여러 개 가능) · R = 마지막 점 삭제
       동시에 별도 2D 평면도 창에 '내 위치(초록)'와 '찍은 점(빨강 X)' 이 실시간 표시되고,
       다른 방이 이미 찍은 점은 파란 원으로 보여 문을 맞추기 쉽다.
       3D 창을 닫으면 → 콘솔에서 label 입력 → JSON 즉시 저장.
  [2] 전체 통합 뷰 — 방=층별색, 빨간 구=문/연결, 초록 선=연결관계.
       두 방의 통로 점이 0.9m 안으로 가까우면 같은 문으로 보고 자동 연결.

실행:
  python annotate_rooms.py                 # 전체 (방별 평면도 클릭 → 통합)
  python annotate_rooms.py --only 011,013  # 지정 방만
  python annotate_rooms.py --start 011     # 특정 방부터
  python annotate_rooms.py --review        # 통합 결과만 보기
  python annotate_rooms.py --fresh         # 기존 json 무시하고 빈 상태로
"""

import os
import sys
import json
import argparse
import colorsys
import numpy as np
import open3d as o3d

# 콘솔 한글 깨짐 방지 (Windows cp949 → utf-8)
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_NPY_DIR  = os.path.join(_BASE_DIR, 'npy')
_GRAPH    = os.path.join(_BASE_DIR, 'rooms_graph.json')

_FLOOR_HUE = {0: 0.58, 1: 0.33, 2: 0.08}
_GUI_INIT  = False
_KO_FONT_OK = None   # Open3D HUD 한글 폰트 사용 가능 여부

# 시스템에 있는 한글 폰트 경로 (matplotlib / Open3D 공용)
_KO_FONT_PATHS = [
    r"C:\Windows\Fonts\malgun.ttf",      # 맑은 고딕
    r"C:\Windows\Fonts\malgunsl.ttf",
    r"C:\Windows\Fonts\NanumGothic.ttf",
    r"C:\Windows\Fonts\gulim.ttc",
]


def _find_ko_font():
    for p in _KO_FONT_PATHS:
        if os.path.exists(p):
            return p
    return None


# ──────────────────────────────────────────
# 방 메타 / 색
# ──────────────────────────────────────────
def scan_rooms():
    rooms = {}
    for folder in sorted(os.listdir(_NPY_DIR)):
        cpath = os.path.join(_NPY_DIR, folder, 'coord.npy')
        if not os.path.exists(cpath):
            continue
        coord = np.load(cpath).astype(np.float64)
        rid = folder.split('_')[-1]
        rooms[rid] = {
            'npy':      folder,
            'floor':    int(folder.split('_')[-2]),
            'bbox_min': [round(float(v), 3) for v in coord.min(0)],
            'bbox_max': [round(float(v), 3) for v in coord.max(0)],
            'center':   [round(float(v), 3) for v in (coord.min(0) + coord.max(0)) / 2],
            'n_points': len(coord),
            'label':    None,
            'passages': [],   # 직접 찍은 통로 점들 [[x,y,z], ...]
        }
    return rooms


def floor_colormap(rooms, ids):
    by_floor = {}
    for rid in ids:
        by_floor.setdefault(rooms[rid]['floor'], []).append(rid)
    cmap = {}
    for fl, mem in by_floor.items():
        hue = _FLOOR_HUE.get(fl, (fl * 0.27) % 1.0)
        m = max(len(mem) - 1, 1)
        for k, rid in enumerate(mem):
            cmap[rid] = colorsys.hsv_to_rgb(hue, 0.45, 0.62 + 0.30 * (k / m))
    return cmap


def load_graph(rooms):
    conns = {rid: set() for rid in rooms}
    if os.path.exists(_GRAPH):
        try:
            g = json.load(open(_GRAPH, encoding='utf-8'))
            for rid, r in g.get('rooms', {}).items():
                if rid in rooms and r.get('label'):
                    rooms[rid]['label'] = r['label']
                if rid in rooms and r.get('passages'):
                    rooms[rid]['passages'] = [list(map(float, p)) for p in r['passages']]
                # JSON 에 저장된 center(수동 이동분)를 유지 — 없으면 bbox 중심 그대로
                if rid in rooms and r.get('center'):
                    rooms[rid]['center'] = [float(v) for v in r['center']]
            for e in g.get('edges', []):
                a, b = e.get('a'), e.get('b')
                if a in conns and b in conns:
                    conns[a].add(b); conns[b].add(a)
        except Exception as ex:
            print(f"[경고] 기존 json 읽기 실패: {ex}")
    return conns


# ──────────────────────────────────────────
# 문 힌트 / edge / 저장
# ──────────────────────────────────────────
def door_hint(A, B):
    out = []
    for i in (0, 1):
        lo, hi = max(A['bbox_min'][i], B['bbox_min'][i]), min(A['bbox_max'][i], B['bbox_max'][i])
        out.append((lo + hi) / 2 if hi > lo else (A['center'][i] + B['center'][i]) / 2)
    z = max(A['bbox_min'][2], B['bbox_min'][2]) + 1.0
    return [round(out[0], 3), round(out[1], 3), round(float(z), 3)]


_MERGE_DIST = 0.9   # 두 방의 통로 점이 이보다 가까우면 같은 문으로 보고 연결


def _dist(p, q):
    return float(np.linalg.norm(np.asarray(p, float) - np.asarray(q, float)))


def _nearest_passage_pair(A, B):
    """A,B 통로 점 중 가장 가까운 쌍 → (거리, 중점). 없으면 None."""
    best = None
    for x in A.get('passages', []):
        for y in B.get('passages', []):
            d = _dist(x, y)
            if best is None or d < best[0]:
                mid = [round((x[k] + y[k]) / 2, 3) for k in range(3)]
                best = (d, mid)
    return best


def _conn_door(rooms, a, b):
    """연결된 두 방의 door_center: 통로 점 우선, 없으면 bbox 힌트."""
    pair = _nearest_passage_pair(rooms[a], rooms[b])
    if pair and pair[0] < 2.5:
        return pair[1]
    # 한쪽에만 통로 점이 있으면 상대 방 중심에 가장 가까운 점
    cands = [(_dist(x, rooms[b]['center']), x) for x in rooms[a].get('passages', [])]
    cands += [(_dist(y, rooms[a]['center']), y) for y in rooms[b].get('passages', [])]
    if cands:
        cands.sort()
        return [round(float(v), 3) for v in cands[0][1]]
    return door_hint(rooms[a], rooms[b])


def build_edges(rooms, conns, auto_passages=True):
    """edge = (1) 통로 점이 서로 가까운 방쌍 자동연결(auto_passages)  +  (2) 수동연결(conns).
       수동 편집기에서는 auto_passages=False 로 사용자가 그은 것만 반영."""
    edges = {}

    def put(a, b, dc, src):
        key = tuple(sorted((a, b)))
        same = rooms[key[0]]['floor'] == rooms[key[1]]['floor']
        edges[key] = {'a': key[0], 'b': key[1],
                      'type': 'door' if same else 'stairs',
                      'door_center': dc, 'source': src}

    ids = list(rooms)
    # (1) 통로 점 근접 → 자동 연결
    if auto_passages:
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                pair = _nearest_passage_pair(rooms[ids[i]], rooms[ids[j]])
                if pair and pair[0] <= _MERGE_DIST:
                    put(ids[i], ids[j], pair[1], 'passage')

    # (2) 수동 연결 (이미 있으면 유지)
    for a in conns:
        for b in conns[a]:
            key = tuple(sorted((a, b)))
            if key in edges or key[0] not in rooms or key[1] not in rooms:
                continue
            put(key[0], key[1], _conn_door(rooms, key[0], key[1]), 'manual')

    return list(edges.values())


def save(rooms, conns, auto_passages=True):
    g = {
        'scene': next(iter(rooms.values()))['npy'].rsplit('_', 2)[0],
        'rooms': {rid: dict(r) for rid, r in rooms.items()},
        'edges': build_edges(rooms, conns, auto_passages=auto_passages),
    }
    json.dump(g, open(_GRAPH, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)


# ──────────────────────────────────────────
# [1] 층별 2D top view (matplotlib)
# ──────────────────────────────────────────
def floor_topviews(rooms, max_pts=40000):
    import matplotlib
    import matplotlib.pyplot as plt
    # 한글 폰트 등록 (□ 깨짐 방지)
    fp = _find_ko_font()
    if fp:
        try:
            from matplotlib import font_manager
            font_manager.fontManager.addfont(fp)
            matplotlib.rcParams['font.family'] = font_manager.FontProperties(fname=fp).get_name()
        except Exception:
            pass
    matplotlib.rcParams['axes.unicode_minus'] = False
    ids = list(rooms)
    cmap = floor_colormap(rooms, ids)
    floors = sorted({rooms[r]['floor'] for r in ids})

    print("\n[1단계] 층별 평면도. 각 창을 닫으면 다음 층으로 넘어갑니다.")
    for fl in floors:
        members = [r for r in ids if rooms[r]['floor'] == fl]
        fig, ax = plt.subplots(figsize=(11, 9))
        legend = []
        for rid in members:
            coord = np.load(os.path.join(_NPY_DIR, rooms[rid]['npy'], 'coord.npy'))
            if len(coord) > max_pts:
                coord = coord[np.random.choice(len(coord), max_pts, replace=False)]
            ax.scatter(coord[:, 0], coord[:, 1], s=1, c=[cmap[rid]], linewidths=0)
            cx, cy = rooms[rid]['center'][0], rooms[rid]['center'][1]
            ax.text(cx, cy, rid, fontsize=15, fontweight='bold', ha='center', va='center',
                    color='black', bbox=dict(boxstyle='round', fc='white', ec='none', alpha=0.7))
            legend.append(f"{rid} = {rooms[rid]['npy']}")
        ax.set_title(f"Floor {fl} — Top View (위에서 본 평면도)   [닫으면 다음]", fontsize=13)
        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
        ax.set_aspect('equal', 'box'); ax.grid(True, alpha=0.2)
        # 범례(어떤 npy 인지)
        ax.text(1.02, 1.0, "\n".join(legend), transform=ax.transAxes, va='top', ha='left',
                fontsize=8, family='monospace',
                bbox=dict(boxstyle='round', fc='#f5f5f5', ec='#cccccc'))
        plt.tight_layout()
        print(f"  Floor {fl}: 방 {len(members)}개 표시 — {', '.join(members)}")
        plt.show()


# ──────────────────────────────────────────
# 3D 비행 뷰 (gui FLY + 우상단 위치 HUD)
# ──────────────────────────────────────────
def _materials(point_size):
    import open3d.visualization.rendering as rendering
    solid = rendering.MaterialRecord(); solid.shader = 'defaultUnlit'; solid.point_size = float(point_size)
    line  = rendering.MaterialRecord(); line.shader = 'unlitLine';   line.line_width = 2.0
    return solid, line


def fly_view(geoms, title, info_text, point_size=3):
    """gui FLY 뷰어. 실패하면 legacy(콘솔 위치 출력) 로 대체."""
    try:
        _fly_view_gui(geoms, title, info_text, point_size)
    except Exception as ex:
        print(f"[경고] gui 뷰어 실패({ex}) → legacy WASD 뷰어로 대체 (위치는 콘솔 출력)")
        _fly_view_legacy(geoms, title, point_size)


def _ensure_gui():
    """gui Application 초기화 + 한글 폰트 등록 (한 번만)."""
    global _GUI_INIT, _KO_FONT_OK
    import open3d.visualization.gui as gui
    app = gui.Application.instance
    if _GUI_INIT:
        return app
    app.initialize()
    _KO_FONT_OK = False
    fp = _find_ko_font()
    if fp:
        try:
            fd = gui.FontDescription()
            fd.add_typeface_for_language(fp, "ko")
            app.set_font(gui.Application.DEFAULT_FONT_ID, fd)
            _KO_FONT_OK = True
        except Exception as ex:
            print(f"[안내] Open3D 한글 폰트 등록 실패 → HUD는 영문 표시 ({ex})")
    _GUI_INIT = True
    return app


def _fly_view_gui(geoms, title, info_text, point_size):
    import open3d.visualization.gui as gui
    import open3d.visualization.rendering as rendering

    app = _ensure_gui()
    win = app.create_window(title, 1400, 950)
    widget = gui.SceneWidget()
    widget.scene = rendering.Open3DScene(win.renderer)
    win.add_child(widget)

    solid, line = _materials(point_size)
    for i, g in enumerate(geoms):
        m = line if isinstance(g, o3d.geometry.LineSet) else solid
        widget.scene.add_geometry(f"g{i}", g, m)
    widget.scene.set_background([0.10, 0.10, 0.12, 1.0])

    bounds = widget.scene.bounding_box
    widget.setup_camera(60.0, bounds, bounds.get_center())
    widget.set_view_controls(gui.SceneWidget.Controls.FLY)

    pos_label = "내 위치" if _KO_FONT_OK else "pos"
    hud = gui.Label(info_text + f"\n{pos_label}: ...")
    try:
        hud.text_color = gui.Color(1.0, 1.0, 0.35)
    except Exception:
        pass
    win.add_child(hud)

    def on_layout(ctx):
        r = win.content_rect
        widget.frame = r
        w, h = 360, 96
        hud.frame = gui.Rect(r.get_right() - w - 12, r.y + 12, w, h)
    win.set_on_layout(on_layout)

    def on_tick():
        V = np.asarray(widget.scene.camera.get_view_matrix())
        R, t = V[:3, :3], V[:3, 3]
        eye = -R.T @ t
        hud.text = (f"{info_text}\n"
                    f"{pos_label}  x={eye[0]:+.2f}  y={eye[1]:+.2f}  z={eye[2]:+.2f}")
        return True
    win.set_on_tick_event(on_tick)

    app.run()   # 창 닫을 때까지 블로킹


def _fly_view_legacy(geoms, title, point_size):
    step = [0.25]
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=title, width=1400, height=950)
    for g in geoms:
        vis.add_geometry(g)
    opt = vis.get_render_option()
    opt.background_color = np.array([0.10, 0.10, 0.12]); opt.point_size = point_size

    def move(fwd, rgt, upv):
        def cb(v):
            ctr = v.get_view_control()
            cam = ctr.convert_to_pinhole_camera_parameters()
            ext = np.array(cam.extrinsic).copy()
            R, t = ext[:3, :3], ext[:3, 3]
            C = -R.T @ t
            C = C + (R[2] * fwd + R[0] * rgt + (-R[1]) * upv) * step[0]
            ext[:3, 3] = -R @ C
            cam.extrinsic = ext
            try:
                ctr.convert_from_pinhole_camera_parameters(cam, True)
            except TypeError:
                ctr.convert_from_pinhole_camera_parameters(cam)
            print(f"  내 위치 x={C[0]:+.2f} y={C[1]:+.2f} z={C[2]:+.2f}")
            return False
        return cb
    for k, mv in [('W', (1, 0, 0)), ('S', (-1, 0, 0)), ('A', (0, -1, 0)),
                  ('D', (0, 1, 0)), ('R', (0, 0, 1)), ('F', (0, 0, -1))]:
        vis.register_key_callback(ord(k), move(*mv))
    vis.run(); vis.destroy_window()


# ──────────────────────────────────────────
# 통로 점 찍기 뷰어 (WASD 이동 + E:점추가 / R:점제거 + 우상단 위치 HUD)
# ──────────────────────────────────────────
def fly_and_mark(pcd, title, info_text, existing, point_size=3):
    """방을 둘러보며 통로 점을 직접 찍는다. 찍은 점 리스트(월드좌표)를 반환.

    legacy 뷰어를 쓴다: gui 의 FLY 모드는 E/R 을 카메라 회전(롤)키로 폴링해서
    점 찍기/삭제와 충돌한다(키를 가로채도 회전됨). legacy 는 E/R/WASD 에 내장
    카메라 동작이 없어, 마우스=시점(내장)·WASD=이동(직접)·E/R=점 추가/삭제 만 깔끔하게 동작.
    (화면 HUD 대신 위치는 터미널에 출력)
    """
    return _mark_legacy(pcd, title, existing, point_size)


def _mark_gui(pcd, title, info_text, existing, point_size):
    # 네비게이션은 Open3D 내장 FLY 컨트롤러 하나만 사용 (마우스=시점, WASD=이동).
    # 커스텀 카메라를 두지 않으므로 마우스+WASD 가 절대 충돌/점프하지 않는다.
    # 우리는 E(점 추가)/R(점 제거) 키만 가로챈다.
    import open3d.visualization.gui as gui
    import open3d.visualization.rendering as rendering

    app = _ensure_gui()
    win = app.create_window(title, 1400, 950)
    sw = gui.SceneWidget()
    sw.scene = rendering.Open3DScene(win.renderer)
    win.add_child(sw)

    solid, _ = _materials(point_size)
    sw.scene.add_geometry("room", pcd, solid)
    sw.scene.set_background([0.10, 0.10, 0.12, 1.0])
    bb = sw.scene.bounding_box
    sw.setup_camera(60.0, bb, bb.get_center())
    sw.set_view_controls(gui.SceneWidget.Controls.FLY)   # ← 핵심: 내장 FLY 네비게이션

    mark_mat = rendering.MaterialRecord(); mark_mat.shader = 'defaultUnlit'
    markers = []   # [name, pos(np3)]
    cnt = [0]

    def add_marker(pos):
        cnt[0] += 1
        name = f"mark_{cnt[0]}"
        s = o3d.geometry.TriangleMesh.create_sphere(0.12, resolution=12)
        s.translate(np.asarray(pos, float)); s.paint_uniform_color([1.0, 0.12, 0.12])
        s.compute_vertex_normals()
        sw.scene.add_geometry(name, s, mark_mat)
        markers.append([name, np.asarray(pos, float)])

    for p in existing:
        add_marker(np.asarray(p, float))

    def cam_eye_fwd():
        """현재 카메라 위치(eye)와 전방 벡터(forward)를 view matrix 에서 읽는다."""
        V = np.asarray(sw.scene.camera.get_view_matrix())
        R, t = V[:3, :3], V[:3, 3]
        eye = -R.T @ t          # 월드상 카메라 위치
        fwd = -R[2]             # 렌더링(OpenGL) 규약: 카메라는 -z 를 바라봄
        return eye, fwd

    pos_label = "내 위치" if _KO_FONT_OK else "pos"
    pts_label = "찍은 점" if _KO_FONT_OK else "points"
    hud = gui.Label("")
    try:
        hud.text_color = gui.Color(1.0, 1.0, 0.35)
    except Exception:
        pass
    win.add_child(hud)

    def on_layout(ctx):
        r = win.content_rect
        sw.frame = r
        w, h = 380, 132
        hud.frame = gui.Rect(r.get_right() - w - 12, r.y + 12, w, h)
    win.set_on_layout(on_layout)

    def on_tick():
        e, _ = cam_eye_fwd()
        hud.text = (f"{info_text}\n"
                    f"{pos_label}  x={e[0]:+.2f}  y={e[1]:+.2f}  z={e[2]:+.2f}\n"
                    f"{pts_label}: {len(markers)}    [E] add  [R] undo")
        return True
    win.set_on_tick_event(on_tick)

    KN = gui.KeyName
    DOWN = gui.KeyEvent.Type.DOWN
    HANDLED = gui.Widget.EventCallbackResult.HANDLED
    IGNORED = gui.Widget.EventCallbackResult.IGNORED

    def on_key(ev):
        if ev.type != DOWN:
            return IGNORED
        if ev.key == int(KN.E):                      # 점 추가 = 현재 위치 약간 앞
            eye, fwd = cam_eye_fwd()
            p = eye + fwd * 0.5
            add_marker(p)
            print(f"  [점 {len(markers)}] x={p[0]:+.2f} y={p[1]:+.2f} z={p[2]:+.2f}")
            return HANDLED
        if ev.key == int(KN.R):                      # 마지막 점 제거
            if markers:
                name, _ = markers.pop()
                sw.scene.remove_geometry(name)
                print(f"  [삭제] 남은 점 {len(markers)}개")
            return HANDLED
        return IGNORED                               # 나머지(WASD/마우스)는 FLY 내장 처리
    sw.set_on_key(on_key)

    app.run()
    return [list(map(float, m[1])) for m in markers]


def _mark_legacy(pcd, title, existing, point_size):
    step = [0.25]
    markers = []
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=title, width=1400, height=950)
    vis.add_geometry(pcd)
    o = vis.get_render_option(); o.background_color = np.array([0.10, 0.10, 0.12]); o.point_size = point_size

    def eye_R(v):
        cam = v.get_view_control().convert_to_pinhole_camera_parameters()
        ext = np.array(cam.extrinsic); R = ext[:3, :3]
        return -R.T @ ext[:3, 3], R

    def add(pos):
        s = sphere(pos, 0.12, [1.0, 0.12, 0.12])
        vis.add_geometry(s, reset_bounding_box=False)
        markers.append([s, np.asarray(pos, float)])

    for p in existing:
        add(np.asarray(p, float))

    def move(fwd, rgt, upv):
        def cb(v):
            ctr = v.get_view_control()
            cam = ctr.convert_to_pinhole_camera_parameters()
            ext = np.array(cam.extrinsic).copy(); R = ext[:3, :3]; C = -R.T @ ext[:3, 3]
            C = C + (R[2] * fwd + R[0] * rgt + np.array([0, 0, 1.0]) * upv) * step[0]
            ext[:3, 3] = -R @ C; cam.extrinsic = ext
            try:
                ctr.convert_from_pinhole_camera_parameters(cam, True)
            except TypeError:
                ctr.convert_from_pinhole_camera_parameters(cam)
            print(f"  pos x={C[0]:+.2f} y={C[1]:+.2f} z={C[2]:+.2f}")
            return False
        return cb

    def drop(v):
        eye, R = eye_R(v); p = eye + R[2] * 0.4
        add(p); print(f"  [점 {len(markers)}] x={p[0]:+.2f} y={p[1]:+.2f} z={p[2]:+.2f}")
        return True

    def rem(v):
        if markers:
            s, _ = markers.pop(); v.remove_geometry(s, reset_bounding_box=False)
            print(f"  [삭제] 남은 점 {len(markers)}개")
        return True

    for k, mv in [('W', (1, 0, 0)), ('S', (-1, 0, 0)), ('A', (0, -1, 0)),
                  ('D', (0, 1, 0)), (' ', (0, 0, 1)), ('C', (0, 0, -1))]:
        vis.register_key_callback(ord(k), move(*mv))
    vis.register_key_callback(ord('E'), drop)
    vis.register_key_callback(ord('R'), rem)
    vis.run(); vis.destroy_window()
    return [list(map(float, m[1])) for m in markers]


def load_pcd(folder, voxel, rgb=True):
    coord = np.load(os.path.join(_NPY_DIR, folder, 'coord.npy')).astype(np.float64)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coord)
    if rgb:
        col = np.load(os.path.join(_NPY_DIR, folder, 'color.npy')).astype(np.float64) / 255.0
        pcd.colors = o3d.utility.Vector3dVector(col)
    if voxel > 0:
        pcd = pcd.voxel_down_sample(voxel)
    return pcd


def sphere(center, r, color):
    m = o3d.geometry.TriangleMesh.create_sphere(radius=r, resolution=12)
    m.translate(np.asarray(center, float)); m.paint_uniform_color(color); m.compute_vertex_normals()
    return m


# ──────────────────────────────────────────
# 콘솔 입력
# ──────────────────────────────────────────
def ask_room(rid, rooms, conns):
    r = rooms[rid]
    cur_label = r['label'] or '(없음)'
    cur_conn = ','.join(sorted(conns[rid])) or '(없음)'
    print("\n" + "─" * 56)
    print(f"  방 {rid} | npy {r['npy']}")
    print(f"  층 {r['floor']} | 점 {r['n_points']:,} | 중심 {r['center']}")
    print(f"  현재 label       : {cur_label}")
    print(f"  현재 연결된 방   : {cur_conn}")
    print(f"  찍은 통로 점     : {len(r.get('passages', []))}개")
    print("─" * 56)
    print("  [Enter=유지 / 입력=변경 / '-'=비우기 / skip=건너뜀]")
    print("  ※ 연결은 통로 점을 찍어 자동으로 잡힙니다. 아래는 보조 수동 입력입니다.")

    lab = input("  label > ").strip()
    if lab == 'skip':
        return 'skip'
    if lab == '-':
        r['label'] = None
    elif lab:
        r['label'] = lab

    print(f"  연결 방 id (콤마, 예: 011,013). 현재: {cur_conn}")
    c = input("  connections > ").strip()
    if c == '-':
        for other in list(conns[rid]):
            conns[other].discard(rid)
        conns[rid].clear()
    elif c:
        new = {x.strip() for x in c.split(',') if x.strip()}
        unknown = new - set(rooms)
        if unknown:
            print(f"  [경고] 모르는 방 무시: {sorted(unknown)}")
        new &= set(rooms); new.discard(rid)
        for other in list(conns[rid]):
            conns[other].discard(rid)
        conns[rid] = new
        for other in new:
            conns[other].add(rid)
    return 'ok'


# ──────────────────────────────────────────
# [2] 평면도 위에서 통로 점 클릭 찍기
# ──────────────────────────────────────────
def _setup_mpl_korean(matplotlib):
    fp = _find_ko_font()
    if fp:
        try:
            from matplotlib import font_manager
            font_manager.fontManager.addfont(fp)
            matplotlib.rcParams['font.family'] = font_manager.FontProperties(fname=fp).get_name()
        except Exception:
            pass
    matplotlib.rcParams['axes.unicode_minus'] = False


def mark_on_floorplan(rooms, rid, max_pts=30000):
    """rid 가 속한 층의 평면도를 띄우고, 클릭으로 통로 점을 찍는다. 점 리스트 반환.
       좌클릭=점 추가, 우클릭=마지막 점 삭제, 창 닫기=완료."""
    import matplotlib
    import matplotlib.pyplot as plt
    _setup_mpl_korean(matplotlib)

    floor = rooms[rid]['floor']
    members = [r for r in rooms if rooms[r]['floor'] == floor]
    floor_base = min(rooms[m]['bbox_min'][2] for m in members)
    floor_z = round(float(floor_base + 1.0), 3)      # 같은 층은 동일 z → 매칭은 xy 로만
    cmap = floor_colormap(rooms, list(rooms))

    fig, ax = plt.subplots(figsize=(12, 9))
    # 같은 층 방들: 현재 방은 색+진하게, 나머지는 연한 회색
    for m in members:
        coord = np.load(os.path.join(_NPY_DIR, rooms[m]['npy'], 'coord.npy'))
        if len(coord) > max_pts:
            coord = coord[np.random.choice(len(coord), max_pts, replace=False)]
        if m == rid:
            ax.scatter(coord[:, 0], coord[:, 1], s=2, c=[cmap[m]], linewidths=0, zorder=2)
        else:
            ax.scatter(coord[:, 0], coord[:, 1], s=1, c=[[0.80, 0.80, 0.82]], linewidths=0, zorder=1)
        cx, cy = rooms[m]['center'][0], rooms[m]['center'][1]
        ax.text(cx, cy, m, fontsize=14 if m == rid else 10,
                fontweight='bold' if m == rid else 'normal', ha='center', va='center',
                color='black', zorder=7,
                bbox=dict(boxstyle='round', fc=('yellow' if m == rid else 'white'),
                          ec='none', alpha=0.75))
    # 다른 방이 이미 찍은 통로 점(참조용, 파란 원) — 문 맞추기 쉽게
    for m in members:
        if m == rid:
            continue
        for p in rooms[m]['passages']:
            ax.scatter([p[0]], [p[1]], s=80, facecolors='none', edgecolors='blue',
                       linewidths=1.5, zorder=4)

    pts = [list(map(float, p)) for p in rooms[rid]['passages']]
    mark_artists = []

    def redraw():
        while mark_artists:
            mark_artists.pop().remove()
        if pts:
            xy = np.array([[p[0], p[1]] for p in pts])
            sc = ax.scatter(xy[:, 0], xy[:, 1], c='red', s=120, marker='X',
                            edgecolors='white', linewidths=1.2, zorder=8)
            mark_artists.append(sc)
            for i, p in enumerate(pts):
                t = ax.annotate(str(i + 1), (p[0], p[1]), color='white', fontsize=9,
                                ha='center', va='center', zorder=9)
                mark_artists.append(t)
        fig.canvas.draw_idle()

    def on_click(ev):
        if ev.inaxes != ax or ev.xdata is None:
            return
        tb = getattr(fig.canvas, 'toolbar', None)
        if tb is not None and getattr(tb, 'mode', ''):   # 확대/이동 모드면 무시
            return
        if ev.button == 1:       # 좌클릭 = 추가
            pts.append([round(float(ev.xdata), 3), round(float(ev.ydata), 3), floor_z])
            print(f"  [점 {len(pts)}] x={pts[-1][0]:+.2f} y={pts[-1][1]:+.2f} (z={floor_z})")
        elif ev.button == 3:     # 우클릭 = 마지막 삭제
            if pts:
                pts.pop(); print(f"  [삭제] 남은 점 {len(pts)}개")
        redraw()

    fig.canvas.mpl_connect('button_press_event', on_click)
    ax.set_title(f"[통로 점 찍기]  방 {rid}  (npy: {rooms[rid]['npy']},  층 {floor})\n"
                 f"좌클릭=점 추가   우클릭=마지막 삭제   창 닫기=완료    "
                 f"(노랑=현재 방, 파랑원=다른 방이 찍은 점)", fontsize=11)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.set_aspect('equal', 'box'); ax.grid(True, alpha=0.2)
    redraw()
    plt.tight_layout()
    plt.show()
    return pts


# ──────────────────────────────────────────
# [2'] 3D(WASD+E/R) 로 점 찍고, 옆 창에 2D 평면도 실시간 표시
# ──────────────────────────────────────────
def mark_3d_with_plan(rooms, rid, voxel=0.05, max_pts=30000):
    """3D 창에서 WASD 이동 + E(추가)/R(삭제) 로 통로 점을 찍는다.
       별도 2D 평면도 창에 내 위치(초록)와 찍은 점(빨강X)을 실시간 표시.
       찍은 점 리스트([x,y,z]) 반환."""
    import matplotlib
    import matplotlib.pyplot as plt
    _setup_mpl_korean(matplotlib)

    floor = rooms[rid]['floor']
    members = [r for r in rooms if rooms[r]['floor'] == floor]
    floor_base = min(rooms[m]['bbox_min'][2] for m in members)
    cmap = floor_colormap(rooms, list(rooms))

    # ---------- 3D (Open3D legacy: WASD + E/R) ----------
    step = [0.25]
    markers = []   # [sphere, pos(np3)]
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=f"[3D] room {rid}  (WASD move / E add / R del)",
                      width=1100, height=850)
    pcd = load_pcd(rooms[rid]['npy'], voxel, rgb=True)
    vis.add_geometry(pcd)
    ro = vis.get_render_option(); ro.background_color = np.array([0.10, 0.10, 0.12]); ro.point_size = 3

    def add(pos):
        s = sphere(pos, 0.12, [1.0, 0.12, 0.12])
        vis.add_geometry(s, reset_bounding_box=False)
        markers.append([s, np.asarray(pos, float)])

    for p in rooms[rid].get('passages', []):
        add(np.asarray(p, float))

    def cam(v):
        c = v.get_view_control().convert_to_pinhole_camera_parameters()
        ext = np.array(c.extrinsic); R = ext[:3, :3]
        return -R.T @ ext[:3, 3], R

    def move(fwd, rgt, upv):
        def cb(v):
            ctr = v.get_view_control()
            c = ctr.convert_to_pinhole_camera_parameters()
            ext = np.array(c.extrinsic).copy(); R = ext[:3, :3]; C = -R.T @ ext[:3, 3]
            C = C + (R[2] * fwd + R[0] * rgt + np.array([0, 0, 1.0]) * upv) * step[0]
            ext[:3, 3] = -R @ C; c.extrinsic = ext
            try:
                ctr.convert_from_pinhole_camera_parameters(c, True)
            except TypeError:
                ctr.convert_from_pinhole_camera_parameters(c)
            return False
        return cb

    def drop(v):
        eye, R = cam(v); p = eye + R[2] * 0.4
        add(p); print(f"  [점 {len(markers)}] x={p[0]:+.2f} y={p[1]:+.2f} z={p[2]:+.2f}")
        return True

    def rem(v):
        if markers:
            s, _ = markers.pop(); v.remove_geometry(s, reset_bounding_box=False)
            print(f"  [삭제] 남은 점 {len(markers)}개")
        return True

    for k, mv in [('W', (1, 0, 0)), ('S', (-1, 0, 0)), ('A', (0, -1, 0)),
                  ('D', (0, 1, 0)), (' ', (0, 0, 1)), ('C', (0, 0, -1))]:
        vis.register_key_callback(ord(k), move(*mv))
    vis.register_key_callback(ord('E'), drop)
    vis.register_key_callback(ord('R'), rem)

    # ---------- 2D 평면도 (matplotlib, 실시간) ----------
    plt.ion()
    fig, ax = plt.subplots(figsize=(9, 8), num=f"[2D] floor {floor} (room {rid})")
    for m in members:
        coord = np.load(os.path.join(_NPY_DIR, rooms[m]['npy'], 'coord.npy'))
        if len(coord) > max_pts:
            coord = coord[np.random.choice(len(coord), max_pts, replace=False)]
        if m == rid:
            ax.scatter(coord[:, 0], coord[:, 1], s=2, c=[cmap[m]], linewidths=0, zorder=2)
        else:
            ax.scatter(coord[:, 0], coord[:, 1], s=1, c=[[0.80, 0.80, 0.82]], linewidths=0, zorder=1)
        ax.text(rooms[m]['center'][0], rooms[m]['center'][1], m,
                fontsize=13 if m == rid else 9, fontweight='bold' if m == rid else 'normal',
                ha='center', va='center', zorder=7,
                bbox=dict(boxstyle='round', fc=('yellow' if m == rid else 'white'), ec='none', alpha=0.75))
    for m in members:                       # 다른 방이 찍은 점(파란 원, 참조)
        if m == rid:
            continue
        for p in rooms[m].get('passages', []):
            ax.scatter([p[0]], [p[1]], s=80, facecolors='none', edgecolors='blue', linewidths=1.5, zorder=4)
    ax.set_aspect('equal', 'box'); ax.grid(True, alpha=0.2)
    ax.set_title(f"floor {floor} 평면도  (빨강 X=찍은 점, 초록=내 위치/방향)")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    mark_sc = ax.scatter([], [], c='red', s=120, marker='X', edgecolors='white', linewidths=1.2, zorder=8)
    pos_sc = ax.scatter([], [], c='lime', s=150, marker='o', edgecolors='black', zorder=9)
    head_ln, = ax.plot([], [], c='lime', lw=2.5, zorder=9)
    fig.tight_layout()

    # ---------- 합동 이벤트 루프 ----------
    plot_alive = True
    i = 0
    try:
        while True:
            if not vis.poll_events():       # 3D 창 닫힘
                break
            vis.update_renderer()
            i += 1
            if plot_alive and i % 3 == 0:
                if not plt.fignum_exists(fig.number):
                    plot_alive = False
                else:
                    if markers:
                        mark_sc.set_offsets(np.array([[m[1][0], m[1][1]] for m in markers]))
                    else:
                        mark_sc.set_offsets(np.empty((0, 2)))
                    eye, R = cam(vis); f = R[2]
                    pos_sc.set_offsets([[eye[0], eye[1]]])
                    head_ln.set_data([eye[0], eye[0] + f[0] * 0.6], [eye[1], eye[1] + f[1] * 0.6])
                    try:
                        fig.canvas.draw_idle(); fig.canvas.flush_events()
                    except Exception:
                        plot_alive = False
    finally:
        plt.ioff()
        try:
            plt.close(fig)
        except Exception:
            pass
        vis.destroy_window()

    return [list(map(float, m[1])) for m in markers]


# ──────────────────────────────────────────
# 찍은 점만 확인 (--points)
# ──────────────────────────────────────────
def show_points(rooms, conns, voxel=0.06):
    """찍은 통로 점을 3D 로 보며 편집(삭제)한다.
       N=다음 점 / P=이전 점 (선택 점은 노랑↔빨강 깜빡 + 카메라 이동),
       X 또는 Delete=선택 점 삭제.  창 닫으면 변경분 저장."""
    ids = list(rooms)
    cmap = floor_colormap(rooms, ids)
    RED = [1.0, 0.1, 0.1]
    YELLOW = [1.0, 1.0, 0.1]

    # ---- 콘솔 요약 ----
    print("\n=== 방별 통로 점 ===")
    tot = 0
    for rid in ids:
        ps = rooms[rid].get('passages', [])
        tot += len(ps)
        flag = '' if ps else '   (점 없음)'
        coords = '  '.join(f"[{p[0]:.2f},{p[1]:.2f},{p[2]:.2f}]" for p in ps)
        print(f"  {rid} (층{rooms[rid]['floor']}) : {len(ps)}개 {coords}{flag}")
    print(f"총 통로 점: {tot}개")

    # ---- 3D 편집 뷰어 ----
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="통로 점 편집  (N=다음 P=이전 X/Del=삭제)", width=1300, height=900)
    ro = vis.get_render_option(); ro.background_color = np.array([0.10, 0.10, 0.12]); ro.point_size = 2
    for rid in ids:
        pcd = load_pcd(rooms[rid]['npy'], voxel, rgb=False)
        pcd.paint_uniform_color([c * 0.5 for c in cmap[rid]])   # 방은 흐리게
        vis.add_geometry(pcd)

    points = []   # {rid, pos(np3), sphere}
    for rid in ids:
        for p in rooms[rid].get('passages', []):
            s = sphere(p, 0.16, RED)
            vis.add_geometry(s, reset_bounding_box=False)
            points.append({'rid': rid, 'pos': np.asarray(p, float), 'sphere': s})

    state = {'sel': 0, 'frame': 0, 'on': True, 'changed': False}
    step = [0.3]

    def repaint(idx, col):
        if 0 <= idx < len(points):
            points[idx]['sphere'].paint_uniform_color(col)
            vis.update_geometry(points[idx]['sphere'])

    def focus(idx):
        ctr = vis.get_view_control()
        cam = ctr.convert_to_pinhole_camera_parameters()
        ext = np.array(cam.extrinsic).copy(); R = ext[:3, :3]
        eye = points[idx]['pos'] - R[2] * 2.0       # 점에서 2m 뒤로 물러나 바라봄
        ext[:3, 3] = -R @ eye; cam.extrinsic = ext
        try:
            ctr.convert_from_pinhole_camera_parameters(cam, True)
        except TypeError:
            ctr.convert_from_pinhole_camera_parameters(cam)

    def select(new, repaint_old=True):
        if not points:
            return
        new %= len(points)
        if repaint_old:
            repaint(state['sel'], RED)
        state['sel'] = new; state['on'] = True
        p = points[new]
        print(f"  [선택 {new+1}/{len(points)}] 방 {p['rid']}  "
              f"x={p['pos'][0]:+.2f} y={p['pos'][1]:+.2f} z={p['pos'][2]:+.2f}")
        focus(new)

    def nxt(v): select(state['sel'] + 1); return True
    def prv(v): select(state['sel'] - 1); return True

    def dele(v):
        if not points:
            print("  삭제할 점이 없습니다."); return False
        idx = state['sel']
        p = points.pop(idx)
        v.remove_geometry(p['sphere'], reset_bounding_box=False)
        pl = rooms[p['rid']].get('passages', [])
        for k, q in enumerate(pl):
            if np.allclose(np.asarray(q, float), p['pos'], atol=1e-6):
                pl.pop(k); break
        state['changed'] = True
        print(f"  [삭제] 방 {p['rid']} 의 점 제거 → 남은 점 {len(points)}개")
        if points:
            select(idx, repaint_old=False)
        return True

    def anim(v):
        if not points:
            return False
        state['frame'] += 1
        if state['frame'] % 10 == 0:                # 선택 점 깜빡 (노랑↔빨강)
            state['on'] = not state['on']
            repaint(state['sel'], YELLOW if state['on'] else RED)
            return True
        return False

    def move(fwd, rgt, upv):
        def cb(v):
            ctr = v.get_view_control()
            cam = ctr.convert_to_pinhole_camera_parameters()
            ext = np.array(cam.extrinsic).copy(); R = ext[:3, :3]; C = -R.T @ ext[:3, 3]
            C = C + (R[2] * fwd + R[0] * rgt + np.array([0, 0, 1.0]) * upv) * step[0]
            ext[:3, 3] = -R @ C; cam.extrinsic = ext
            try:
                ctr.convert_from_pinhole_camera_parameters(cam, True)
            except TypeError:
                ctr.convert_from_pinhole_camera_parameters(cam)
            return False
        return cb

    for k, mv in [('W', (1, 0, 0)), ('S', (-1, 0, 0)), ('A', (0, -1, 0)),
                  ('D', (0, 1, 0)), (' ', (0, 0, 1)), ('C', (0, 0, -1))]:
        vis.register_key_callback(ord(k), move(*mv))
    vis.register_key_callback(ord('N'), nxt)
    vis.register_key_callback(ord('P'), prv)
    vis.register_key_callback(ord('X'), dele)
    vis.register_key_callback(261, dele)            # Delete 키
    vis.register_animation_callback(anim)

    print(f"\n[3D 점 편집] 점 {len(points)}개")
    print("  N=다음 점, P=이전 점 (선택 점=노랑 깜빡 + 카메라 이동)")
    print("  X 또는 Delete=선택 점 삭제   이동 W/S/A/D · Space/C 상하 · 마우스=시점")
    print("  창을 닫으면 변경분 저장.")
    if points:
        select(0, repaint_old=False)

    vis.run(); vis.destroy_window()

    if state['changed']:
        save(rooms, conns)
        left = sum(len(rooms[r].get('passages', [])) for r in ids)
        print(f"\n저장됨 → {os.path.basename(_GRAPH)} (남은 통로 점 {left}개)")
    else:
        print("\n변경 없음 — 저장 안 함.")


# ──────────────────────────────────────────
# 방 연결(간선) 2D 수동 편집 — 모든 층을 한 화면에 나란히
# ──────────────────────────────────────────
def edit_edges_2d(rooms, conns, max_pts=20000):
    """모든 층을 좌→우로 나란히 펼친 2D 평면도에서 방을 클릭해 연결을 직접 그린다.
       방 A 클릭 → 방 B 클릭 = A-B 연결(이미 있으면 해제). 다른 층 방을 클릭하면 계단(층간) 연결.
       우클릭=선택 취소.  창 닫으면 '사용자가 그은 간선만' 저장(자동 병합 끔)."""
    import matplotlib
    import matplotlib.pyplot as plt
    _setup_mpl_korean(matplotlib)

    ids = list(rooms)
    cmap = floor_colormap(rooms, ids)
    floors = sorted({rooms[r]['floor'] for r in ids})

    # 층별 x 오프셋 (좌→우로 나열해서 한 화면에 다 보이게)
    gap = 3.0
    running = 0.0
    xoff = {}
    for fl in floors:
        mem = [r for r in ids if rooms[r]['floor'] == fl]
        xmin = min(rooms[m]['bbox_min'][0] for m in mem)
        xmax = max(rooms[m]['bbox_max'][0] for m in mem)
        xoff[fl] = running - xmin
        running += (xmax - xmin) + gap

    centers = {rid: (rooms[rid]['center'][0] + xoff[rooms[rid]['floor']],
                     rooms[rid]['center'][1]) for rid in ids}

    fig, ax = plt.subplots(figsize=(16, 9))
    for rid in ids:
        ox = xoff[rooms[rid]['floor']]
        coord = np.load(os.path.join(_NPY_DIR, rooms[rid]['npy'], 'coord.npy'))
        if len(coord) > max_pts:
            coord = coord[np.random.choice(len(coord), max_pts, replace=False)]
        ax.scatter(coord[:, 0] + ox, coord[:, 1], s=1, c=[cmap[rid]], linewidths=0, zorder=1)
        cx, cy = centers[rid]
        ax.text(cx, cy, rid, fontsize=10, fontweight='bold', ha='center', va='center', zorder=6,
                bbox=dict(boxstyle='round', fc='white', ec='none', alpha=0.7))
        for p in rooms[rid].get('passages', []):     # 찍은 점(참조)
            ax.scatter([p[0] + ox], [p[1]], s=30, c='red', marker='x', zorder=4)
    for fl in floors:                                  # 층 제목
        mem = [r for r in ids if rooms[r]['floor'] == fl]
        cx = float(np.mean([centers[m][0] for m in mem]))
        ytop = max(rooms[m]['bbox_max'][1] for m in mem)
        ax.text(cx, ytop + 1.5, f"Floor {fl}", fontsize=15, fontweight='bold',
                ha='center', color='navy', zorder=7)

    sel = [None]
    sel_artist = ax.scatter([], [], s=500, facecolors='none', edgecolors='lime', linewidths=2.5, zorder=8)
    edge_artists = []

    def redraw_edges():
        while edge_artists:
            edge_artists.pop().remove()
        drawn = set()
        for a in conns:
            for b in conns[a]:
                key = tuple(sorted((a, b)))
                if key in drawn or a not in centers or b not in centers:
                    continue
                drawn.add(key)
                (x1, y1), (x2, y2) = centers[key[0]], centers[key[1]]
                cross = rooms[key[0]]['floor'] != rooms[key[1]]['floor']
                ln, = ax.plot([x1, x2], [y1, y2], c=('orange' if cross else 'lime'),
                              lw=2.2, zorder=5, alpha=0.9)
                edge_artists.append(ln)
        fig.canvas.draw_idle()

    def set_sel(rid):
        sel[0] = rid
        sel_artist.set_offsets(np.empty((0, 2)) if rid is None else [centers[rid]])
        fig.canvas.draw_idle()

    def room_at(x, y):
        cand = []
        for rid in ids:
            ox = xoff[rooms[rid]['floor']]
            if (rooms[rid]['bbox_min'][0] + ox <= x <= rooms[rid]['bbox_max'][0] + ox and
                    rooms[rid]['bbox_min'][1] <= y <= rooms[rid]['bbox_max'][1]):
                cand.append(rid)
        if cand:
            return min(cand, key=lambda r: (centers[r][0] - x) ** 2 + (centers[r][1] - y) ** 2)
        r = min(ids, key=lambda r: (centers[r][0] - x) ** 2 + (centers[r][1] - y) ** 2)
        d = ((centers[r][0] - x) ** 2 + (centers[r][1] - y) ** 2) ** 0.5
        return r if d < 3.0 else None

    def on_click(ev):
        if ev.inaxes != ax or ev.xdata is None:
            return
        tb = getattr(fig.canvas, 'toolbar', None)
        if tb is not None and getattr(tb, 'mode', ''):
            return
        if ev.button == 3:                  # 우클릭 = 선택 취소
            set_sel(None); return
        rid = room_at(ev.xdata, ev.ydata)
        if rid is None:
            return
        if sel[0] is None:
            set_sel(rid)
        elif sel[0] == rid:
            set_sel(None)
        else:
            a, b = sel[0], rid
            if b in conns[a]:
                conns[a].discard(b); conns[b].discard(a)
                print(f"  [연결 해제] {a} - {b}")
            else:
                conns[a].add(b); conns[b].add(a)
                cross = rooms[a]['floor'] != rooms[b]['floor']
                print(f"  [연결] {a} - {b}   {'(계단/층간)' if cross else ''}")
            set_sel(None); redraw_edges()

    fig.canvas.mpl_connect('button_press_event', on_click)
    redraw_edges()
    ax.set_aspect('equal', 'box'); ax.grid(True, alpha=0.2)
    ax.set_title("방 연결 편집 — 방 두 개를 차례로 클릭=연결/해제,  우클릭=선택취소,  창 닫기=저장\n"
                 "초록선=같은 층,  주황선=층간(계단)  /  빨간 x=찍은 통로 점", fontsize=11)
    print("\n[방 연결 편집] 방 두 개를 차례로 클릭하면 연결(다시 클릭하면 해제).")
    print("  다른 층 패널의 방을 클릭하면 계단(층간) 연결.  우클릭=선택취소.  창 닫으면 저장.")
    plt.tight_layout()
    plt.show()

    save(rooms, conns, auto_passages=False)   # 사용자가 그은 간선만
    edges = build_edges(rooms, conns, auto_passages=False)
    nd = sum(1 for e in edges if e['type'] == 'door')
    ns = sum(1 for e in edges if e['type'] == 'stairs')
    print(f"\n저장됨 → {os.path.basename(_GRAPH)}  (간선 {len(edges)}개: 문 {nd}, 계단 {ns})")


# ──────────────────────────────────────────
# [3] 전체 통합 뷰
# ──────────────────────────────────────────
def review(rooms, conns, voxel):
    print("\n[3단계] 전체 통합 — 방=층별색, 빨간 구=문/연결, 초록 선=연결")
    ids = list(rooms)
    cmap = floor_colormap(rooms, ids)
    geoms = []
    for rid in ids:
        pcd = load_pcd(rooms[rid]['npy'], voxel, rgb=False)
        pcd.paint_uniform_color(cmap[rid])
        geoms.append(pcd)

    edges = build_edges(rooms, conns)
    segs = []
    for e in edges:
        dc = e['door_center']
        geoms.append(sphere(dc, 0.15, [0.95, 0.1, 0.1] if e['type'] == 'door' else [0.95, 0.85, 0.1]))
        segs.append((rooms[e['a']]['center'], dc)); segs.append((dc, rooms[e['b']]['center']))
    if segs:
        pts, lines = [], []
        for p0, p1 in segs:
            i = len(pts); pts += [p0, p1]; lines.append([i, i + 1])
        ls = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(np.asarray(pts, float)),
                                  lines=o3d.utility.Vector2iVector(np.asarray(lines, int)))
        ls.colors = o3d.utility.Vector3dVector(np.tile([0.1, 0.9, 0.2], (len(lines), 1)))
        geoms.append(ls)

    print("\n  === 라벨 / 연결 요약 ===")
    for rid in ids:
        print(f"  {rid}: {(rooms[rid]['label'] or '(라벨없음)'):<14} -> "
              f"{','.join(sorted(conns[rid])) or '(연결없음)'}")
    iso = [rid for rid in ids if not conns[rid]]
    if iso:
        print(f"  [주의] 연결 없는 고립 방: {iso}")

    fly_view(geoms, "[3] 전체 통합 (WASD 이동)",
             f"ALL rooms: {len(ids)}  /  links: {len(edges)}", point_size=2)


# ──────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default=None)
    ap.add_argument('--only', default=None)
    ap.add_argument('--review', action='store_true')
    ap.add_argument('--points', action='store_true', help='찍은 통로 점만 확인/편집(간선 무시)')
    ap.add_argument('--edges', action='store_true', help='모든 층 펼친 2D 에서 방 연결 직접 편집')
    ap.add_argument('--rebuild', action='store_true', help='현재 점으로 간선만 다시 계산해 저장')
    ap.add_argument('--merge-dist', type=float, default=None,
                    help='통로 점 연결 임계거리(m). 기본 0.9. 예: --merge-dist 1.6')
    ap.add_argument('--fresh', action='store_true')
    ap.add_argument('--voxel', type=float, default=0.05)
    args = ap.parse_args()

    global _MERGE_DIST
    if args.merge_dist is not None:
        _MERGE_DIST = args.merge_dist
        print(f"[설정] 연결 임계거리 = {_MERGE_DIST}m")

    rooms = scan_rooms()
    conns = ({rid: set() for rid in rooms} if args.fresh else load_graph(rooms))
    ids = list(rooms)

    if args.points:
        show_points(rooms, conns)
        return

    if args.edges:
        edit_edges_2d(rooms, conns)
        return

    if args.rebuild:
        save(rooms, conns)
        edges = build_edges(rooms, conns)
        print(f"\n[간선 재계산] 임계 {_MERGE_DIST}m → 간선 {len(edges)}개")
        for e in edges:
            print(f"  {e['a']} - {e['b']}  {e['type']}  ({e['source']})  @ {e['door_center']}")
        print(f"저장됨 → {os.path.basename(_GRAPH)}")
        return

    if args.review:
        review(rooms, conns, max(args.voxel, 0.05))
        return

    # 대상 선정
    if args.only:
        targets = [x.strip() for x in args.only.split(',') if x.strip() in rooms]
    else:
        targets = ids[ids.index(args.start):] if (args.start in ids) else ids

    print("\n[통로 점 찍기] 3D 창에서 날아다니며 문/통로에 점을 찍습니다. (옆 2D 창=내 위치/점 실시간)")
    print("  이동: W/S 전후  A/D 좌우  Space=상승  C=하강   시점: 마우스 드래그")
    print("  점:   E=현재 위치에 점 추가   R=마지막 점 삭제   창 닫기=다음 방으로")
    print("  2D 평면도의 파란 원 = 다른 방이 찍은 점 (그 근처에 찍으면 두 방 자동 연결)")

    try:
        for n, rid in enumerate(targets, 1):
            if not args.only and rooms[rid]['label']:
                a = input(f"\n방 {rid} 이미 label='{rooms[rid]['label']}'. 다시 찍을까요? [y/N] ").strip().lower()
                if a != 'y':
                    continue
            folder = rooms[rid]['npy']
            print(f"\n[{n}/{len(targets)}] 방 {rid}  (npy: {folder}) — 층 {rooms[rid]['floor']}  3D+2D 창 표시")
            pts = mark_3d_with_plan(rooms, rid, voxel=args.voxel)
            rooms[rid]['passages'] = pts
            print(f"  통로 점 {len(pts)}개 기록: {pts}")
            if ask_room(rid, rooms, conns) == 'skip':
                print("  (label 입력은 건너뜀, 통로 점은 저장)")
            save(rooms, conns)
            print(f"  저장됨 → {os.path.basename(_GRAPH)}")
    except KeyboardInterrupt:
        print("\n[중단] 저장 후 종료."); save(rooms, conns); return

    save(rooms, conns)
    print("\n모든 방 완료. 전체 통합을 띄웁니다...")
    review(rooms, conns, max(args.voxel, 0.05))


if __name__ == '__main__':
    main()
