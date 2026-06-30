"""
centers.pkl viser 시각화 + 자연어 검색 하이라이트

사용법:
    python visualize.py --npy_root project/data/npy   # 전체 방
    python visualize.py --pkl project/data/npy/ROOM/centers.pkl  # 방 하나

브라우저에서 http://localhost:8080 접속
검색창에 "거실 소파", "침실 침대" 등 입력 후 엔터
"""
import argparse
import json
import pickle
import threading
import time
from pathlib import Path

import numpy as np
import viser

from export_json import ROOM_TYPE_KO

SCANNET_CLASSES = [
    'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table',
    'door', 'window', 'bookshelf', 'picture', 'counter', 'desk',
    'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink',
    'bathtub', 'otherfurniture',
]
CLASS_COLORS = {
    'wall':             (0.8, 0.8, 0.8),
    'floor':            (0.6, 0.5, 0.4),
    'cabinet':          (0.8, 0.4, 0.1),
    'bed':              (0.2, 0.4, 0.9),
    'chair':            (0.1, 0.8, 0.2),
    'sofa':             (0.6, 0.2, 0.8),
    'table':            (0.9, 0.7, 0.1),
    'door':             (0.5, 0.3, 0.1),
    'window':           (0.4, 0.8, 0.9),
    'bookshelf':        (0.1, 0.5, 0.3),
    'picture':          (0.9, 0.3, 0.5),
    'counter':          (0.7, 0.6, 0.3),
    'desk':             (0.8, 0.5, 0.2),
    'curtain':          (0.3, 0.3, 0.9),
    'refrigerator':     (0.1, 0.7, 0.7),
    'shower curtain':   (0.5, 0.9, 0.8),
    'toilet':           (0.9, 0.9, 0.2),
    'sink':             (0.2, 0.6, 0.9),
    'bathtub':          (0.4, 0.4, 0.9),
    'otherfurniture':   (0.5, 0.5, 0.5),
}
IGNORE_CLASSES = {'wall', 'floor'}

LABEL_COLOR_ARR = np.array(
    [CLASS_COLORS[c] for c in SCANNET_CLASSES], dtype=np.float32
)

def get_label_color(label_idx, class_name):
    c = CLASS_COLORS.get(class_name)
    if c is None and 0 <= label_idx < len(LABEL_COLOR_ARR):
        c = tuple(LABEL_COLOR_ARR[label_idx])
    return np.array(c or (0.5, 0.5, 0.5), dtype=np.float32)





def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--pkl',      type=Path)
    group.add_argument('--npy_root', type=Path)
    parser.add_argument('--port',    type=int, default=8080)
    parser.add_argument('--max_pts', type=int, default=200000)
    args = parser.parse_args()

    if args.pkl:
        pkl_files = [args.pkl]
        json_path = args.pkl.parent.parent / 'detections.json'
    else:
        pkl_files = sorted(args.npy_root.glob('*/centers.pkl'))
        json_path = args.npy_root / 'detections.json'

    print(f'총 {len(pkl_files)}개 방 로드 중...')

    room_data = []
    for pkl_path in pkl_files:
        d = pickle.load(open(pkl_path, 'rb'))
        coord   = d['coord']
        labels  = d['pred_labels']
        classes = d['classes']

        valid_mask   = labels >= 0
        coord_valid  = coord[valid_mask]
        labels_valid = labels[valid_mask]

        N    = len(coord_valid)
        step = max(1, N // args.max_pts)
        idx  = np.arange(0, N, step)

        pts_show    = coord_valid[idx].astype(np.float32)
        labels_show = labels_valid[idx]

        lbl_colors = np.zeros((len(idx), 3), dtype=np.float32)
        valid_lbl  = (labels_show >= 0) & (labels_show < len(LABEL_COLOR_ARR))
        lbl_colors[valid_lbl] = LABEL_COLOR_ARR[labels_show[valid_lbl]]

        npy_dir  = pkl_path.parent
        rgb_path = npy_dir / 'color.npy'
        if rgb_path.exists():
            rgb_raw = np.load(rgb_path).astype(np.float32)
            if rgb_raw.max() > 1.0:
                rgb_raw /= 255.0
            rgb_colors = rgb_raw[valid_mask][idx, :3]
        else:
            rgb_colors = np.ones_like(lbl_colors) * 0.5

        # detections.json에서 room_type 읽기 (있으면)
        json_path = pkl_path.parent.parent / 'detections.json'
        room_type = 'unknown'
        if json_path.exists():
            import json as _json
            with open(json_path, encoding='utf-8') as f:
                all_entries = _json.load(f)
            for e in all_entries:
                if e['room'] == d['room']:
                    room_type = e.get('room_type', 'unknown')
                    break

        room_data.append(dict(
            room=d['room'],
            room_type=room_type,
            pts=pts_show,
            lbl_colors=lbl_colors,
            rgb_colors=rgb_colors,
            labels_show=labels_show,
            centers=d['centers'],
            classes=classes,
            coord_all=coord_valid,
            labels_all=labels_valid,
            step=step,
        ))
        ko = ROOM_TYPE_KO.get(room_type, room_type)
        print(f'  {d["room"]}: {N} pts, {len(d["centers"])} instances  →  {ko}')

    # ── detections.json / room_names.json 로드 ───────────────────────────────
    if json_path.exists():
        with open(json_path, encoding='utf-8') as f:
            detections_json: list = json.load(f)
    else:
        detections_json = []

    names_path = json_path.parent / 'room_names.json'
    if names_path.exists():
        with open(names_path, encoding='utf-8') as f:
            room_names: dict = json.load(f)
    else:
        room_names = {}

    # room_data에 room_name 주입
    for rd in room_data:
        rd['room_name'] = room_names.get(rd['room'], rd['room'])

    def save_room_names():
        with open(names_path, 'w', encoding='utf-8') as f:
            json.dump(room_names, f, ensure_ascii=False, indent=2)
        # detections.json room_name 필드도 갱신
        for e in detections_json:
            e['room_name'] = room_names.get(e['room'], e['room'])
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(detections_json, f, ensure_ascii=False, indent=2)

    def _center_key(room: str, label: str, cx: float, cy: float, cz: float) -> str:
        return f'{room}|{label}|{cx:.3f}|{cy:.3f}|{cz:.3f}'

    removed_keys: set = set()

    def remove_from_json(room: str, label: str, cx: float, cy: float, cz: float):
        """detections.json에서 해당 인스턴스 제거 후 저장."""
        key = _center_key(room, label, cx, cy, cz)
        removed_keys.add(key)
        filtered = [
            e for e in detections_json
            if _center_key(e['room'], e['label'],
                           e['center'][0], e['center'][1], e['center'][2]) not in removed_keys
        ]
        detections_json.clear()
        detections_json.extend(filtered)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(detections_json, f, ensure_ascii=False, indent=2)
        print(f'  [제거] {label} @ {room} ({cx:.2f},{cy:.2f},{cz:.2f}) → {json_path.name} 저장')

    # ── viser 서버 ────────────────────────────────────────────────────────────
    server = viser.ViserServer(port=args.port)
    server.scene.world_axes.visible = True

    # ── GUI ──────────────────────────────────────────────────────────────────
    current_room_text = server.gui.add_text(label='현재 방', initial_value='-')

    with server.gui.add_folder('Point Cloud'):
        mode_dropdown = server.gui.add_dropdown(
            label='Color Mode',
            options=['Label', 'RGB', 'Blend'],
            initial_value='Blend',
        )
        blend_slider = server.gui.add_slider(
            label='Blend (label %)',
            min=0, max=100, step=1, initial_value=30,
        )
        size_slider = server.gui.add_slider(
            label='Point Size',
            min=0.001, max=0.1, step=0.001, initial_value=0.05,
        )

    with server.gui.add_folder('🏠 방 이름'):
        room_select = server.gui.add_dropdown(
            label='방 선택',
            options=[rd['room_name'] for rd in room_data],
            initial_value=room_data[0]['room_name'] if room_data else '',
        )
        room_name_input = server.gui.add_text(
            label='이름 변경',
            initial_value=room_data[0]['room_name'] if room_data else '',
        )
        rename_btn = server.gui.add_button(label='저장')
        rename_status = server.gui.add_text(label='', initial_value='')

    def get_rd_by_name(name: str):
        return next((rd for rd in room_data if rd['room_name'] == name), None)

    @room_select.on_update
    def _(event):
        rd = get_rd_by_name(room_select.value)
        if rd:
            room_name_input.value = rd['room_name']

    @rename_btn.on_click
    def _(_):
        new_name = room_name_input.value.strip()
        if not new_name:
            rename_status.value = '이름을 입력하세요'
            return
        old_name = room_select.value
        rd = get_rd_by_name(old_name)
        if rd is None:
            return
        # 중복 체크
        if any(r['room_name'] == new_name and r is not rd for r in room_data):
            rename_status.value = f'"{new_name}" 이미 사용 중'
            return
        # 적용
        room_names[rd['room']] = new_name
        rd['room_name'] = new_name
        save_room_names()
        # 드롭다운 옵션 갱신
        room_select.options = [r['room_name'] for r in room_data]
        room_select.value = new_name
        rename_status.value = f'저장 완료: {old_name} → {new_name}'
        print(f'  [이름변경] {rd["room"]}: "{old_name}" → "{new_name}"')

    # ── 렌더 함수 ─────────────────────────────────────────────────────────────
    def compute_colors(rd):
        mode  = mode_dropdown.value
        alpha = blend_slider.value / 100.0

        if mode == 'Label':
            base = rd['lbl_colors'].copy()
        elif mode == 'RGB':
            base = rd['rgb_colors'].copy()
        else:
            base = rd['rgb_colors'] * (1 - alpha) + rd['lbl_colors'] * alpha

        return (base.clip(0, 1) * 255).astype(np.uint8)

    def apply_colors():
        size = size_slider.value
        for room_idx, rd in enumerate(room_data):
            server.scene.add_point_cloud(
                name=f'room/{room_idx:02d}/pc',
                points=rd['pts'],
                colors=compute_colors(rd),
                point_size=size,
            )

    # label_handles: room_idx → list of (handle, center_np)
    # label_info: room_idx → list of (label_name, text, position)
    label_info = {i: [] for i in range(len(room_data))}

    # detections.json에 살아있는 항목 키 집합
    active_keys = {
        _center_key(e['room'], e['label'], e['center'][0], e['center'][1], e['center'][2])
        for e in detections_json
    }

    def add_centers(room_idx, centers):
        label_info[room_idx].clear()
        rd = room_data[room_idx]
        z_min = float(rd['pts'][:, 2].min())
        z_max = float(rd['pts'][:, 2].max())
        ceiling_thresh = z_min + (z_max - z_min) * 0.85
        for i, obj in enumerate(centers):
            cx, cy, cz = obj['center'].astype(float)
            if cz > ceiling_thresh:
                continue
            # detections.json에서 제거된 항목은 표시 안 함
            if _center_key(rd['room'], obj['class_name'], cx, cy, cz) not in active_keys:
                continue
            c = get_label_color(obj['label_idx'], obj['class_name'])
            sphere_name = f'room/{room_idx:02d}/center/{i:03d}'
            label_name  = f'room/{room_idx:02d}/clabel/{i:03d}'
            sphere_handle = server.scene.add_icosphere(
                name=sphere_name,
                radius=0.08,
                position=(cx, cy, cz),
                color=(int(c[0]*255), int(c[1]*255), int(c[2]*255)),
            )
            # 라벨 정보만 저장, 아직 씬에 추가 안 함
            label_info[room_idx].append((label_name, obj['class_name'], (cx, cy, cz + 0.15)))

            room_name  = rd['room']
            class_name = obj['class_name']
            _cx, _cy, _cz = cx, cy, cz

            @sphere_handle.on_click
            def _on_click(_, *, _sn=sphere_name, _ln=label_name,
                          _r=room_name, _l=class_name,
                          _x=_cx, _y=_cy, _z=_cz,
                          _ridx=room_idx):
                server.scene.remove_by_name(_sn)
                server.scene.remove_by_name(_ln)
                label_info[_ridx][:] = [
                    e for e in label_info[_ridx] if e[0] != _ln
                ]
                remove_from_json(_r, _l, _x, _y, _z)

    # 방 바운딩박스 + 중심점 캐시
    room_bboxes = [(rd['pts'].min(axis=0), rd['pts'].max(axis=0)) for rd in room_data]
    room_centroids = [rd['pts'].mean(axis=0) for rd in room_data]

    current_room_idx = [-1]

    def find_room_for_pos(pos: np.ndarray) -> int:
        """카메라 위치가 속한 방 인덱스 반환. bbox 안에 있으면 가장 작은 방, 없으면 가장 가까운 방."""
        inside = []
        for i, (lo, hi) in enumerate(room_bboxes):
            if np.all(pos >= lo) and np.all(pos <= hi):
                vol = float(np.prod(hi - lo))
                inside.append((vol, i))
        if inside:
            return min(inside)[1]
        # bbox 밖이면 중심점 거리로 fallback
        dists = [np.linalg.norm(pos - c) for c in room_centroids]
        return int(np.argmin(dists))

    def update_labels_for_room(room_idx: int):
        if room_idx == current_room_idx[0]:
            return
        # 이전 방 라벨 제거
        prev = current_room_idx[0]
        if prev >= 0:
            for name, _, _ in label_info[prev]:
                server.scene.remove_by_name(name)
        # 새 방 라벨 추가
        for name, text, pos in label_info[room_idx]:
            server.scene.add_label(name=name, text=text, position=pos)
        current_room_idx[0] = room_idx
        current_room_text.value = room_data[room_idx]['room_name']

    def camera_tracker():
        while True:
            time.sleep(0.3)
            clients = server.get_clients()
            if not clients:
                continue
            client = next(iter(clients.values()))
            try:
                pos = np.array(client.camera.position, dtype=np.float32)
                nearest = find_room_for_pos(pos)
                update_labels_for_room(nearest)
            except Exception:
                pass

    # 초기 렌더
    apply_colors()
    for room_idx, rd in enumerate(room_data):
        add_centers(room_idx, rd['centers'])

    threading.Thread(target=camera_tracker, daemon=True).start()

    @mode_dropdown.on_update
    def _(_): apply_colors()

    @blend_slider.on_update
    def _(_): apply_colors()

    @size_slider.on_update
    def _(_): apply_colors()

    def rotate_10deg(client):
        """카메라를 Z축 기준 10도 수평 회전."""
        import math
        angle = math.radians(10)
        s = math.sin(angle / 2)
        c = math.cos(angle / 2)
        w, x, y, z = client.camera.wxyz
        # q_rot(Z, 10°) * q_cam
        client.camera.wxyz = np.array([
            c*w - s*z,
            c*x - s*y,
            c*y + s*x,
            c*z + s*w,
        ])

    @server.on_client_connect
    def _(client):
        client.gui.add_command(
            'rotate10',
            hotkey='R',
            description='카메라 10° 수평 회전 (누르는 동안 연속)',
        ).on_trigger(lambda _: rotate_10deg(client))


    print(f'\nviser 서버: http://localhost:{args.port}')
    print('종료: Ctrl+C')

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
