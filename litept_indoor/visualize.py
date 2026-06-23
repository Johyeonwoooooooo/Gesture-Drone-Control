"""
centers.pkl viser 시각화

사용법:
    python visualize.py --npy_root project/data/npy   # 전체 방
    python visualize.py --pkl project/data/npy/ROOM/centers.pkl  # 방 하나

브라우저에서 http://localhost:8080 접속
"""
import argparse
import pickle
import threading
import time
from pathlib import Path

import numpy as np
import viser

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

# 라벨별 색상 배열 (인덱스 순)
LABEL_COLOR_ARR = np.array(
    [CLASS_COLORS[c] for c in SCANNET_CLASSES], dtype=np.float32
)

def get_label_color(label_idx, class_name):
    c = CLASS_COLORS.get(class_name)
    if c is None and 0 <= label_idx < len(LABEL_COLOR_ARR):
        c = tuple(LABEL_COLOR_ARR[label_idx])
    return np.array(c or (0.5, 0.5, 0.5), dtype=np.float32)


def run_dbscan(room_data, eps, min_samples):
    from sklearn.cluster import DBSCAN
    all_centers = []
    for rd in room_data:
        coord_all  = rd['coord_all']
        labels_all = rd['labels_all']
        classes    = rd['classes']
        centers = []
        for label_idx, class_name in enumerate(classes):
            if class_name in IGNORE_CLASSES:
                continue
            mask = labels_all == label_idx
            if mask.sum() < min_samples:
                continue
            pts = coord_all[mask]
            db  = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
            cids = db.fit_predict(pts)
            for cid in np.unique(cids):
                if cid == -1:
                    continue
                cm = cids == cid
                centers.append(dict(
                    class_name=class_name,
                    label_idx=label_idx,
                    center=pts[cm].mean(axis=0),
                    n_points=int(cm.sum()),
                ))
        all_centers.append(centers)
    return all_centers


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
    else:
        pkl_files = sorted(args.npy_root.glob('*/centers.pkl'))

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

        # 벡터화된 라벨 색 계산
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

        room_data.append(dict(
            pts=pts_show,
            lbl_colors=lbl_colors,
            rgb_colors=rgb_colors,
            labels_show=labels_show,
            centers=d['centers'],
            classes=classes,
            coord_all=coord_valid,
            labels_all=labels_valid,
        ))
        print(f'  {d["room"]}: {N} pts, {len(d["centers"])} instances')

    # ── viser 서버 ────────────────────────────────────────────
    server = viser.ViserServer(port=args.port)
    server.scene.world_axes.visible = True

    # ── GUI ──────────────────────────────────────────────────
    with server.gui.add_folder('Point Cloud'):
        mode_dropdown = server.gui.add_dropdown(
            label='Color Mode',
            options=['Label', 'RGB', 'Blend'],
            initial_value='Label',
        )
        blend_slider = server.gui.add_slider(
            label='Blend (label %)',
            min=0, max=100, step=1, initial_value=30,
        )
        size_slider = server.gui.add_slider(
            label='Point Size',
            min=0.001, max=0.1, step=0.001, initial_value=0.05,
        )

    with server.gui.add_folder('DBSCAN'):
        eps_slider = server.gui.add_slider(
            label='eps (m)',
            min=0.05, max=1.0, step=0.05, initial_value=0.3,
        )
        min_pts_slider = server.gui.add_slider(
            label='min_samples',
            min=10, max=500, step=10, initial_value=150,
        )
        dbscan_btn = server.gui.add_button(label='Apply DBSCAN')

    with server.gui.add_folder('Labels'):
        show_labels_cb = server.gui.add_checkbox(label='Show center labels', initial_value=False)

    # ── 렌더 함수 ─────────────────────────────────────────────
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

    # label 핸들 저장소 {room_idx: [handle, ...]}
    label_handles = {i: [] for i in range(len(room_data))}

    def add_centers(room_idx, centers):
        label_handles[room_idx].clear()
        show_lbl = show_labels_cb.value
        for i, obj in enumerate(centers):
            cx, cy, cz = obj['center'].astype(float)
            c = get_label_color(obj['label_idx'], obj['class_name'])
            server.scene.add_icosphere(
                name=f'room/{room_idx:02d}/center/{i:03d}',
                radius=0.08,
                position=(cx, cy, cz),
                color=(int(c[0]*255), int(c[1]*255), int(c[2]*255)),
            )
            h = server.scene.add_label(
                name=f'room/{room_idx:02d}/clabel/{i:03d}',
                text=obj['class_name'],
                position=(cx, cy, cz + 0.15),
            )
            h.visible = show_lbl
            label_handles[room_idx].append(h)

    def apply_center_labels():
        show_lbl = show_labels_cb.value
        for handles in label_handles.values():
            for h in handles:
                h.visible = show_lbl

    def apply_centers_async():
        eps         = eps_slider.value
        min_samples = int(min_pts_slider.value)
        print(f'DBSCAN 실행 중... eps={eps}, min_samples={min_samples}')
        all_centers = run_dbscan(room_data, eps, min_samples)
        for room_idx, centers in enumerate(all_centers):
            server.scene.remove_by_name(f'room/{room_idx:02d}/center')
            server.scene.remove_by_name(f'room/{room_idx:02d}/clabel')
            add_centers(room_idx, centers)
        print('DBSCAN 완료')

    # 초기 렌더
    apply_colors()
    for room_idx, rd in enumerate(room_data):
        add_centers(room_idx, rd['centers'])

    @mode_dropdown.on_update
    def _(_): apply_colors()

    @blend_slider.on_update
    def _(_): apply_colors()

    @size_slider.on_update
    def _(_): apply_colors()

    @show_labels_cb.on_update
    def _(_): apply_center_labels()

    @dbscan_btn.on_click
    def _(_):
        threading.Thread(target=apply_centers_async, daemon=True).start()

    print(f'\nviser 서버: http://localhost:{args.port}')
    print('종료: Ctrl+C')

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
