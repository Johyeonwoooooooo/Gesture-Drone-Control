# app_query_highlight.py
import numpy as np
import pickle
import torch
import open_clip
import viser

_TOKENIZER = open_clip.get_tokenizer('ViT-B-32')
import time

INDEX_FILE = '/data1/workspaces/jgshin22/Gesture-Drone-Control/unidet3d/data/clip_index.pkl'
TOPK = 3

# ── 색상 팔레트 ──────────────────────────────
COLOR_INBOX  = np.array([1.0, 0.8, 0.2], dtype=np.float32)  # bbox 내부 point: 노랑
COLOR_MATCH  = np.array([1.0, 0.1, 0.1], dtype=np.float32)  # 쿼리 매칭 bbox: 빨강
COLOR_OTHERS = np.array([0.3, 0.6, 1.0], dtype=np.float32)  # 일반 bbox: 파랑


def load_clip(device):
    model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai', device=device)
    model.eval()
    return model


def query_clip(model, device, box_embeds_t, text, topk):
    tokens = _TOKENIZER([text]).to(device)

    with torch.no_grad():
        text_emb = model.encode_text(tokens)
        text_emb = text_emb.float()
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

    sims = (box_embeds_t @ text_emb.T).squeeze(-1).detach().cpu().numpy()
    top_idx = np.argsort(sims)[::-1][:topk]
    return top_idx, sims


def bbox_corners(box):
    """
    box: [cx, cy, cz, dx, dy, dz, yaw] 또는 [cx, cy, cz, dx, dy, dz]
    return: (8, 3)
    """
    if len(box) >= 7:
        cx, cy, cz, dx, dy, dz, yaw = box[:7]
    else:
        cx, cy, cz, dx, dy, dz = box[:6]
        yaw = 0.0

    x = np.array([-1, 1, 1, -1, -1, 1, 1, -1], dtype=np.float32) * dx / 2
    y = np.array([-1, -1, 1, 1, -1, -1, 1, 1], dtype=np.float32) * dy / 2
    z = np.array([-1, -1, -1, -1, 1, 1, 1, 1], dtype=np.float32) * dz / 2

    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array(
        [
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1],
        ],
        dtype=np.float32,
    )

    corners = (R @ np.stack([x, y, z])).T
    corners = corners + np.array([cx, cy, cz], dtype=np.float32)
    return corners.astype(np.float32)


def remove_box_edges(server, box_idx):
    """
    /boxes/box_i/edge_j 형태의 edge들을 개별 삭제.
    remove_by_name('/boxes/box_i')가 하위 객체까지 안 지우는 경우를 방지한다.
    """
    for edge_i in range(12):
        try:
            server.scene.remove_by_name(f'/boxes/box_{box_idx}/edge_{edge_i}')
        except Exception:
            pass


def draw_box_lines(server, box_idx, box, color, line_width=3):
    """
    viser에 3D bbox wireframe 그리기.
    기존 edge를 확실히 지운 뒤 다시 그려서 색상 변경이 바로 반영되도록 한다.
    """
    remove_box_edges(server, box_idx)

    corners = bbox_corners(box)

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    for edge_i, (a, b) in enumerate(edges):
        pts = np.stack([corners[a], corners[b]], axis=0).astype(np.float32)
        cols = np.stack([color, color], axis=0).astype(np.float32)

        server.scene.add_line_segments(
            name=f'/boxes/box_{box_idx}/edge_{edge_i}',
            points=pts[None, :, :],
            colors=cols[None, :, :],
            line_width=line_width,
        )


def add_box_label(server, i, box, label_text, highlighted=False):
    """
    viser label 추가.
    position은 np.ndarray shape (3,)로 전달한다.
    """
    try:
        server.scene.remove_by_name(f'/labels/box_{i}')
    except Exception:
        pass

    cx, cy, cz, dx, dy, dz = box[:6]

    label_pos = np.array(
        [
            float(cx),
            float(cy),
            float(cz + dz / 2.0),
        ],
        dtype=np.float32,
    )

    prefix = '★ ' if highlighted else ''
    server.scene.add_label(
        name=f'/labels/box_{i}',
        text=prefix + label_text,
        position=label_pos,
    )


def make_centered_scene(coords, bboxes, mode='floor_center'):
    """
    point cloud와 bbox 중심을 같은 origin 기준으로 이동한다.
    floor_center: x, y는 scene 중심, z는 바닥 높이를 원점으로 둔다.
    """
    coords = coords.astype(np.float32)
    bboxes_vis = bboxes.copy().astype(np.float32)

    if mode == 'mean_center':
        scene_origin = coords.mean(axis=0).astype(np.float32)
    else:
        scene_origin = np.array(
            [
                coords[:, 0].mean(),
                coords[:, 1].mean(),
                coords[:, 2].min(),
            ],
            dtype=np.float32,
        )

    coords_vis = coords - scene_origin
    bboxes_vis[:, :3] = bboxes_vis[:, :3] - scene_origin

    return coords_vis.astype(np.float32), bboxes_vis.astype(np.float32), scene_origin


def main():
    # ── 데이터 로드 ───────────────────────────
    idx = pickle.load(open(INDEX_FILE, 'rb'))

    points = idx['points']
    bboxes = idx['bboxes']
    scores = idx['scores']
    labels = idx['labels']
    classes = idx['classes']
    box_pts = idx['box_pts']
    box_embeds = idx['box_embeds']

    print(f'points: {points.shape}')
    print(f'bboxes: {bboxes.shape}')
    print(f'box_embeds: {box_embeds.shape}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'CLIP device: {device}')

    clip_model = load_clip(device)

    box_embeds_t = torch.tensor(box_embeds, device=device, dtype=torch.float32)
    box_embeds_t = box_embeds_t / box_embeds_t.norm(dim=-1, keepdim=True)

    coords = points[:, :3].astype(np.float32)

    # ── 집/방 scene 원점을 viser 원점에 맞추기 ─────────────
    coords_vis, bboxes_vis, scene_origin = make_centered_scene(
        coords,
        bboxes,
        mode='floor_center',
    )
    print(f'scene_origin: {scene_origin}')

    # 색상 처리
    raw_rgb = points[:, 3:6].astype(np.float32)

    if raw_rgb.min() < 0:
        rgb = (raw_rgb + 1.0) / 2.0
    elif raw_rgb.max() > 1.0:
        rgb = raw_rgb / 255.0
    else:
        rgb = raw_rgb

    rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32)

    # ── viser 서버 ────────────────────────────
    server = viser.ViserServer(port=8080)
    print('viser running at http://localhost:8080')

    pc_handle = server.scene.add_point_cloud(
        name='/scene/points',
        points=coords_vis,
        colors=rgb,
        point_size=0.01,
    )

    # ── GUI 패널 ─────────────────────────────
    with server.gui.add_folder('Query'):
        query_input = server.gui.add_text(
            'Text Query',
            initial_value='sofa',
        )
        topk_slider = server.gui.add_slider(
            'Top-K',
            min=1,
            max=10,
            step=1,
            initial_value=TOPK,
        )
        search_btn = server.gui.add_button('Search 🔍')
        clear_btn = server.gui.add_button('Clear Highlight')
        result_text = server.gui.add_text('Results', initial_value='—')

    with server.gui.add_folder('Display'):
        show_all_boxes = server.gui.add_checkbox(
            'Show all boxes',
            initial_value=True,
        )
        show_labels = server.gui.add_checkbox(
            'Show labels',
            initial_value=True,
        )
        highlight_points = server.gui.add_checkbox(
            'Highlight points in matched boxes',
            initial_value=True,
        )
        pt_size = server.gui.add_slider(
            'Point size',
            min=0.001,
            max=0.05,
            step=0.001,
            initial_value=0.01,
        )

    # ── 상태 ─────────────────────────────────
    state = {
        'top_idx': [],
        'pt_colors': rgb.copy().astype(np.float32),
    }

    def get_label_text(i):
        label_idx = int(labels[i])
        label_name = classes[label_idx] if 0 <= label_idx < len(classes) else f'class_{label_idx}'

        cx, cy, cz = bboxes_vis[i][:3]

        return (
            f'{label_idx}: {label_name} '
            f'score={float(scores[i]):.2f} '
            f'center=({cx:.2f}, {cy:.2f}, {cz:.2f})'
        )

    def redraw_label(i, highlighted=False, sim=None):
        if not show_labels.value:
            try:
                server.scene.remove_by_name(f'/labels/box_{i}')
            except Exception:
                pass
            return

        add_box_label(
            server,
            i,
            bboxes_vis[i],
            get_label_text(i),
            highlighted=highlighted,
        )

    def redraw_box(i, highlighted=False):
        """
        highlighted=True이면 쿼리에 해당하는 bbox를 빨강+두꺼운 선으로 표시.
        highlighted=False이면 일반 bbox를 파랑+얇은 선으로 표시.
        """
        if not show_all_boxes.value and not highlighted:
            remove_box_edges(server, i)
            return

        color = COLOR_MATCH if highlighted else COLOR_OTHERS
        lw = 6 if highlighted else 2

        draw_box_lines(
            server,
            i,
            bboxes_vis[i],
            color=color,
            line_width=lw,
        )

    def redraw_all():
        top_set = set(state['top_idx'])

        for i in range(len(bboxes_vis)):
            highlighted = i in top_set
            redraw_box(i, highlighted=highlighted)
            redraw_label(i, highlighted=highlighted)

    def reset_point_colors():
        state['pt_colors'] = rgb.copy().astype(np.float32)
        pc_handle.colors = state['pt_colors']

    def clear_highlight():
        state['top_idx'] = []
        reset_point_colors()
        result_text.value = '—'
        redraw_all()

    # 초기 bbox와 label 표시
    redraw_all()

    @search_btn.on_click
    def on_search(_):
        text = query_input.value.strip()
        if not text:
            return

        k = int(topk_slider.value)

        # 기존 강조 초기화
        state['top_idx'] = []
        reset_point_colors()
        redraw_all()

        # CLIP 검색
        top_idx, sims = query_clip(
            clip_model,
            device,
            box_embeds_t,
            text,
            k,
        )

        state['top_idx'] = top_idx.tolist()

        # 결과 표시 + bbox 색상 변경
        lines = []

        for rank, i in enumerate(top_idx):
            label_idx = int(labels[i])
            label_name = classes[label_idx] if 0 <= label_idx < len(classes) else f'class_{label_idx}'
            sim = float(sims[i])

            cx, cy, cz = bboxes_vis[i][:3]

            lines.append(
                f'#{rank + 1} box={i} label={label_idx} {label_name} '
                f'score={float(scores[i]):.2f} '
                f'center=({cx:.2f}, {cy:.2f}, {cz:.2f})'
            )

            # 쿼리에 해당하는 bbox는 빨강+두꺼운 선으로 변경
            redraw_box(i, highlighted=True)
            redraw_label(i, highlighted=True)

            # 해당 bbox 안의 point도 노란색으로 강조
            if highlight_points.value:
                pts_i = box_pts[i]
                if len(pts_i) > 0:
                    state['pt_colors'][pts_i] = COLOR_INBOX

        pc_handle.colors = state['pt_colors']
        result_text.value = '\n'.join(lines)

        print(f"Query: '{text}' → {lines}")

    @clear_btn.on_click
    def _(_):
        clear_highlight()

    @show_all_boxes.on_update
    def _(_):
        redraw_all()

    @show_labels.on_update
    def _(_):
        redraw_all()

    @highlight_points.on_update
    def _(_):
        # 포인트 강조 옵션을 바꾸면 현재 top_idx 기준으로 point color만 다시 계산
        reset_point_colors()

        if highlight_points.value:
            for i in state['top_idx']:
                pts_i = box_pts[i]
                if len(pts_i) > 0:
                    state['pt_colors'][pts_i] = COLOR_INBOX

        pc_handle.colors = state['pt_colors']

    @pt_size.on_update
    def _(_):
        pc_handle.point_size = float(pt_size.value)

    # ── 루프 ─────────────────────────────────
    while True:
        time.sleep(0.01)


if __name__ == '__main__':
    main()
