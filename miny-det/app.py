# app_query_highlight.py — UniDet3D + CLIP 쿼리 뷰어 (씬 드롭다운으로 선택)
import os
import numpy as np
import pickle
import torch
import clip
import viser
import time

import scenes

TOPK = 3

# ── 색상 팔레트 ──────────────────────────────
COLOR_INBOX  = np.array([1.0, 0.8, 0.2], dtype=np.float32)  # bbox 내부 point: 노랑
COLOR_MATCH  = np.array([1.0, 0.1, 0.1], dtype=np.float32)  # 쿼리 매칭 bbox: 빨강
COLOR_OTHERS = np.array([0.3, 0.6, 1.0], dtype=np.float32)  # 일반 bbox: 파랑


def load_clip(device):
    model, _ = clip.load('ViT-B/32', device=device)
    model.eval()
    return model


def query_clip(model, device, box_embeds_t, text, topk):
    tokens = clip.tokenize([text]).to(device)

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


def load_scene_data(region, device):
    """clip_index .pkl 로드 → 렌더에 필요한 배열들로 정리해서 dict 반환."""
    idx = pickle.load(open(scenes.index_path(region), 'rb'))

    points = idx['points']
    bboxes = idx['bboxes']
    box_embeds = idx['box_embeds']

    coords = points[:, :3].astype(np.float32)
    coords_vis, bboxes_vis, scene_origin = make_centered_scene(
        coords, bboxes, mode='floor_center')

    raw_rgb = points[:, 3:6].astype(np.float32)
    if raw_rgb.min() < 0:
        rgb = (raw_rgb + 1.0) / 2.0
    elif raw_rgb.max() > 1.0:
        rgb = raw_rgb / 255.0
    else:
        rgb = raw_rgb
    rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32)

    if len(box_embeds) > 0:
        box_embeds_t = torch.tensor(box_embeds, device=device, dtype=torch.float32)
        box_embeds_t = box_embeds_t / box_embeds_t.norm(dim=-1, keepdim=True)
    else:
        box_embeds_t = torch.zeros((0, 512), device=device, dtype=torch.float32)

    return dict(
        region=region,
        points=points,
        coords_vis=coords_vis,
        bboxes_vis=bboxes_vis,
        scene_origin=scene_origin,
        rgb=rgb,
        scores=idx['scores'],
        labels=idx['labels'],
        classes=idx['classes'],
        box_pts=idx['box_pts'],
        box_embeds_t=box_embeds_t,
        n_boxes=len(bboxes_vis),
    )


def main():
    regions = scenes.available_regions()
    if not regions:
        raise SystemExit(
            f'전처리된 씬이 없습니다 ({scenes.DATA_DIR}/det_*_clip_index.pkl). '
            'convert.py → infer.py → clip_index.py 를 먼저 실행하세요.'
        )
    print(f'사용 가능한 씬 {len(regions)}개')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'CLIP device: {device}')
    clip_model = load_clip(device)

    # ── 씬 데이터 상태 (드롭다운으로 교체) ──────────────
    S = load_scene_data(regions[0], device)

    # ── viser 서버 ────────────────────────────
    server = viser.ViserServer(port=8081)
    print('viser running at http://localhost:8081')

    pc_handle = server.scene.add_point_cloud(
        name='/scene/points',
        points=S['coords_vis'],
        colors=S['rgb'],
        point_size=0.01,
    )

    # ── GUI 패널 ─────────────────────────────
    with server.gui.add_folder('Scene'):
        scene_dropdown = server.gui.add_dropdown(
            'Scene',
            options=regions,
            initial_value=regions[0],
        )
        scene_info = server.gui.add_text('Info', initial_value='—')

    with server.gui.add_folder('Query'):
        query_input = server.gui.add_text('Text Query', initial_value='sofa')
        topk_slider = server.gui.add_slider(
            'Top-K', min=1, max=10, step=1, initial_value=TOPK)
        search_btn = server.gui.add_button('Search 🔍')
        clear_btn = server.gui.add_button('Clear Highlight')
        result_text = server.gui.add_text('Results', initial_value='—')

    with server.gui.add_folder('Display'):
        show_all_boxes = server.gui.add_checkbox('Show all boxes', initial_value=True)
        show_labels = server.gui.add_checkbox('Show labels', initial_value=True)
        highlight_points = server.gui.add_checkbox(
            'Highlight points in matched boxes', initial_value=True)
        pt_size = server.gui.add_slider(
            'Point size', min=0.001, max=0.05, step=0.001, initial_value=0.01)
        score_thr = server.gui.add_slider(
            'Score threshold', min=0.0, max=1.0, step=0.01, initial_value=0.0)

    # ── 상태 ─────────────────────────────────
    state = {
        'top_idx': [],
        'pt_colors': S['rgb'].copy().astype(np.float32),
        'max_drawn_boxes': S['n_boxes'],  # 씬 전환 시 지울 박스 개수 추적
    }

    def get_label_text(i):
        labels, classes, scores, bboxes_vis = (
            S['labels'], S['classes'], S['scores'], S['bboxes_vis'])
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
        add_box_label(server, i, S['bboxes_vis'][i], get_label_text(i),
                      highlighted=highlighted)

    def redraw_box(i, highlighted=False):
        if float(S['scores'][i]) < score_thr.value:
            remove_box_edges(server, i)
            return
        if not show_all_boxes.value and not highlighted:
            remove_box_edges(server, i)
            return
        color = COLOR_MATCH if highlighted else COLOR_OTHERS
        lw = 6 if highlighted else 2
        draw_box_lines(server, i, S['bboxes_vis'][i], color=color, line_width=lw)

    def redraw_all():
        top_set = set(state['top_idx'])
        for i in range(S['n_boxes']):
            highlighted = i in top_set
            redraw_box(i, highlighted=highlighted)
            redraw_label(i, highlighted=highlighted)

    def reset_point_colors():
        state['pt_colors'] = S['rgb'].copy().astype(np.float32)
        pc_handle.colors = state['pt_colors']

    def clear_highlight():
        state['top_idx'] = []
        reset_point_colors()
        result_text.value = '—'
        redraw_all()

    def clear_scene_graph():
        """현재까지 그린 모든 박스 edge / label 제거 (씬 전환용)."""
        for i in range(state['max_drawn_boxes']):
            remove_box_edges(server, i)
            try:
                server.scene.remove_by_name(f'/labels/box_{i}')
            except Exception:
                pass

    def load_scene(region):
        nonlocal S
        clear_scene_graph()
        S = load_scene_data(region, device)
        state['top_idx'] = []
        state['pt_colors'] = S['rgb'].copy().astype(np.float32)
        state['max_drawn_boxes'] = max(state['max_drawn_boxes'], S['n_boxes'])
        pc_handle.points = S['coords_vis']
        pc_handle.colors = state['pt_colors']
        result_text.value = '—'
        scene_info.value = f"{region} | pts={len(S['points'])} boxes={S['n_boxes']}"
        redraw_all()
        print(f"[scene] loaded {region}: {len(S['points'])} pts, {S['n_boxes']} boxes")

    # 초기 표시
    scene_info.value = f"{S['region']} | pts={len(S['points'])} boxes={S['n_boxes']}"
    redraw_all()

    @scene_dropdown.on_update
    def _(_):
        load_scene(scene_dropdown.value)

    @search_btn.on_click
    def on_search(_):
        text = query_input.value.strip()
        if not text:
            return
        if S['n_boxes'] == 0:
            result_text.value = '(이 씬에 박스 없음)'
            return

        k = int(topk_slider.value)
        state['top_idx'] = []
        reset_point_colors()
        redraw_all()

        top_idx, sims = query_clip(clip_model, device, S['box_embeds_t'], text, k)
        state['top_idx'] = top_idx.tolist()

        labels, classes, scores, bboxes_vis, box_pts = (
            S['labels'], S['classes'], S['scores'], S['bboxes_vis'], S['box_pts'])
        lines = []
        for rank, i in enumerate(top_idx):
            label_idx = int(labels[i])
            label_name = classes[label_idx] if 0 <= label_idx < len(classes) else f'class_{label_idx}'
            cx, cy, cz = bboxes_vis[i][:3]
            lines.append(
                f'#{rank + 1} box={i} label={label_idx} {label_name} '
                f'score={float(scores[i]):.2f} '
                f'center=({cx:.2f}, {cy:.2f}, {cz:.2f})'
            )
            redraw_box(i, highlighted=True)
            redraw_label(i, highlighted=True)
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
        reset_point_colors()
        if highlight_points.value:
            for i in state['top_idx']:
                pts_i = S['box_pts'][i]
                if len(pts_i) > 0:
                    state['pt_colors'][pts_i] = COLOR_INBOX
        pc_handle.colors = state['pt_colors']

    @pt_size.on_update
    def _(_):
        pc_handle.point_size = float(pt_size.value)

    @score_thr.on_update
    def _(_):
        redraw_all()

    # ── 루프 ─────────────────────────────────
    while True:
        time.sleep(0.01)


if __name__ == '__main__':
    main()
