from __future__ import annotations

import argparse
import json
import pickle
import time
from dataclasses import asdict

import numpy as np
import viser

from local_llm_intent import LocalLLMParser


# ============================================================
# Color palette
# ============================================================

COLOR_INBOX = np.array([1.0, 0.8, 0.2], dtype=np.float32)   # matched bbox 내부 point: yellow
COLOR_MATCH = np.array([1.0, 0.1, 0.1], dtype=np.float32)   # matched bbox: red
COLOR_OTHERS = np.array([0.3, 0.6, 1.0], dtype=np.float32)  # normal bbox: blue


# ============================================================
# Label matching
# ============================================================

LABEL_SYNONYMS = {
    "toilet": ["toilet", "bathroom", "wc", "commode"],
    "sofa": ["sofa", "couch"],
    "chair": ["chair", "seat"],
    "desk": ["desk"],
    "table": ["table", "dining table"],
    "bed": ["bed"],
    "refrigerator": ["refrigerator", "fridge"],
    "tv": ["tv", "television"],
    "monitor": ["monitor", "screen"],
    "door": ["door"],
    "window": ["window"],
    "sink": ["sink", "basin"],
    "microwave": ["microwave"],
    "oven": ["oven"],
    "cabinet": ["cabinet", "cupboard"],
    "wardrobe": ["wardrobe", "closet"],
    "stairs": ["stairs", "stair"],
    "plant": ["plant", "potted plant"],
    "lamp": ["lamp", "light"],
    "trash can": ["trash can", "bin"],
    "bag": ["bag", "backpack"],
    "laptop": ["laptop", "notebook"],
}


def normalize_name(text: str) -> str:
    return (
        str(text)
        .lower()
        .replace("_", " ")
        .replace("-", " ")
        .replace("/", " ")
        .strip()
    )


def get_label_name(labels, classes, i: int) -> str:
    label_idx = int(labels[i])

    if 0 <= label_idx < len(classes):
        return str(classes[label_idx])

    return f"class_{label_idx}"


def label_match_score(target_object: str, label_name: str) -> float:
    """
    CLIP 없이 Local LLM이 뽑은 target_object와 detector label 이름만으로 점수 계산.
    """
    target = normalize_name(target_object)
    label = normalize_name(label_name)

    if not target or target == "object":
        return 0.0

    candidates = [target]
    candidates.extend(LABEL_SYNONYMS.get(target, []))

    best = 0.0

    for cand in candidates:
        cand = normalize_name(cand)

        # 완전 일치
        if cand == label:
            best = max(best, 1.0)

        # 포함 관계
        if cand in label or label in cand:
            best = max(best, 0.85)

        # 단어 단위 겹침
        cand_words = set(cand.split())
        label_words = set(label.split())

        if cand_words and label_words:
            overlap = len(cand_words & label_words)
            union = len(cand_words | label_words)
            jaccard = overlap / union

            if jaccard > 0:
                best = max(best, 0.45 + 0.4 * jaccard)

    return float(best)


def match_detected_boxes_by_label(
    *,
    bboxes_world: np.ndarray,
    bboxes_vis: np.ndarray,
    labels,
    classes,
    scores,
    intent: dict,
    topk: int,
):
    """
    자연어 intent에서 나온 target_object를 탐지 label과 매칭하고,
    Top-K bbox의 world center / bbox / score를 반환한다.
    """
    target_object = intent["target_object"]

    final_scores = []
    match_scores = []

    for i in range(len(bboxes_world)):
        label_name = get_label_name(labels, classes, i)

        m_score = label_match_score(target_object, label_name)
        det_score = float(scores[i])

        # label match가 핵심이고 detector confidence는 보조로만 반영
        final_score = m_score + 0.1 * det_score

        match_scores.append(m_score)
        final_scores.append(final_score)

    match_scores = np.array(match_scores, dtype=np.float32)
    final_scores = np.array(final_scores, dtype=np.float32)

    valid_idx = np.where(match_scores > 0)[0]

    if len(valid_idx) == 0:
        print(f"[warn] No label matched target_object='{target_object}'. Fallback to detector score top-k.")
        score_arr = np.array(scores, dtype=np.float32)
        top_idx = np.argsort(score_arr)[::-1][:topk]
    else:
        top_idx = valid_idx[np.argsort(final_scores[valid_idx])[::-1][:topk]]

    candidates = []

    for rank, i in enumerate(top_idx, start=1):
        label_idx = int(labels[i])
        label_name = get_label_name(labels, classes, i)

        world_center = bboxes_world[i][:3].astype(float).tolist()
        vis_center = bboxes_vis[i][:3].astype(float).tolist()
        bbox_world = bboxes_world[i].astype(float).tolist()

        candidates.append(
            {
                "rank": rank,
                "box_id": int(i),
                "label_id": label_idx,
                "label_name": label_name,
                "target_object": target_object,
                "world_center": {
                    "x": float(world_center[0]),
                    "y": float(world_center[1]),
                    "z": float(world_center[2]),
                },
                "vis_center": {
                    "x": float(vis_center[0]),
                    "y": float(vis_center[1]),
                    "z": float(vis_center[2]),
                },
                "bbox_world": bbox_world,
                "det_score": float(scores[i]),
                "label_match_score": float(match_scores[i]),
                "final_score": float(final_scores[i]),
            }
        )

    return top_idx.tolist(), candidates


# ============================================================
# Geometry / visualization
# ============================================================

def bbox_corners(box: np.ndarray) -> np.ndarray:
    """
    box: [cx, cy, cz, dx, dy, dz, yaw] or [cx, cy, cz, dx, dy, dz]
    return: [8, 3]
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


def remove_box_edges(server: viser.ViserServer, box_idx: int):
    for edge_i in range(12):
        try:
            server.scene.remove_by_name(f"/boxes/box_{box_idx}/edge_{edge_i}")
        except Exception:
            pass


def draw_box_lines(
    server: viser.ViserServer,
    box_idx: int,
    box: np.ndarray,
    color: np.ndarray,
    line_width: int = 3,
):
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
            name=f"/boxes/box_{box_idx}/edge_{edge_i}",
            points=pts[None, :, :],
            colors=cols[None, :, :],
            line_width=line_width,
        )


def add_box_label(
    server: viser.ViserServer,
    i: int,
    box: np.ndarray,
    label_text: str,
    highlighted: bool = False,
):
    try:
        server.scene.remove_by_name(f"/labels/box_{i}")
    except Exception:
        pass

    cx, cy, cz, dx, dy, dz = box[:6]

    label_pos = np.array(
        [float(cx), float(cy), float(cz + dz / 2.0)],
        dtype=np.float32,
    )

    prefix = "★ " if highlighted else ""

    server.scene.add_label(
        name=f"/labels/box_{i}",
        text=prefix + label_text,
        position=label_pos,
    )


def make_centered_scene(
    coords: np.ndarray,
    bboxes: np.ndarray,
    mode: str = "floor_center",
):
    """
    viser에서 보기 좋게 scene origin을 옮긴다.
    반환 좌표:
    - coords_vis, bboxes_vis: 시각화용 좌표
    - scene_origin: 원래 world 좌표와의 차이
    """
    coords = coords.astype(np.float32)
    bboxes_vis = bboxes.copy().astype(np.float32)

    if mode == "mean_center":
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


def normalize_rgb(points: np.ndarray) -> np.ndarray:
    raw_rgb = points[:, 3:6].astype(np.float32)

    if raw_rgb.min() < 0:
        rgb = (raw_rgb + 1.0) / 2.0
    elif raw_rgb.max() > 1.0:
        rgb = raw_rgb / 255.0
    else:
        rgb = raw_rgb

    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


# ============================================================
# Intent
# ============================================================

def parsed_intent_to_dict(parsed) -> dict:
    """
    local_llm_intent.py의 ParsedIntent dataclass를 dict로 변환.
    """
    if hasattr(parsed, "__dataclass_fields__"):
        data = asdict(parsed)
    else:
        data = dict(parsed)

    target = str(data.get("target_object", "")).strip().lower() or "object"
    clip_prompt = str(data.get("clip_prompt", "")).strip() or f"a {target}"

    return {
        "target_object": target,
        "clip_prompt": clip_prompt,
        "location_hint": str(data.get("location_hint", "")).strip(),
        "action": str(data.get("action", "other")).strip(),
        "return_home": bool(data.get("return_home", False)),
        "raw": data.get("raw", {}),
        "raw_text": data.get("raw_text", ""),
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--index-file",
        type=str,
        default="/shareHost/minyoy/project/data/00809_Qpor2mEya8F_000_002/clip_index.pkl",
    )
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--port", type=int, default=8080)

    parser.add_argument("--llm-model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--llm-device", type=str, default="cuda:0")
    parser.add_argument("--llm-dtype", type=str, default="float16")

    args = parser.parse_args()

    # --------------------------------------------------------
    # Load index
    # --------------------------------------------------------
    idx = pickle.load(open(args.index_file, "rb"))

    points = idx["points"]
    bboxes = idx["bboxes"]
    scores = idx["scores"]
    labels = idx["labels"]
    classes = idx["classes"]
    box_pts = idx["box_pts"]

    print(f"[data] points: {points.shape}")
    print(f"[data] bboxes: {bboxes.shape}")
    print(f"[data] labels: {len(labels)}")
    print(f"[data] classes: {len(classes)}")

    coords = points[:, :3].astype(np.float32)
    rgb = normalize_rgb(points)

    coords_vis, bboxes_vis, scene_origin = make_centered_scene(
        coords,
        bboxes,
        mode="floor_center",
    )

    print(f"[data] scene_origin: {scene_origin}")

    # --------------------------------------------------------
    # Load local LLM intent parser
    # --------------------------------------------------------
    print(f"[llm] loading LocalLLMParser: {args.llm_model}")

    intent_parser = LocalLLMParser(
        model_id=args.llm_model,
        device=args.llm_device,
        dtype=args.llm_dtype,
    )

    # --------------------------------------------------------
    # Viser server
    # --------------------------------------------------------
    server = viser.ViserServer(port=args.port)
    print(f"[viser] running at http://localhost:{args.port}")

    pc_handle = server.scene.add_point_cloud(
        name="/scene/points",
        points=coords_vis,
        colors=rgb,
        point_size=0.01,
    )

    # --------------------------------------------------------
    # GUI
    # --------------------------------------------------------
    with server.gui.add_folder("Natural Language Command"):
        command_input = server.gui.add_text(
            "Command",
            initial_value="위층 방의 화장실 사진 촬영해줘",
        )
        run_btn = server.gui.add_button("Parse & Search 🔍")
        clear_btn = server.gui.add_button("Clear Highlight")
        intent_text = server.gui.add_text("Parsed Intent JSON", initial_value="—")
        result_text = server.gui.add_text("Top-K 3D Candidates", initial_value="—")

    with server.gui.add_folder("Search Settings"):
        topk_slider = server.gui.add_slider(
            "Top-K",
            min=1,
            max=10,
            step=1,
            initial_value=args.topk,
        )

    with server.gui.add_folder("Display"):
        show_all_boxes = server.gui.add_checkbox(
            "Show all boxes",
            initial_value=True,
        )
        show_labels = server.gui.add_checkbox(
            "Show labels",
            initial_value=True,
        )
        highlight_points = server.gui.add_checkbox(
            "Highlight points in matched boxes",
            initial_value=True,
        )
        pt_size = server.gui.add_slider(
            "Point size",
            min=0.001,
            max=0.05,
            step=0.001,
            initial_value=0.01,
        )

    # --------------------------------------------------------
    # State
    # --------------------------------------------------------
    state = {
        "top_idx": [],
        "pt_colors": rgb.copy().astype(np.float32),
        "last_intent": None,
        "last_candidates": [],
    }

    def candidate_by_box_id(box_id: int):
        for cand in state["last_candidates"]:
            if cand["box_id"] == int(box_id):
                return cand
        return None

    def get_label_text(i: int, candidate=None) -> str:
        label_idx = int(labels[i])
        label_name = get_label_name(labels, classes, i)

        cx, cy, cz = bboxes_vis[i][:3]

        text = (
            f"{label_idx}: {label_name} "
            f"det={float(scores[i]):.2f} "
            f"center=({cx:.2f}, {cy:.2f}, {cz:.2f})"
        )

        if candidate is not None:
            text += (
                f" final={candidate['final_score']:.3f}"
                f" label_match={candidate['label_match_score']:.2f}"
            )

        return text

    def redraw_label(i: int, highlighted: bool = False, candidate=None):
        if not show_labels.value:
            try:
                server.scene.remove_by_name(f"/labels/box_{i}")
            except Exception:
                pass
            return

        add_box_label(
            server,
            i,
            bboxes_vis[i],
            get_label_text(i, candidate=candidate),
            highlighted=highlighted,
        )

    def redraw_box(i: int, highlighted: bool = False):
        if not show_all_boxes.value and not highlighted:
            remove_box_edges(server, i)
            return

        color = COLOR_MATCH if highlighted else COLOR_OTHERS
        line_width = 6 if highlighted else 2

        draw_box_lines(
            server,
            i,
            bboxes_vis[i],
            color=color,
            line_width=line_width,
        )

    def redraw_all():
        top_set = set(state["top_idx"])

        for i in range(len(bboxes_vis)):
            highlighted = i in top_set
            cand = candidate_by_box_id(i) if highlighted else None

            redraw_box(i, highlighted=highlighted)
            redraw_label(i, highlighted=highlighted, candidate=cand)

    def reset_point_colors():
        state["pt_colors"] = rgb.copy().astype(np.float32)
        pc_handle.colors = state["pt_colors"]

    def recolor_matched_points():
        reset_point_colors()

        if highlight_points.value:
            for i in state["top_idx"]:
                pts_i = box_pts[i]

                if len(pts_i) > 0:
                    state["pt_colors"][pts_i] = COLOR_INBOX

        pc_handle.colors = state["pt_colors"]

    def clear_highlight():
        state["top_idx"] = []
        state["last_intent"] = None
        state["last_candidates"] = []

        reset_point_colors()

        intent_text.value = "—"
        result_text.value = "—"

        redraw_all()

    def format_candidates(candidates: list[dict]) -> str:
        lines = []

        for cand in candidates:
            wc = cand["world_center"]

            lines.append(
                f"#{cand['rank']} "
                f"box={cand['box_id']} "
                f"label={cand['label_name']} "
                f"world=({wc['x']:.2f}, {wc['y']:.2f}, {wc['z']:.2f}) "
                f"final={cand['final_score']:.3f} "
                f"label_match={cand['label_match_score']:.2f}"
            )

        return "\n".join(lines)

    # 초기 bbox와 label 표시
    redraw_all()

    # --------------------------------------------------------
    # GUI callbacks
    # --------------------------------------------------------

    @run_btn.on_click
    def on_run(_):
        user_text = command_input.value.strip()

        if not user_text:
            return

        print(f"\n[user] {user_text}")

        # 1. 자연어 입력 → Local LLM 기반 목표 객체/행동 추출
        parsed = intent_parser.parse(user_text)
        intent = parsed_intent_to_dict(parsed)

        print("[intent]")
        print(json.dumps(intent, ensure_ascii=False, indent=2))

        # 2. 탐지 라벨과 매칭 → Top-K 3D 좌표 반환
        top_idx, candidates = match_detected_boxes_by_label(
            bboxes_world=bboxes,
            bboxes_vis=bboxes_vis,
            labels=labels,
            classes=classes,
            scores=scores,
            intent=intent,
            topk=int(topk_slider.value),
        )

        state["top_idx"] = top_idx
        state["last_intent"] = intent
        state["last_candidates"] = candidates

        # 3. GUI 출력
        intent_text.value = json.dumps(
            {
                "target_object": intent["target_object"],
                "clip_prompt": intent["clip_prompt"],
                "location_hint": intent["location_hint"],
                "action": intent["action"],
                "return_home": intent["return_home"],
            },
            ensure_ascii=False,
            indent=2,
        )

        result_text.value = format_candidates(candidates)

        # 4. 선택 객체 위치 시각화
        recolor_matched_points()
        redraw_all()

        output = {
            "input_command": user_text,
            "intent": {
                "target_object": intent["target_object"],
                "clip_prompt": intent["clip_prompt"],
                "location_hint": intent["location_hint"],
                "action": intent["action"],
                "return_home": intent["return_home"],
            },
            "topk_candidates": candidates,
        }

        print("[result]")
        print(json.dumps(output, ensure_ascii=False, indent=2))

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
        recolor_matched_points()

    @pt_size.on_update
    def _(_):
        pc_handle.point_size = float(pt_size.value)

    # --------------------------------------------------------
    # Keep server alive
    # --------------------------------------------------------
    while True:
        time.sleep(0.01)


if __name__ == "__main__":
    main()