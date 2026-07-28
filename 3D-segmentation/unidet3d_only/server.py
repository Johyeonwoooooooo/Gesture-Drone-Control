"""Standalone UniDet3D viser visualizer — no LLM, no CLIP, no heatmap.

Purpose: validate that UniDet3D detection works end-to-end on a given .bin
before wiring it into the larger LLM webapp.

Usage:
    python 3D-segmentation/unidet3d_only/server.py \
        --bin unidet3d/data/my_scene.bin \
        --dataset scannetpp \
        --device cuda:0 \
        --port 8091

Renders the point cloud + a "Run detection" button. Detected bboxes
are drawn as colored wireframes with `<class> s=<score>` labels.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import viser

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent))
from unidet3d_detector import UniDet3DDetector, DetectionResult  # noqa: E402


_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)


def bbox_corners(box: np.ndarray) -> np.ndarray:
    if len(box) >= 7:
        cx, cy, cz, dx, dy, dz, yaw = (float(v) for v in box[:7])
    else:
        cx, cy, cz, dx, dy, dz = (float(v) for v in box[:6])
        yaw = 0.0
    x = np.array([-1, 1, 1, -1, -1, 1, 1, -1], dtype=np.float32) * dx / 2
    y = np.array([-1, -1, 1, 1, -1, -1, 1, 1], dtype=np.float32) * dy / 2
    z = np.array([-1, -1, -1, -1, 1, 1, 1, 1], dtype=np.float32) * dz / 2
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    corners = (R @ np.stack([x, y, z])).T + np.array([cx, cy, cz], dtype=np.float32)
    return corners.astype(np.float32)


def load_bin(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size % 6 == 0 and raw.size % 9 != 0:
        return raw.reshape(-1, 6)
    if raw.size % 9 == 0:
        return raw.reshape(-1, 9)
    return raw.reshape(-1, 6)


def palette(n: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    cols = rng.uniform(0.3, 1.0, size=(max(n, 1), 3)).astype(np.float32)
    return cols


def main() -> None:
    ap = argparse.ArgumentParser()
    repo = _THIS.parents[2]
    udroot = repo / "unidet3d"
    ap.add_argument("--bin", default=str(udroot / "data" / "my_scene.bin"))
    ap.add_argument("--cfg", default=str(udroot / "configs" /
                    "unidet3d_1xb8_scannet_s3dis_multiscan_3rscan_scannetpp_arkitscenes.py"))
    ap.add_argument("--ckpt", default=str(udroot / "work_dirs" / "unidet3d.pth"))
    ap.add_argument("--unidet3d-root", default=str(udroot))
    ap.add_argument("--dataset", default="scannetpp",
                    help="scannet/s3dis/multiscan/3rscan/scannetpp/arkitscenes")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--max-points", type=int, default=40000)
    args = ap.parse_args()

    bin_path = Path(args.bin)
    if not bin_path.exists():
        raise SystemExit(f"bin not found: {bin_path}")

    pts = load_bin(bin_path)
    coords = pts[:, :3].astype(np.float32)
    rgb_raw = pts[:, 3:6].astype(np.float32)
    if rgb_raw.max() > 1.5:
        rgb = np.clip(rgb_raw / 255.0, 0, 1)
    else:
        rgb = np.clip(rgb_raw, 0, 1)
    ext = coords.max(0) - coords.min(0)
    print(f"[unidet3d-only] scene: N={len(coords)}  ext={ext[0]:.1f}x{ext[1]:.1f}x{ext[2]:.1f} m")

    detector = UniDet3DDetector(
        cfg_path=args.cfg,
        ckpt_path=args.ckpt,
        unidet3d_root=args.unidet3d_root,
        device=args.device,
        dataset_name=args.dataset,
    )

    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.world_axes.visible = True
    server.scene.add_point_cloud(
        name="/points", points=coords, colors=rgb, point_size=0.01,
    )

    state = {"drawn": [], "det": None}

    with server.gui.add_folder("UniDet3D"):
        score_thr = server.gui.add_slider("score_thr", 0.05, 0.9, 0.05, 0.3)
        max_pts = server.gui.add_slider("max_points (subsample)",
                                        5000, 200000, 5000, int(args.max_points))
        run_btn = server.gui.add_button("Run detection")
        clear_btn = server.gui.add_button("Clear boxes")
        status = server.gui.add_markdown("_ready_")

    def clear() -> None:
        for k in state["drawn"]:
            try:
                server.scene.remove_by_name(k)
            except Exception:
                pass
        state["drawn"] = []

    def draw(det: DetectionResult) -> None:
        clear()
        cols = palette(len(det.bboxes))
        for i, box in enumerate(det.bboxes):
            corners = bbox_corners(box)
            color = tuple(float(v) for v in cols[i])
            for ei, (a, b) in enumerate(_EDGES):
                seg = np.stack([corners[a], corners[b]], axis=0)[None]
                ccol = np.array([[color, color]], dtype=np.float32)
                name = f"/boxes/box_{i}/edge_{ei}"
                server.scene.add_line_segments(
                    name=name, points=seg.astype(np.float32),
                    colors=ccol, line_width=3,
                )
                state["drawn"].append(name)
            cls = det.classes[int(det.labels[i])] if 0 <= int(det.labels[i]) < len(det.classes) else f"cls_{det.labels[i]}"
            lname = f"/boxes/box_{i}/label"
            label_pos = (float(box[0]), float(box[1]),
                         float(box[2]) + float(box[5]) / 2 + 0.05)
            server.scene.add_label(
                name=lname,
                text=f"{cls} s={float(det.scores[i]):.2f}",
                position=label_pos,
            )
            state["drawn"].append(lname)

    @run_btn.on_click
    def _(_):
        status.content = "_running detection ..._"
        try:
            t0 = time.time()
            det = detector.detect(
                pts,
                score_thr=float(score_thr.value),
                max_points=int(max_pts.value),
            )
            state["det"] = det
            draw(det)
            lines = [f"### {len(det.bboxes)} boxes in {time.time()-t0:.2f}s"]
            for i, (lbl, s) in enumerate(zip(det.labels, det.scores)):
                cls = det.classes[int(lbl)]
                lines.append(f"- #{i} `{cls}` score={float(s):.2f}")
            status.content = "\n".join(lines[:30])
            print(status.content)
        except Exception as e:
            status.content = f"_error: {e}_"
            raise

    @clear_btn.on_click
    def _(_):
        clear()
        state["det"] = None
        status.content = "_cleared_"

    print(f"[unidet3d-only] running at http://{args.host}:{args.port}")
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
