"""Shared UniDet3D detection backend for the viser webapps.

Lets both `webapp/server.py` and `webapp_llm/server.py` offer a UniDet3D 3D
object-detection backend (bbox + CLIP class match) as an alternative to the
Mosaic3D heatmap/DBSCAN path — running on the *same* currently-loaded cache
scene, selected by a "Backend" dropdown.

Design notes
------------
* The heavy mmdet3d / MinkowskiEngine deps stay lazy: `UniDet3DDetector` only
  imports them on the first `detect()`. Importing *this* module is therefore
  safe even from the `mosaic3d` conda env; only *running* a detection there
  fails, and that failure is caught and surfaced as a UI status message.
* UniDet3D's `detect()` accepts an `(N, 6)` xyz+rgb cloud (it ignores normals),
  so we feed the already-loaded `asset.coord` (display/centered frame) plus
  `asset.vertex_colors` rescaled to `[-1, 1]` exactly like `miny-det/convert.py`
  (`color / 127.5 - 1`). No raw .npy reload or normals are needed, and it works
  for both single-room and whole-building views.
* Because the input coords are in the centered display frame, detection bboxes
  come back in that frame too — render them directly, and report world coords as
  `box_center + asset.center` (the same contract the Mosaic3D path uses).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch

# `unidet3d_only` is a sibling namespace package under 3D-segmentation; the
# servers put that dir on sys.path before importing this module.
from unidet3d_only.unidet3d_detector import (  # noqa: E402
    UniDet3DDetector,
    build_class_embeds,
    topk_boxes_for_query,
)

COLOR_OTHERS: Tuple[int, int, int] = (80, 150, 255)   # non-matched boxes: blue
COLOR_MATCH: Tuple[int, int, int] = (255, 30, 30)     # query-matched boxes: red

_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # bottom
    (4, 5), (5, 6), (6, 7), (7, 4),  # top
    (0, 4), (1, 5), (2, 6), (3, 7),  # verticals
]


# --------------------------------------------------------------------------- #
# CLI wiring
# --------------------------------------------------------------------------- #
def add_unidet3d_args(ap) -> None:
    """Register the `--enable-unidet3d` + `--unidet3d-*` flags on an argparser.

    Defaults assume the `unidet3d/` git submodule is checked out at the repo
    root (same defaults the removed webapp_llm panel used).
    """
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]  # .../Gesture-Drone-Control
    unidet_root = repo_root / "unidet3d"
    ap.add_argument(
        "--enable-unidet3d", action="store_true",
        help="Offer a UniDet3D detection backend (launch from the `unidet3d` env).",
    )
    ap.add_argument("--unidet3d-root", default=str(unidet_root),
                    help="UniDet3D research repo root (added to sys.path).")
    ap.add_argument(
        "--unidet3d-cfg",
        default=str(unidet_root / "configs"
                    / "unidet3d_1xb8_scannet_s3dis_multiscan_3rscan_"
                      "scannetpp_arkitscenes.py"),
    )
    ap.add_argument("--unidet3d-ckpt",
                    default=str(unidet_root / "work_dirs" / "unidet3d.pth"))
    ap.add_argument(
        "--unidet3d-dataset", default="scannetpp",
        help="Decoder head: scannet/s3dis/multiscan/3rscan/scannetpp/arkitscenes.",
    )
    ap.add_argument("--unidet3d-device", default="cuda:0")
    ap.add_argument("--unidet3d-score-thr", type=float, default=0.30)


def make_detector(args) -> Optional[UniDet3DDetector]:
    """Construct a `UniDet3DDetector` and warm it up, or `None` if disabled/failed.

    The model is loaded HERE, in the caller's (main) thread, on purpose:
    `mmengine.Config.fromfile` walks `pkg_resources.working_set`, which is `None`
    when first touched from a viser GUI callback *worker thread* (the crash seen
    when detection was triggered lazily). Loading on the main thread — exactly
    like `miny-det/infer.py` — initialises that global once for all threads, so
    the worker-thread `detect()` only runs the cached forward pass. It also makes
    a bad env / missing checkpoint fail loudly at startup, and lets the caller
    drop the `unidet3d` dropdown option when the model can't load.
    """
    if not getattr(args, "enable_unidet3d", False):
        return None
    det = UniDet3DDetector(
        cfg_path=args.unidet3d_cfg,
        ckpt_path=args.unidet3d_ckpt,
        unidet3d_root=args.unidet3d_root,
        device=args.unidet3d_device,
        dataset_name=args.unidet3d_dataset,
    )
    try:
        det._ensure_loaded()  # main-thread warmup (see docstring)
    except Exception as e:  # noqa: BLE001 — surface any load failure, then disable
        print(f"[unidet3d] backend disabled — model load failed: "
              f"{type(e).__name__}: {e}")
        return None
    return det


# --------------------------------------------------------------------------- #
# Detection on the loaded scene
# --------------------------------------------------------------------------- #
@dataclass
class UniDet3DSession:
    """Per-server cache of the most recent detection + box embeddings."""
    det: object = None                                # DetectionResult or None
    box_class_embeds: Optional[torch.Tensor] = None   # (M, D) class embed per box
    scene_key: Optional[str] = None                   # which scene was detected
    handles: list = field(default_factory=list)       # viser handles to clear


def scene_points6(asset) -> np.ndarray:
    """Build an (N, 6) xyz+rgb cloud from a loaded RegionAssets.

    Coords are the centered/display frame; colors are rescaled to [-1, 1] to
    match the .bin format the UniDet3D checkpoint was fed (miny-det/convert.py).
    """
    coord = np.asarray(asset.coord, dtype=np.float32)
    vc = getattr(asset, "vertex_colors", None)
    if vc is not None:
        rgb = np.asarray(vc, dtype=np.float32)[:, :3]
    else:
        rgb = np.full((len(coord), 3), 200.0, dtype=np.float32)  # gray fallback
    rgb = rgb / 127.5 - 1.0
    return np.concatenate([coord, rgb], axis=1).astype(np.float32)


def detect_scene(detector, text_encoder, asset, score_thr: float):
    """Run UniDet3D on the loaded scene and build per-box CLIP class embeds.

    Returns (DetectionResult, box_class_embeds[(M, D)]).
    Raises RuntimeError (friendly message) if the env lacks the mmdet3d stack.
    """
    pts = scene_points6(asset)
    try:
        det = detector.detect(pts, score_thr=float(score_thr))
    except (ImportError, ModuleNotFoundError, OSError) as e:
        raise RuntimeError(
            "UniDet3D unavailable in this environment — launch the server from "
            f"the `unidet3d` conda env. ({type(e).__name__}: {e})"
        ) from e

    class_embeds = build_class_embeds(det.classes, text_encoder)  # (C, D)
    if len(det.labels) > 0:
        box_class_embeds = class_embeds[np.asarray(det.labels)]   # (M, D)
    else:
        box_class_embeds = class_embeds.new_zeros((0, class_embeds.shape[1]))
    return det, box_class_embeds


def match_boxes(det, box_class_embeds, text_encoder, clip_prompt: str, topk: int):
    """CLIP-match a text prompt against detected box class embeddings.

    Returns (top_idx[np.ndarray], sims[np.ndarray] or None).
    """
    if det is None or len(det.bboxes) == 0:
        return np.array([], dtype=int), None
    qf = text_encoder.encode([clip_prompt])[0]  # (D,) normalized
    order, sims = topk_boxes_for_query(qf, box_class_embeds, int(topk))
    return np.asarray(order, dtype=int), sims


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _box_corners(box: np.ndarray) -> np.ndarray:
    """(8, 3) corners for a [cx,cy,cz,dx,dy,dz(,yaw)] box (yaw-aware)."""
    if len(box) >= 7:
        cx, cy, cz, dx, dy, dz, yaw = (float(v) for v in box[:7])
    else:
        cx, cy, cz, dx, dy, dz = (float(v) for v in box[:6])
        yaw = 0.0
    x = np.array([-1, 1, 1, -1, -1, 1, 1, -1], np.float32) * dx / 2
    y = np.array([-1, -1, 1, 1, -1, -1, 1, 1], np.float32) * dy / 2
    z = np.array([-1, -1, -1, -1, 1, 1, 1, 1], np.float32) * dz / 2
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float32)
    corners = (R @ np.stack([x, y, z])).T + np.array([cx, cy, cz], np.float32)
    return corners.astype(np.float32)


def _box_segments(box: np.ndarray) -> np.ndarray:
    cor = _box_corners(box)
    return np.stack([np.stack([cor[a], cor[b]]) for a, b in _EDGES], axis=0)  # (12,2,3)


def clear_boxes(session: UniDet3DSession) -> None:
    for h in session.handles:
        try:
            h.remove()
        except Exception:
            pass
    session.handles = []


def render_boxes(
    server,
    session: UniDet3DSession,
    det,
    *,
    show_all: bool = True,
    highlight_idx=(),
) -> None:
    """Draw detected boxes; matched ones in red with a ``★ <class>`` label."""
    clear_boxes(session)
    if det is None:
        return
    top = {int(i) for i in highlight_idx}
    handles = []
    for i, box in enumerate(det.bboxes):
        is_top = i in top
        if not show_all and not is_top:
            continue
        color = COLOR_MATCH if is_top else COLOR_OTHERS
        handles.append(server.scene.add_line_segments(
            f"/unidet3d/box_{i}",
            points=_box_segments(box),
            colors=color,
            line_width=5.0 if is_top else 1.5,
        ))
        if is_top:
            li = int(det.labels[i])
            cls = det.classes[li] if 0 <= li < len(det.classes) else f"class_{li}"
            cx, cy = float(box[0]), float(box[1])
            top_z = float(box[2]) + float(box[5]) / 2.0 + 0.15
            handles.append(server.scene.add_label(
                f"/unidet3d/label_{i}", text=f"★ {cls}", position=(cx, cy, top_z),
            ))
    session.handles = handles


def world_center(box: np.ndarray, asset_center: np.ndarray) -> np.ndarray:
    """Box center (display frame) → world frame for the path planner."""
    return np.asarray(box[:3], dtype=np.float32) + np.asarray(asset_center, dtype=np.float32)
