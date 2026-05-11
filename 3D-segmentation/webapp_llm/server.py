"""LLM-driven 3D segmentation webapp (separate from `../webapp/server.py`).

Pipeline per query:
    natural-language command  ->  local LLM (intent parsing)
                              ->  CLIP text encode (target_object)
                              ->  per-point cosine sim heatmap
                              ->  DBSCAN clustering -> ranked 3D candidates
                              ->  viser visualization (heatmap + markers + bbox)

This file imports building/region loading + clustering helpers from the
sibling `webapp/server.py` to avoid duplication; the original webapp is
not modified.

Run:
    python 3D-segmentation/webapp_llm/server.py \
        --port 8090 \
        --llm-model Qwen/Qwen2.5-3B-Instruct \
        --llm-device cuda:1 \
        --clip-device cuda:0
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import viser

# Make sibling packages importable.
_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parents[1]))  # .../3D-segmentation
sys.path.insert(0, str(_THIS.parents[1] / "webapp"))  # for direct import

from inference.cluster_candidates import ClusterParams, candidates_from_heatmap  # noqa: E402
from webapp.server import (  # noqa: E402
    CLIP_MODEL_ID,
    RegionAssets,
    TextEncoder,
    _bbox_edge_segments,
    blend_with_rgb,
    cluster_palette,
    heatmap_colors,
    list_buildings,
    list_regions,
    load_building,
    load_region,
    query_single,
    regions_for_building,
    upload_points,
)

from webapp_llm.llm_parser import LocalLLMParser, ParsedIntent  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--feat-dir",
        default=str(_THIS.parents[1] / "cache" / "feat"),
    )
    ap.add_argument(
        "--match-dir",
        default=str(_THIS.parents[1] / "cache" / "match"),
    )
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--clip-device", default="cuda:0")
    ap.add_argument("--llm-model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--llm-device", default="cuda:1")
    ap.add_argument("--llm-dtype", default="float16",
                    choices=["float16", "bfloat16", "float32"])
    ap.add_argument(
        "--llm-device-map",
        default=None,
        help="Pass 'auto' to shard the LLM across all visible GPUs "
             "(useful for >7B models on 8GB cards).",
    )
    args = ap.parse_args()

    feat_dir = Path(args.feat_dir)
    match_dir = Path(args.match_dir)
    if not feat_dir.exists():
        raise SystemExit(f"No feat dir: {feat_dir} (run inference first)")

    clip_device = torch.device(
        args.clip_device if torch.cuda.is_available() else "cpu"
    )
    llm_device = (
        args.llm_device if torch.cuda.is_available() and args.llm_device_map is None
        else "cuda:0"
    )

    regions = list_regions(feat_dir)
    buildings = list_buildings(feat_dir)
    if not regions:
        raise SystemExit(f"No regions found under {feat_dir}")

    print(f"[webapp_llm] buildings={len(buildings)} regions={len(regions)}")
    text_encoder = TextEncoder(CLIP_MODEL_ID, clip_device)
    llm = LocalLLMParser(
        model_id=args.llm_model,
        device=llm_device,
        dtype=args.llm_dtype,
        device_map=args.llm_device_map,
    )

    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.world_axes.visible = True

    state: Dict[str, object] = {
        "asset": None,
        "cluster_handles": [],
        "region_label_handles": [],
    }

    # ---------------- GUI ----------------
    gui_md_title = server.gui.add_markdown(
        "## Natural-language → 3D segmentation\n"
        "Type a drone command in Korean or English.\n"
        "Example: `위층 방의 화장실 사진 촬영해줘`"
    )
    view_level = server.gui.add_dropdown(
        "View level", options=["building", "room"], initial_value="building"
    )
    building_dropdown = server.gui.add_dropdown(
        "Building", options=buildings, initial_value=buildings[0]
    )
    region_dropdown = server.gui.add_dropdown(
        "Room", options=regions, initial_value=regions[0]
    )
    region_dropdown.visible = False

    nl_text = server.gui.add_text(
        "Drone command (natural language)",
        initial_value="위층 방의 화장실 사진 촬영해줘",
    )
    run_btn = server.gui.add_button("Parse + Localize")

    point_size_slider = server.gui.add_slider(
        "Point size", min=0.005, max=0.10, step=0.005, initial_value=0.02
    )
    overlay_alpha = server.gui.add_slider(
        "Heatmap opacity", min=0.0, max=1.0, step=0.05, initial_value=0.7
    )
    cluster_top_pct = server.gui.add_slider(
        "Cluster top-percentile", min=80.0, max=99.5, step=0.5, initial_value=95.0
    )
    cluster_eps = server.gui.add_slider(
        "Cluster eps (m)", min=0.05, max=1.0, step=0.05, initial_value=0.25
    )
    cluster_top_k = server.gui.add_slider(
        "Cluster top-K", min=1, max=10, step=1, initial_value=5
    )
    cluster_marker_size = server.gui.add_slider(
        "Marker radius (m)", min=0.05, max=0.5, step=0.01, initial_value=0.15
    )
    show_region_labels = server.gui.add_checkbox(
        "Show region IDs (building view)", initial_value=True
    )

    status_md = server.gui.add_markdown("_status: ready_")
    intent_md = server.gui.add_markdown("")
    legend_md = server.gui.add_markdown("")

    # ---------------- helpers ----------------
    def clear_region_labels() -> None:
        for h in state["region_label_handles"]:  # type: ignore[assignment]
            try:
                h.remove()
            except Exception:
                pass
        state["region_label_handles"] = []

    def render_region_labels(building_id: str, world_center: np.ndarray) -> None:
        """Place a 3D text label at each region's centroid (display coords)."""
        clear_region_labels()
        if not show_region_labels.value:
            return
        handles = []
        for r in regions_for_building(building_id, feat_dir):
            cpath = feat_dir / r / "coord.npy"
            if not cpath.exists():
                continue
            coord = np.load(cpath).astype(np.float32)
            centroid = coord.mean(axis=0) - world_center
            top_z = float(coord[:, 2].max()) - float(world_center[2]) + 0.3
            handles.append(server.scene.add_label(
                f"/region_labels/{r}",
                text=r,
                position=(float(centroid[0]), float(centroid[1]), top_z),
            ))
        state["region_label_handles"] = handles

    def clear_cluster_markers() -> None:
        for h in state["cluster_handles"]:  # type: ignore[assignment]
            try:
                h.remove()
            except Exception:
                pass
        state["cluster_handles"] = []

    def rgb_for_asset(asset: RegionAssets) -> np.ndarray:
        if asset.vertex_colors is not None:
            return asset.vertex_colors
        return np.full((asset.n_render_units, 3), 200, np.uint8)

    def reset_camera_to_asset(asset: RegionAssets) -> None:
        ext = asset.coord.max(0) - asset.coord.min(0)
        diag = float(np.linalg.norm(ext)) or 5.0
        for client in server.get_clients().values():
            client.camera.position = (diag * 0.9, -diag * 0.9, diag * 0.6)
            client.camera.look_at = (0.0, 0.0, 0.0)

    def render_rgb() -> None:
        asset: RegionAssets = state["asset"]  # type: ignore[assignment]
        if asset is None:
            return
        upload_points(server, "/region", asset, rgb_for_asset(asset),
                      float(point_size_slider.value))

    def load_and_render_building(building_id: str) -> None:
        t0 = time.time()
        status_md.content = f"_loading building **{building_id}** ..._"
        asset = load_building(building_id, feat_dir, match_dir, clip_device)
        state["asset"] = asset
        clear_cluster_markers()
        render_rgb()
        render_region_labels(building_id, asset.center)
        reset_camera_to_asset(asset)
        n_rooms = len(regions_for_building(building_id, feat_dir))
        status_md.content = (
            f"_loaded **{building_id}**: {n_rooms} rooms, "
            f"{asset.n_render_units} points "
            f"({time.time()-t0:.2f}s)_"
        )

    def load_and_render_region(region_name: str) -> None:
        t0 = time.time()
        asset = load_region(region_name, feat_dir, match_dir, clip_device)
        state["asset"] = asset
        clear_cluster_markers()
        clear_region_labels()
        render_rgb()
        reset_camera_to_asset(asset)
        status_md.content = (
            f"_loaded **{region_name}**: {asset.n_render_units} points "
            f"({time.time()-t0:.2f}s)_"
        )

    def run_pipeline() -> None:
        asset: RegionAssets = state["asset"]  # type: ignore[assignment]
        if asset is None:
            status_md.content = "_no scene loaded_"
            return
        user_text = nl_text.value.strip()
        if not user_text:
            status_md.content = "_query is empty_"
            return

        # 1. LLM parse
        t0 = time.time()
        status_md.content = "_parsing with LLM ..._"
        intent: ParsedIntent = llm.parse(user_text)
        t_llm = time.time() - t0
        intent_md.content = (
            f"### Parsed intent ({t_llm:.2f}s)\n"
            f"- target_object: `{intent.target_object}`\n"
            f"- clip_prompt:   `{intent.clip_prompt}`\n"
            f"- location_hint: `{intent.location_hint}`\n"
            f"- action:        `{intent.action}`\n"
            f"- return_home:   `{intent.return_home}`"
        )

        # 2. CLIP heatmap
        t1 = time.time()
        tf = text_encoder.encode([intent.clip_prompt])[0]
        scores = query_single(asset, tf)

        # 3. Cluster
        params = ClusterParams(
            top_percentile=float(cluster_top_pct.value),
            threshold=None,
            eps=float(cluster_eps.value),
            min_points=40,
            top_k=int(cluster_top_k.value),
        )
        cutoff = float(np.percentile(scores, params.top_percentile))
        mask = scores >= cutoff

        rgb_base = rgb_for_asset(asset)
        overlay = rgb_base.copy()
        if mask.any():
            overlay[mask] = heatmap_colors(scores[mask])
        blended = blend_with_rgb(overlay, rgb_base, float(overlay_alpha.value))
        upload_points(server, "/region", asset, blended,
                      float(point_size_slider.value))

        result = candidates_from_heatmap(asset.coord, scores, params)
        cands = result["candidates"]

        clear_cluster_markers()
        handles = []
        radius = float(cluster_marker_size.value)
        palette = cluster_palette(max(len(cands), 1))
        for c in cands:
            rank = c["rank"]
            center = tuple(float(v) for v in c["center"])
            color = tuple(int(v) for v in palette[rank])
            handles.append(server.scene.add_icosphere(
                f"/cluster/sphere_{rank}",
                radius=radius, color=color, opacity=0.85, position=center,
            ))
            bb_min = np.asarray(c["bbox_min"], dtype=np.float32)
            bb_max = np.asarray(c["bbox_max"], dtype=np.float32)
            segs = _bbox_edge_segments(bb_min, bb_max)
            handles.append(server.scene.add_line_segments(
                f"/cluster/bbox_{rank}",
                points=segs, colors=color, line_width=2.0,
            ))
            label_pos = (center[0], center[1], float(bb_max[2]) + 0.15)
            handles.append(server.scene.add_label(
                f"/cluster/label_{rank}",
                text=f"#{rank} n={c['n_points']} s={c['mean_score']:.3f}",
                position=label_pos,
            ))
        state["cluster_handles"] = handles

        # The candidate centers above are in display (centered) coordinates;
        # report world coordinates too so they can be fed to the path planner.
        world_lines = []
        for c in cands:
            wc = np.asarray(c["center"], dtype=np.float32) + asset.center
            world_lines.append(
                f"- **#{c['rank']}** world=({wc[0]:.2f}, {wc[1]:.2f}, {wc[2]:.2f}) "
                f"• n={c['n_points']} • mean={c['mean_score']:.3f} "
                f"• max={c['max_score']:.3f}"
            )
        t_seg = time.time() - t1

        if not cands:
            world_lines.append("_no candidates — try lowering top-percentile or eps_")
        legend_md.content = (
            f"### Target `{intent.clip_prompt}` "
            f"(LLM {t_llm:.2f}s • seg {t_seg:.2f}s)\n"
            f"score range [{result['stats']['score_min']:.3f}, "
            f"{result['stats']['score_max']:.3f}] • "
            f"active {result['stats']['n_active_points']} pts • "
            f"clusters={result['stats']['n_clusters']}\n\n"
            + "\n".join(world_lines)
        )
        status_md.content = (
            f"_done: {len(cands)} candidate(s) in {t_llm + t_seg:.2f}s_"
        )

    # ---------------- callbacks ----------------
    @view_level.on_update
    def _(_):
        is_b = view_level.value == "building"
        building_dropdown.visible = is_b
        region_dropdown.visible = not is_b
        if is_b:
            load_and_render_building(building_dropdown.value)
        else:
            load_and_render_region(region_dropdown.value)

    @building_dropdown.on_update
    def _(_):
        if view_level.value == "building":
            load_and_render_building(building_dropdown.value)

    @region_dropdown.on_update
    def _(_):
        if view_level.value == "room":
            load_and_render_region(region_dropdown.value)

    @run_btn.on_click
    def _(_):
        try:
            run_pipeline()
        except Exception as e:
            status_md.content = f"_error: {e}_"
            raise

    @point_size_slider.on_update
    def _(_):
        if state["asset"] is not None:
            render_rgb()

    @show_region_labels.on_update
    def _(_):
        if view_level.value == "building" and state["asset"] is not None:
            asset: RegionAssets = state["asset"]  # type: ignore[assignment]
            if show_region_labels.value:
                render_region_labels(building_dropdown.value, asset.center)
            else:
                clear_region_labels()

    load_and_render_building(buildings[0])

    print(f"[webapp_llm] running at http://{args.host}:{args.port}")
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
