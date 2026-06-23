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
from typing import Dict, List, Optional

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
    _feat_path,
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
from webapp import unidet3d_backend as u3d  # noqa: E402  (alt detection backend)

from webapp_llm.llm_parser import LocalLLMParser, ParsedIntent  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cache-dir",
        default=str(_THIS.parents[1] / "cache"),
        help="Root cache dir. Layout: <cache>/<building>/feat/<region>/...",
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
    u3d.add_unidet3d_args(ap)
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    if not cache_dir.exists():
        raise SystemExit(f"No cache dir: {cache_dir} (run inference first)")

    clip_device = torch.device(
        args.clip_device if torch.cuda.is_available() else "cpu"
    )
    llm_device = (
        args.llm_device if torch.cuda.is_available() and args.llm_device_map is None
        else "cuda:0"
    )

    regions = list_regions(cache_dir)
    buildings = list_buildings(cache_dir)
    if not regions:
        raise SystemExit(f"No regions found under {cache_dir}")

    print(f"[webapp_llm] buildings={len(buildings)} regions={len(regions)}")
    text_encoder = TextEncoder(CLIP_MODEL_ID, clip_device)
    llm = LocalLLMParser(
        model_id=args.llm_model,
        device=llm_device,
        dtype=args.llm_dtype,
        device_map=args.llm_device_map,
    )

    # Optional UniDet3D detection backend (lazy; mmdet3d only loads on detect()).
    detector = u3d.make_detector(args)
    u3d_session = u3d.UniDet3DSession()
    if detector is not None:
        print(f"[webapp_llm] UniDet3D backend enabled (head={args.unidet3d_dataset}); "
              f"select it via the 'Backend' dropdown.")

    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.world_axes.visible = True

    state: Dict[str, object] = {
        "asset": None,
        "cluster_handles": [],
        "region_label_handles": [],
        "suppress_scene_cb": False,  # mute dropdown callbacks during LLM-driven scene switch
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

    backend = server.gui.add_dropdown(
        "Backend",
        options=["mosaic3d"] + (["unidet3d"] if detector is not None else []),
        initial_value="mosaic3d",
    )
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
    # --- UniDet3D backend controls (hidden unless Backend == unidet3d) ---
    u_score_thr = server.gui.add_slider(
        "UniDet3D: score thr", min=0.05, max=0.9, step=0.05,
        initial_value=float(getattr(args, "unidet3d_score_thr", 0.30)),
    )
    u_topk = server.gui.add_slider(
        "UniDet3D: top-K match", min=1, max=10, step=1, initial_value=3
    )
    u_show_all = server.gui.add_checkbox(
        "UniDet3D: show all boxes", initial_value=True
    )
    u_detect_btn = server.gui.add_button("UniDet3D: (re)run detection")
    for _w in (u_score_thr, u_topk, u_show_all, u_detect_btn):
        _w.visible = False
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
        for r in regions_for_building(building_id, cache_dir):
            cpath = _feat_path(r, cache_dir) / "coord.npy"
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

    def invalidate_unidet() -> None:
        """Drop cached UniDet3D boxes when the loaded scene changes."""
        u3d.clear_boxes(u3d_session)
        u3d_session.det = None
        u3d_session.scene_key = None

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
        asset = load_building(building_id, cache_dir, clip_device)
        state["asset"] = asset
        clear_cluster_markers()
        invalidate_unidet()
        render_rgb()
        render_region_labels(building_id, asset.center)
        reset_camera_to_asset(asset)
        n_rooms = len(regions_for_building(building_id, cache_dir))
        status_md.content = (
            f"_loaded **{building_id}**: {n_rooms} rooms, "
            f"{asset.n_render_units} points "
            f"({time.time()-t0:.2f}s)_"
        )

    def load_and_render_region(region_name: str) -> None:
        t0 = time.time()
        asset = load_region(region_name, cache_dir, clip_device)
        state["asset"] = asset
        clear_cluster_markers()
        invalidate_unidet()
        clear_region_labels()
        render_rgb()
        reset_camera_to_asset(asset)
        status_md.content = (
            f"_loaded **{region_name}**: {asset.n_render_units} points "
            f"({time.time()-t0:.2f}s)_"
        )

    def update_backend_visibility() -> None:
        is_uni = detector is not None and backend.value == "unidet3d"
        for w in (cluster_top_pct, cluster_eps, cluster_top_k, cluster_marker_size):
            w.visible = not is_uni
        for w in (u_score_thr, u_topk, u_show_all, u_detect_btn):
            w.visible = is_uni

    def run_unidet3d_match(intent: ParsedIntent, t_llm: float,
                           force_detect: bool = False) -> None:
        """UniDet3D backend: detect on the loaded scene, CLIP-match clip_prompt."""
        asset: RegionAssets = state["asset"]  # type: ignore[assignment]
        clear_cluster_markers()
        render_rgb()

        scene_key = getattr(asset, "name", None)
        need = (force_detect or u3d_session.det is None
                or u3d_session.scene_key != scene_key)
        if need:
            # 1) precomputed cache (no GPU). 2) manual button → live detect.
            #    3) auto path w/o cache → ask for precompute (avoid OOM).
            det = u3d.assemble_cached_detection(asset, cache_dir)
            if det is not None:
                emb = u3d.embeds_for_det(det, text_encoder)
                status_md.content = "_UniDet3D: loaded cached detection_"
            elif force_detect:
                status_md.content = "_UniDet3D: detecting live (no cache) ..._"
                try:
                    det, emb = u3d.detect_scene(
                        detector, text_encoder, asset, u_score_thr.value)
                except RuntimeError as e:
                    u3d.clear_boxes(u3d_session)
                    u3d_session.det = None
                    legend_md.content = ""
                    status_md.content = f"_{e}_"
                    return
            else:
                u3d.clear_boxes(u3d_session)
                u3d_session.det = None
                legend_md.content = ""
                status_md.content = (
                    f"_UniDet3D: no precomputed detection for `{scene_key}`. "
                    "Run `precompute_unidet3d.py`, or click "
                    "'UniDet3D: (re)run detection' to detect live._")
                return
            u3d_session.det = det
            u3d_session.box_class_embeds = emb
            u3d_session.scene_key = scene_key

        det = u3d_session.det
        t1 = time.time()
        top_idx, sims = u3d.match_boxes(
            det, u3d_session.box_class_embeds, text_encoder,
            intent.clip_prompt, int(u_topk.value))
        u3d.render_boxes(server, u3d_session, det,
                         show_all=u_show_all.value, highlight_idx=top_idx)
        t_seg = time.time() - t1

        lines = [
            f"### Target `{intent.clip_prompt}` via **unidet3d** "
            f"({args.unidet3d_dataset}) (LLM {t_llm:.2f}s • match {t_seg:.2f}s)",
            f"{len(det.bboxes)} boxes • score-thr={u_score_thr.value:.2f}",
            "",
        ]
        for rank, i in enumerate(top_idx):
            i = int(i)
            li = int(det.labels[i])
            cls = det.classes[li] if 0 <= li < len(det.classes) else f"class_{li}"
            wc = u3d.world_center(det.bboxes[i], asset.center)
            sim = float(sims[i]) if sims is not None else 0.0
            lines.append(
                f"- **#{rank+1}** `{cls}` sim={sim:.3f} • "
                f"world=({wc[0]:.2f}, {wc[1]:.2f}, {wc[2]:.2f})"
            )
        if len(top_idx) == 0:
            lines.append("_no boxes detected — lower score thr or pick another head_")
        legend_md.content = "\n".join(lines)
        status_md.content = (
            f"_UniDet3D: {len(det.bboxes)} boxes, {len(top_idx)} matched_"
        )

    def apply_intent_scene(intent: ParsedIntent) -> str:
        """Switch the loaded scene per the LLM's room/scope.

        - scope == "room" + target_room=N → load the N-th room (1-based) of the
          currently selected building.
        - scope == "building" → load the whole current building.
        - otherwise → keep the current scene.
        Dropdowns are updated to match (their callbacks are muted to avoid a
        double load). Returns a short status note for the intent panel.
        """
        bld = building_dropdown.value
        rooms = regions_for_building(bld, cache_dir)
        state["suppress_scene_cb"] = True
        try:
            if intent.scope == "room" and intent.target_room is not None:
                n = int(intent.target_room)
                if 1 <= n <= len(rooms):
                    target = rooms[n - 1]
                    view_level.value = "room"
                    building_dropdown.visible = False
                    region_dropdown.visible = True
                    region_dropdown.value = target
                    load_and_render_region(target)
                    return f"room {n} → `{target}`"
                return (f"⚠ room {n} out of range (building has "
                        f"{len(rooms)} rooms) — searching current scene")
            if intent.scope == "building":
                view_level.value = "building"
                building_dropdown.visible = True
                region_dropdown.visible = False
                load_and_render_building(bld)
                return f"whole building `{bld}` ({len(rooms)} rooms)"
            return "current scene (no room/scope specified)"
        finally:
            state["suppress_scene_cb"] = False

    def run_pipeline() -> None:
        if state["asset"] is None:
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

        # 1b. Switch scene per parsed room/scope, then refresh the asset.
        scene_note = apply_intent_scene(intent)
        asset: RegionAssets = state["asset"]  # type: ignore[assignment]

        raw_preview = intent.raw_text.strip()
        if len(raw_preview) > 1200:
            raw_preview = raw_preview[:1200] + "  …(truncated)"
        intent_md.content = (
            f"### Parsed intent ({t_llm:.2f}s)\n"
            f"- target_object: `{intent.target_object}`\n"
            f"- clip_prompt:   `{intent.clip_prompt}`\n"
            f"- location_hint: `{intent.location_hint}`\n"
            f"- target_room:   `{intent.target_room}`  scope: `{intent.scope or '—'}`\n"
            f"- search scene:  {scene_note}\n"
            f"- action:        `{intent.action}`\n"
            f"- return_home:   `{intent.return_home}`\n\n"
            f"**LLM raw output**\n"
            f"```\n{raw_preview}\n```"
        )

        # UniDet3D backend: detect + CLIP-match instead of the heatmap path.
        if detector is not None and backend.value == "unidet3d":
            run_unidet3d_match(intent, t_llm)
            return

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
        if state["suppress_scene_cb"]:
            return
        if is_b:
            load_and_render_building(building_dropdown.value)
        else:
            load_and_render_region(region_dropdown.value)

    @building_dropdown.on_update
    def _(_):
        if not state["suppress_scene_cb"] and view_level.value == "building":
            load_and_render_building(building_dropdown.value)

    @region_dropdown.on_update
    def _(_):
        if not state["suppress_scene_cb"] and view_level.value == "room":
            load_and_render_region(region_dropdown.value)

    @run_btn.on_click
    def _(_):
        try:
            run_pipeline()
        except Exception as e:
            status_md.content = f"_error: {e}_"
            raise

    @backend.on_update
    def _(_):
        update_backend_visibility()
        if backend.value != "unidet3d":
            u3d.clear_boxes(u3d_session)
            if state["asset"] is not None:
                render_rgb()

    @u_detect_btn.on_click
    def _(_):
        invalidate_unidet()
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

    update_backend_visibility()
    load_and_render_building(buildings[0])

    print(f"[webapp_llm] running at http://{args.host}:{args.port}")
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
