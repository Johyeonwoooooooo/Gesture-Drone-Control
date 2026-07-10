"""webapp_llm_v2 — terminal-driven NL → localize → path-plan → Tello-SDK export.

Pipeline per query (typed in the TERMINAL, not the viser GUI):

    natural-language command
        -> local LLM intent parse            (webapp_llm.llm_parser)
        -> CLIP heatmap + DBSCAN candidates   (webapp.server + inference)
        -> viser: heatmap + target marker
        -> A* / RRT* path from current pos to the target   (webapp_llm_v2.planner)
        -> viser: path polyline + start/goal markers
        -> Tello SDK command program written to out/       (webapp_llm_v2.sdk_export)

Continuous mission: the first query starts from `--home-xyz` (default = the
lowest-floor first room's centroid); each later query starts from the previous
query's goal. Scope: houses 00800 / 00809 only. Mosaic3D backend only.

Run (in the `mosaic3d` conda env):
    python 3D-segmentation/webapp_llm_v2/server.py \
        --building 00809_Qpor2mEya8F \
        --llm-device cuda:1 --clip-device cuda:0
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

# Make sibling packages importable (same trick as webapp_llm/server.py).
_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parents[1]))  # .../3D-segmentation
sys.path.insert(0, str(_THIS.parents[1] / "webapp"))

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
    load_building,
    query_single,
    regions_for_building,
    upload_points,
)
from webapp_llm.llm_parser import LocalLLMParser, ParsedIntent  # noqa: E402
from webapp_llm.room_labels import (  # noqa: E402
    load_room_labels,
    match_room_by_hint,
    room_directory_text,
)

from webapp_llm_v2 import planner, sdk_export  # noqa: E402

# Only these two houses are in scope.
ALLOWED_BUILDINGS = {"00800_TEEsavR23oF", "00809_Qpor2mEya8F"}
HILITE_TINT = np.array([90, 160, 255], dtype=np.float32)  # target-room tint


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=str(_THIS.parents[1] / "cache"))
    ap.add_argument("--building", default="00800_TEEsavR23oF")
    ap.add_argument("--port", type=int, default=8095)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--clip-device", default="cuda:0")
    ap.add_argument("--llm-model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--llm-device", default="cuda:1")
    ap.add_argument("--llm-dtype", default="float16",
                    choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--llm-device-map", default=None)
    # planner
    ap.add_argument("--algo", default="astar", choices=["astar", "rrt"])
    ap.add_argument("--resolution", type=float, default=0.15)
    ap.add_argument("--margin", type=int, default=1)
    ap.add_argument("--sample", type=int, default=10)
    ap.add_argument("--rrt-iter", type=int, default=8000,
                    help="RRT* sample budget (naive NN is O(iter^2); raise for "
                         "long cross-room paths, expect it to be slow).")
    # cluster
    ap.add_argument("--cluster-top-pct", type=float, default=95.0)
    ap.add_argument("--cluster-eps", type=float, default=0.25)
    ap.add_argument("--cluster-top-k", type=int, default=5)
    # tello / output
    ap.add_argument("--tello-speed", type=int, default=40)
    ap.add_argument("--home-xyz", type=float, nargs=3, default=None,
                    metavar=("X", "Y", "Z"),
                    help="World-meter launch point. Default: lowest-floor first "
                         "room centroid.")
    ap.add_argument("--out-dir", default=str(_THIS.parent / "out"))
    ap.add_argument("--point-size", type=float, default=0.02)
    ap.add_argument("--overlay-alpha", type=float, default=0.7)
    # drone flight animation (viser)
    ap.add_argument("--anim-speed", type=float, default=1.0,
                    help="Drone animation speed (m/s) along the planned path.")
    ap.add_argument("--anim-fps", type=float, default=20.0,
                    help="Animation frames per second.")
    ap.add_argument("--no-anim", action="store_true",
                    help="Skip the drone fly-through; jump straight to the goal.")
    # Unity simulator link (simulator/bridge; see README-integration.md)
    ap.add_argument("--sim", action="store_true",
                    help="Fly the planned path in the Unity Tello simulator "
                         "instead of the viser-only animation.")
    ap.add_argument("--unity-host", default=None,
                    help="IP of the machine running Unity (required with --sim).")
    ap.add_argument("--unity-port", type=int, default=9000)
    ap.add_argument("--unity-state-port", type=int, default=9002)
    ap.add_argument("--unity-local-port", type=int, default=9001)
    ap.add_argument("--sim-transform", default=None,
                    help="Path to a calibrated transform JSON. Default: "
                         "simulator/bridge/transforms/<building>.json")
    ap.add_argument("--sim-speed", type=float, default=2.0,
                    help="Flight speed in Unity units/s (house is at scale 5, "
                         "so 2.0 u/s = 0.4 m/s real).")
    ap.add_argument("--sim-rc-limit", type=int, default=30)
    ap.add_argument("--sim-timeout", type=float, default=0.0,
                    help="Flight timeout seconds; 0 = auto from path length.")
    ap.add_argument("--sim-no-status", action="store_true",
                    help="Do not push status text to the Unity on-screen banner.")
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    if not cache_dir.exists():
        raise SystemExit(f"No cache dir: {cache_dir}")
    if args.building not in ALLOWED_BUILDINGS:
        raise SystemExit(
            f"--building must be one of {sorted(ALLOWED_BUILDINGS)} (got {args.building})")

    buildings = [b for b in list_buildings(cache_dir) if b in ALLOWED_BUILDINGS]
    if args.building not in buildings:
        raise SystemExit(f"Building {args.building} not found under {cache_dir}")

    clip_device = torch.device(
        args.clip_device if torch.cuda.is_available() else "cpu")
    llm_device = (
        args.llm_device if torch.cuda.is_available() and args.llm_device_map is None
        else "cuda:0")

    print(f"[v2] loading CLIP text encoder on {clip_device} ...")
    text_encoder = TextEncoder(CLIP_MODEL_ID, clip_device)
    print(f"[v2] loading LLM {args.llm_model} on {llm_device} ...")
    llm = LocalLLMParser(model_id=args.llm_model, device=llm_device,
                         dtype=args.llm_dtype, device_map=args.llm_device_map)

    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.world_axes.visible = True
    print(f"[v2] viser running at http://{args.host}:{args.port}")

    # ---------------- Unity simulator link (lazy: only with --sim) ----------
    bridge = None
    coord_transform = follow_path = None  # modules, bound under --sim
    if args.sim:
        if not args.unity_host:
            raise SystemExit("--sim requires --unity-host <ip-of-unity-machine>")
        sys.path.insert(0, str(_THIS.parents[2]))  # repo root
        from simulator.bridge import coord_transform, follow_path  # noqa: E402
        from simulator.bridge.unity_bridge import UnityTelloBridge  # noqa: E402
        bridge = UnityTelloBridge(args.unity_host, args.unity_port,
                                  args.unity_local_port, args.unity_state_port)
        bridge.connect()
        reply = bridge.initialize_sdk()
        print(f"[sim] Unity {args.unity_host}:{args.unity_port} -> {reply!r}"
              + ("  (no reply yet — is Unity in Play mode?)" if reply == "timeout" else ""))

    def sim_transform_for(building_id: str):
        """Calibrated SimTransform for the building, or None (sim disabled)."""
        if not args.sim:
            return None
        try:
            if args.sim_transform:
                return coord_transform.SimTransform.load(args.sim_transform)
            return coord_transform.load_building_transform(building_id)
        except FileNotFoundError as e:
            print(f"[sim] WARNING: {e}\n[sim] simulator flight disabled for "
                  f"{building_id}.")
            return None

    def status(text: str) -> None:
        """Progress line: terminal always, Unity banner when connected."""
        print(f"[status] {text}")
        if bridge is not None and not args.sim_no_status:
            try:
                bridge.send_status(text)
            except Exception:
                pass

    state: Dict[str, object] = {
        "asset": None,          # RegionAssets (building view)
        "building": args.building,
        "points_world": None,   # (N,3) world coords of asset
        "gm": None,             # cached voxel grid for the current building
        "home": None,           # (3,) world launch point
        "last_goal": None,      # (3,) world — next query starts here
        "handles": [],          # per-query viser handles (cleared each query)
        "label_handles": [],    # room-label handles (toggled by GUI checkbox)
        "home_handles": [],     # persistent HOME marker handles
        "drone_handles": [],    # persistent DRONE (current pos) marker handles
        "show_labels": True,    # room-label visibility (GUI checkbox)
        "room_labels": {},      # building -> labels dict
        "sim_transform": None,  # SimTransform for the current building (--sim)
    }

    # ---------------- GUI (viser): room-label on/off toggle ----------------
    server.gui.add_markdown("### webapp_llm_v2\nType queries in the **terminal**.")
    show_labels_cb = server.gui.add_checkbox("Show room labels", initial_value=True)

    # ------------------------------------------------------------------ helpers
    def labels_for(building_id: str) -> Dict[str, Dict]:
        cache: Dict[str, Dict] = state["room_labels"]  # type: ignore[assignment]
        if building_id not in cache:
            cache[building_id] = load_room_labels(building_id, cache_dir)
        return cache[building_id]

    def rgb_for_asset(asset: RegionAssets) -> np.ndarray:
        if asset.vertex_colors is not None:
            return asset.vertex_colors
        return np.full((asset.n_render_units, 3), 200, np.uint8)

    def scene_base_rgb(asset: RegionAssets, target_region: Optional[str]) -> np.ndarray:
        rgb = rgb_for_asset(asset).astype(np.float32).copy()
        slices = getattr(asset, "region_slices", None)
        if target_region and slices and target_region in slices:
            a, b = slices[target_region]
            rgb[a:b] = 0.6 * rgb[a:b] + 0.4 * HILITE_TINT
        return np.clip(rgb, 0, 255).astype(np.uint8)

    def clear_handles() -> None:
        for h in state["handles"]:  # type: ignore[assignment]
            try:
                h.remove()
            except Exception:
                pass
        state["handles"] = []

    def first_region_centroid(building_id: str) -> np.ndarray:
        """World centroid of the lowest-floor first region (default home)."""
        regs = sorted(regions_for_building(building_id, cache_dir))
        coord = np.load(_feat_path(regs[0], cache_dir) / "coord.npy").astype(np.float32)
        return coord.mean(axis=0)

    def load_building_scene(building_id: str) -> None:
        t0 = time.time()
        print(f"[v2] loading building {building_id} ...")
        asset = load_building(building_id, cache_dir, clip_device)
        state["asset"] = asset
        state["building"] = building_id
        points_world = asset.coord + asset.center  # (N,3) world meters
        state["points_world"] = points_world
        print(f"[v2] voxelizing {len(points_world)} points "
              f"(res={args.resolution} margin={args.margin} sample={args.sample}) ...")
        state["gm"] = planner.voxelize(points_world, args.resolution,
                                       args.margin, args.sample)
        if args.home_xyz is not None:
            state["home"] = np.asarray(args.home_xyz, dtype=float)
        else:
            state["home"] = first_region_centroid(building_id)
        state["last_goal"] = None
        state["sim_transform"] = sim_transform_for(building_id)
        if bridge is not None and state["sim_transform"] is not None:
            hu = state["sim_transform"].mosaic_to_unity(np.asarray(state["home"]))
            bridge.set_position(float(hu[0]), float(hu[1]), float(hu[2]), 0.0)
        clear_handles()
        upload_points(server, "/region", asset, rgb_for_asset(asset),
                      float(args.point_size))
        render_region_labels(building_id, asset.center)
        render_home_marker(asset)
        render_drone_marker(asset, state["home"])  # drone starts at home
        print(f"[v2] building ready: {len(regions_for_building(building_id, cache_dir))} "
              f"rooms, {asset.n_render_units} pts, grid={state['gm'].shape} "
              f"({time.time()-t0:.1f}s). home={np.round(state['home'],2)}")

    def clear_region_labels() -> None:
        for h in state["label_handles"]:  # type: ignore[assignment]
            try:
                h.remove()
            except Exception:
                pass
        state["label_handles"] = []

    def render_region_labels(building_id: str, world_center: np.ndarray) -> None:
        clear_region_labels()
        if not state["show_labels"]:
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
                f"/region_labels/{r}", text=r,
                position=(float(centroid[0]), float(centroid[1]), top_z)))
        state["label_handles"] = handles

    def render_home_marker(asset: RegionAssets) -> None:
        """Persistent gold HOME marker (drone launch / return point)."""
        for h in state["home_handles"]:  # type: ignore[assignment]
            try:
                h.remove()
            except Exception:
                pass
        hp = np.asarray(state["home"], dtype=float) - asset.center
        pos = tuple(float(v) for v in hp)
        state["home_handles"] = [
            server.scene.add_icosphere("/home/sphere", radius=0.22,
                                       color=(255, 200, 0), opacity=0.55, position=pos),
            server.scene.add_label("/home/label", text="HOME",
                                   position=(pos[0], pos[1], pos[2] + 0.35)),
        ]

    def render_drone_marker(asset: RegionAssets, pos_world) -> None:
        """Persistent cyan DRONE marker at the drone's current position."""
        for h in state["drone_handles"]:  # type: ignore[assignment]
            try:
                h.remove()
            except Exception:
                pass
        dp = np.asarray(pos_world, dtype=float) - asset.center
        pos = tuple(float(v) for v in dp)
        state["drone_handles"] = [
            server.scene.add_icosphere("/drone/sphere", radius=0.16,
                                       color=(0, 210, 210), opacity=0.95, position=pos),
            server.scene.add_label("/drone/label", text="DRONE",
                                   position=(pos[0], pos[1], pos[2] + 0.3)),
        ]

    def animate_drone(asset: RegionAssets, path_world) -> None:
        """Fly the DRONE marker slowly along the planned path (world waypoints)."""
        pts = [np.asarray(p, dtype=float) for p in path_world]
        if len(pts) < 2 or args.no_anim:
            render_drone_marker(asset, pts[-1] if pts else state["home"])
            return
        step = max(1e-3, float(args.anim_speed) / max(1.0, float(args.anim_fps)))
        dt = 1.0 / max(1.0, float(args.anim_fps))
        for i in range(1, len(pts)):
            a, b = pts[i - 1], pts[i]
            seg = float(np.linalg.norm(b - a))
            n = max(1, int(np.ceil(seg / step)))
            for k in range(1, n + 1):
                render_drone_marker(asset, a + (b - a) * (k / n))
                time.sleep(dt)

    @show_labels_cb.on_update
    def _(_):
        state["show_labels"] = bool(show_labels_cb.value)
        asset = state["asset"]
        if asset is not None:
            render_region_labels(state["building"], asset.center)

    # ---------------------------------------------------------- scope resolution
    def resolve_target_region(intent: ParsedIntent, building_id: str) -> Optional[str]:
        """Full region name to scope the search to, or None for whole building."""
        rooms = regions_for_building(building_id, cache_dir)

        def _full(suffix: Optional[str]) -> Optional[str]:
            if not suffix:
                return None
            for r in rooms:
                if r.endswith(suffix) or r.endswith("_" + suffix):
                    return r
            return None

        if intent.scope == "building":
            return None
        if intent.target_room:
            return _full(intent.target_room)
        hint_room = match_room_by_hint(intent.location_hint, intent.target_object,
                                       labels_for(building_id))
        return _full(hint_room)

    # ------------------------------------------------------------- per-query run
    def run_query(user_text: str) -> None:
        asset: RegionAssets = state["asset"]  # type: ignore[assignment]
        building_id = state["building"]  # type: ignore[assignment]

        # 1. LLM intent parse
        status(f"의도 분석 중... ({user_text})")
        t0 = time.time()
        room_dir = room_directory_text(labels_for(building_id))
        intent = llm.parse(user_text, room_directory=room_dir)
        t_llm = time.time() - t0
        target_region = resolve_target_region(intent, building_id)
        print(f"[intent] object={intent.target_object!r} clip={intent.clip_prompt!r} "
              f"room={target_region or 'whole building'} action={intent.action} "
              f"return_home={intent.return_home}  ({t_llm:.2f}s)")

        # 2. CLIP heatmap
        status(f"'{intent.target_object}' 위치 탐색 중 (segmentation)...")
        tf = text_encoder.encode([intent.clip_prompt])[0]
        scores = query_single(asset, tf).astype(np.float32)
        slices = getattr(asset, "region_slices", None)
        if target_region and slices and target_region in slices:
            a, b = slices[target_region]
            room_mask = np.zeros(len(scores), dtype=bool)
            room_mask[a:b] = True
            in_scope = scores[room_mask]
            scores = np.where(room_mask, scores, -1e9).astype(np.float32)
        else:
            room_mask = np.ones(len(scores), dtype=bool)
            in_scope = scores

        # 3. Cluster -> candidates
        params = ClusterParams(top_percentile=float(args.cluster_top_pct),
                               threshold=None, eps=float(args.cluster_eps),
                               min_points=40, top_k=int(args.cluster_top_k))
        result = candidates_from_heatmap(asset.coord, scores, params)
        cands = result["candidates"]
        if not cands:
            status(f"'{intent.target_object}' 를 찾지 못했습니다")
            print("[query] no candidate found — try another phrasing / room. Skipped.")
            return
        top = cands[0]
        goal_world = np.asarray(top["center"], dtype=np.float32) + asset.center
        print(f"[target] #{top['rank']} world={np.round(goal_world,2)} "
              f"n={top['n_points']} mean={top['mean_score']:.3f}")

        # 4. Visualize heatmap + target marker
        clear_handles()
        rgb_base = scene_base_rgb(asset, target_region)
        cutoff = float(np.percentile(in_scope, params.top_percentile)) if in_scope.size else 0.0
        mask = (scores >= cutoff) & room_mask
        overlay = rgb_base.copy()
        if mask.any():
            overlay[mask] = heatmap_colors(scores[mask])
        blended = blend_with_rgb(overlay, rgb_base, float(args.overlay_alpha))
        upload_points(server, "/region", asset, blended, float(args.point_size))

        handles = []
        center_disp = tuple(float(v) for v in top["center"])
        palette = cluster_palette(1)
        handles.append(server.scene.add_icosphere(
            "/target/sphere", radius=0.18, color=(255, 60, 60), opacity=0.9,
            position=center_disp))
        bb_min = np.asarray(top["bbox_min"], dtype=np.float32)
        bb_max = np.asarray(top["bbox_max"], dtype=np.float32)
        handles.append(server.scene.add_line_segments(
            "/target/bbox", points=_bbox_edge_segments(bb_min, bb_max),
            colors=(255, 60, 60), line_width=2.0))
        handles.append(server.scene.add_label(
            "/target/label", text=f"{intent.target_object}",
            position=(center_disp[0], center_disp[1], float(bb_max[2]) + 0.2)))

        # 5. Path plan (continuous mission: start = last goal or home)
        status("경로 계산 중...")
        start_world = state["last_goal"] if state["last_goal"] is not None else state["home"]
        start_world = np.asarray(start_world, dtype=float)
        t1 = time.time()
        path, info, _ = planner.plan_path(
            state["points_world"], start_world, goal_world, algo=args.algo,
            resolution=args.resolution, margin=args.margin, sample=args.sample,
            rrt_iter=args.rrt_iter, gm=state["gm"])
        t_plan = time.time() - t1
        if path is None:
            status("경로 계산 실패")
            print(f"[plan] {args.algo} FAILED ({info.get('reason','?')}, {t_plan:.2f}s) "
                  f"— no path saved.")
            state["handles"] = handles
            state["last_goal"] = goal_world  # still advance the mission target
            render_drone_marker(asset, goal_world)
            return
        print(f"[plan] {args.algo}: {info['n_waypoints']} waypoints, "
              f"{info['length_m']:.2f} m ({t_plan:.2f}s)")

        # 6. Visualize path (display frame) + start/goal spheres
        center = asset.center
        disp = np.array([p - center for p in path], dtype=np.float32)
        if len(disp) >= 2:
            segs = np.stack([disp[:-1], disp[1:]], axis=1)  # (M-1,2,3)
            handles.append(server.scene.add_line_segments(
                "/path/line", points=segs, colors=(60, 220, 90), line_width=4.0))
        handles.append(server.scene.add_icosphere(
            "/path/start", radius=0.15, color=(60, 120, 255), opacity=0.9,
            position=tuple(float(v) for v in (start_world - center))))
        state["handles"] = handles

        # 7. Emit Tello SDK program
        ts = time.strftime("%Y%m%d_%H%M%S")
        program = sdk_export.build_tello_program(
            path, action=intent.action, return_home=intent.return_home,
            home_world=state["home"], start_world=start_world, goal_world=goal_world,
            target_object=intent.target_object, clip_prompt=intent.clip_prompt,
            query=user_text, algo=args.algo, building=building_id,
            speed=args.tello_speed, timestamp=ts)
        out_path = sdk_export.save_program(program, args.out_dir)
        print(f"[sdk] {len(program['commands'])} commands -> {out_path}")

        # 8. Fly the drone along the path — in the Unity simulator when --sim is
        #    active (the viser DRONE marker mirrors the sim state), otherwise the
        #    viser-only animation — then advance the mission.
        sim_tf = state["sim_transform"]
        if bridge is not None and sim_tf is not None:
            wps_unity = sim_tf.mosaic_to_unity(np.asarray(path, dtype=float))
            length_u = float(np.linalg.norm(np.diff(wps_unity, axis=0), axis=1).sum())
            status(f"비행 중... ({info['length_m']:.1f} m)")
            print(f"[sim] flying {length_u:.1f} u at {args.sim_speed} u/s "
                  f"(~{length_u / max(1e-3, args.sim_speed):.0f}s) ...")

            def _on_state(s):
                render_drone_marker(asset, sim_tf.unity_to_mosaic(
                    np.array([s.x, s.y, s.z], dtype=float)))

            res = follow_path.fly_mission(
                bridge, [tuple(p) for p in wps_unity],
                max_speed=float(args.sim_speed), rc_limit=int(args.sim_rc_limit),
                timeout_sec=(args.sim_timeout or None),
                on_state=_on_state, on_status=status)
            status("착륙 완료" if res.success else f"비행 중단 ({res.reason})")
            print(f"[sim] {res.reason}: err={res.final_error_u:.2f}u "
                  f"collisions={res.collision_count} rc={res.rc_commands_sent} "
                  f"{res.duration_s:.0f}s")
            render_drone_marker(asset, goal_world if res.success else
                                sim_tf.unity_to_mosaic(np.asarray(
                                    res.trajectory_unity[-1], dtype=float))
                                if res.trajectory_unity else goal_world)
        else:
            if not args.no_anim:
                flight_s = info["length_m"] / max(1e-3, float(args.anim_speed))
                print(f"[fly] drone flying {info['length_m']:.1f} m at "
                      f"{args.anim_speed} m/s (~{flight_s:.0f}s) ...")
            animate_drone(asset, path)
        state["last_goal"] = goal_world

    # --------------------------------------------------------------- initial load
    load_building_scene(args.building)

    print("\n" + "=" * 68)
    print("  webapp_llm_v2 — type a drone command (Korean/English).")
    print("  viser: gold=HOME, cyan=DRONE(now), red=target, green=path.")
    print("         toggle room labels with the 'Show room labels' checkbox.")
    print("  commands:  home            reset start to launch point")
    print("             building <id>   switch (00800_TEEsavR23oF | 00809_Qpor2mEya8F)")
    print("             quit / exit     stop")
    print("=" * 68)

    while True:
        try:
            user_text = input("\nquery> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[v2] bye.")
            break
        if not user_text:
            continue
        low = user_text.lower()
        if low in ("quit", "exit", "q"):
            print("[v2] bye.")
            break
        if low == "home":
            state["last_goal"] = None
            if state["asset"] is not None:
                render_drone_marker(state["asset"], state["home"])  # drone back at home
            if bridge is not None and state["sim_transform"] is not None:
                hu = state["sim_transform"].mosaic_to_unity(np.asarray(state["home"]))
                bridge.set_position(float(hu[0]), float(hu[1]), float(hu[2]), 0.0)
            print(f"[v2] start reset to home {np.round(state['home'],2)}")
            continue
        if low.startswith("building "):
            bid = user_text.split(None, 1)[1].strip()
            if bid not in ALLOWED_BUILDINGS:
                print(f"[v2] building must be one of {sorted(ALLOWED_BUILDINGS)}")
                continue
            load_building_scene(bid)
            continue
        try:
            run_query(user_text)
        except Exception as e:  # keep the REPL alive on any per-query failure
            print(f"[v2] error: {e}")
            import traceback
            traceback.print_exc()

    if bridge is not None:
        bridge.close()


if __name__ == "__main__":
    main()
