"""webapp_llm_v2 — terminal NL → LitePT detection → user confirm → sim flight.

Pipeline per query (typed in the TERMINAL; visual feedback lives in the Unity
simulator — no viser):

    natural-language command
        -> local LLM intent parse                  (webapp_llm.llm_parser)
        -> LitePT precomputed detection match      (webapp_llm_v2.litept_backend)
        -> Unity camera previews the candidate; the user clicks [이동] to
           confirm or [다음 후보] to cycle candidates   (simulator/bridge)
        -> A* / RRT* path from the drone's position    (webapp_llm_v2.planner)
        -> Tello SDK command program written to out/   (webapp_llm_v2.sdk_export)
        -> the drone flies the path in the simulator   (simulator/bridge)

Detections come from `data/final_npy` (LitePT ScanNet-20 instance centers,
see litept_backend). Continuous mission: each query starts from the drone's
current simulator position (fallback: previous goal, then home).

Run (in the `mosaic3d` conda env):
    python 3D-segmentation/webapp_llm_v2/server.py \
        --sim --unity-host 127.0.0.1 --llm-device cuda:1
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np

# Make sibling packages importable.
_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parents[1]))  # .../3D-segmentation

from webapp_llm.llm_parser import LocalLLMParser  # noqa: E402

from webapp_llm_v2 import planner, sdk_export  # noqa: E402
from webapp_llm_v2.litept_backend import LitePTBackend  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir",
                    default=str(_THIS.parents[2] / "data" / "final_npy"),
                    help="LitePT output dir: detections.json + per-room npy.")
    ap.add_argument("--building", default="00809_Qpor2mEya8F",
                    help="Building id (transform lookup + program metadata).")
    ap.add_argument("--llm-model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--llm-device", default="cuda:1")
    ap.add_argument("--llm-dtype", default="float16",
                    choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--llm-device-map", default=None)
    # planner
    ap.add_argument("--algo", default="astar", choices=["astar", "rrt"])
    ap.add_argument("--resolution", type=float, default=0.15)
    ap.add_argument("--margin", type=int, default=1)
    ap.add_argument("--sample", type=int, default=1,
                    help="Extra voxelize subsampling; the backend already "
                         "strides the merged cloud (see --point-stride).")
    ap.add_argument("--point-stride", type=int, default=4,
                    help="Stride when merging the per-room coord.npy clouds.")
    ap.add_argument("--rrt-iter", type=int, default=8000)
    # tello / output
    ap.add_argument("--tello-speed", type=int, default=40)
    ap.add_argument("--home-xyz", type=float, nargs=3, default=None,
                    metavar=("X", "Y", "Z"),
                    help="World-meter launch point. Default: first room's "
                         "centroid, 1 m above its floor.")
    ap.add_argument("--out-dir", default=str(_THIS.parent / "out"))
    # Unity simulator link (simulator/bridge; see README-integration.md)
    ap.add_argument("--sim", action="store_true",
                    help="Preview + fly in the Unity Tello simulator.")
    ap.add_argument("--unity-host", default=None,
                    help="IP of the machine running Unity (required with --sim).")
    ap.add_argument("--unity-port", type=int, default=9000)
    ap.add_argument("--unity-state-port", type=int, default=9002)
    ap.add_argument("--unity-local-port", type=int, default=9001)
    ap.add_argument("--sim-transform", default=None,
                    help="Path to a calibrated transform JSON. Default: "
                         "simulator/bridge/transforms/<building>.json")
    ap.add_argument("--sim-speed", type=float, default=2.0,
                    help="Flight speed in Unity units/s (house at scale 5, "
                         "so 2.0 u/s = 0.4 m/s real).")
    ap.add_argument("--sim-rc-limit", type=int, default=30)
    ap.add_argument("--sim-timeout", type=float, default=0.0,
                    help="Flight timeout seconds; 0 = auto from path length.")
    ap.add_argument("--sim-no-status", action="store_true",
                    help="Do not push status text to the Unity banner.")
    ap.add_argument("--confirm-timeout", type=float, default=120.0,
                    help="Seconds to wait for the user's [이동]/[다음 후보] "
                         "click per candidate.")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    print(f"[v2] loading LitePT detections from {data_dir} ...")
    backend = LitePTBackend(data_dir)
    print(f"[v2] {len(backend.detections)} detections, "
          f"{len(backend.room_dirs)} rooms")

    llm_device = args.llm_device if args.llm_device_map is None else "cuda:0"
    llm = LocalLLMParser(model_id=args.llm_model, device=llm_device,
                         dtype=args.llm_dtype, device_map=args.llm_device_map)

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

    sim_tf = None
    if args.sim:
        try:
            if args.sim_transform:
                sim_tf = coord_transform.SimTransform.load(args.sim_transform)
            else:
                sim_tf = coord_transform.load_building_transform(args.building)
            if sim_tf.meta.get("provisional"):
                print("[sim] WARNING: transform is PROVISIONAL (not yet "
                      "calibrated against a Unity voxel-map export). Verify "
                      "with simulator/bridge/calibrate_transform.py.")
        except FileNotFoundError as e:
            raise SystemExit(f"[sim] {e}")

    def status(text: str) -> None:
        """Progress line: terminal always, Unity banner when connected."""
        print(f"[status] {text}")
        if bridge is not None and not args.sim_no_status:
            try:
                bridge.send_status(text)
            except Exception:
                pass

    # ------------------------------------------------------------ scene setup
    print(f"[v2] merging point clouds (stride={args.point_stride}) ...")
    t0 = time.time()
    points_world = backend.load_points(stride=args.point_stride)
    print(f"[v2] voxelizing {len(points_world)} points "
          f"(res={args.resolution} margin={args.margin} sample={args.sample}) ...")
    gm = planner.voxelize(points_world, args.resolution, args.margin, args.sample)
    home = (np.asarray(args.home_xyz, dtype=float) if args.home_xyz is not None
            else backend.default_home())
    print(f"[v2] scene ready: grid={gm.shape}, home={np.round(home, 2)} "
          f"({time.time()-t0:.1f}s)")

    state: Dict[str, object] = {"last_goal": None}

    def drone_world_pos() -> np.ndarray:
        """Mission start: live sim position, else previous goal, else home."""
        if bridge is not None and sim_tf is not None:
            s = bridge.get_latest_state()
            if s is not None:
                return sim_tf.unity_to_mosaic(
                    np.array([s.x, s.y, s.z], dtype=float))
        if state["last_goal"] is not None:
            return np.asarray(state["last_goal"], dtype=float)
        return home.copy()

    def teleport_home() -> None:
        if bridge is not None and sim_tf is not None:
            hu = sim_tf.mosaic_to_unity(home)
            bridge.set_position(float(hu[0]), float(hu[1]), float(hu[2]), 0.0)

    teleport_home()

    # ---------------------------------------------------------- confirm stage
    def confirm_candidate(cands) -> Optional[int]:
        """Preview candidates until the user confirms one. Returns the index
        into `cands`, or None on timeout/cancel."""
        n = len(cands)
        i = 0
        seen = 0
        while True:
            det = cands[i]
            tag = f"{det.label_kr} ({i + 1}/{n}) — {det.room_kr} {det.room_name}"
            status(f"후보 {i + 1}/{n}: {det.label_kr} @ {det.room_kr} "
                   f"{det.room_name} — [이동] 또는 [다음 후보]를 눌러주세요")
            print(f"[confirm] {det.describe()}")
            if bridge is not None and sim_tf is not None:
                cu = sim_tf.mosaic_to_unity(det.center)
                bridge.drain_events()
                bridge.preview(float(cu[0]), float(cu[1]), float(cu[2]),
                               label=tag)
                ev = bridge.wait_for_event(args.confirm_timeout)
                if ev == "confirm":
                    bridge.preview_off()
                    return i
                if ev == "next":
                    i = (i + 1) % n
                    seen += 1
                    continue
                bridge.preview_off()
                status("확인 시간 초과 — 쿼리를 취소했습니다")
                return None
            # terminal-only fallback (no Unity)
            try:
                ans = input("[이동=y / 다음=n / 취소=q] > ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                return None
            if ans in ("y", "yes", "이동"):
                return i
            if ans in ("q", "quit", "취소"):
                return None
            i = (i + 1) % n

    # ------------------------------------------------------------- per-query run
    def run_query(user_text: str) -> None:
        # 1. LLM intent parse
        status(f"의도 분석 중... ({user_text})")
        t1 = time.time()
        intent = llm.parse(user_text,
                           room_directory=backend.room_directory_text())
        t_llm = time.time() - t1
        print(f"[intent] object={intent.target_object!r} "
              f"room={intent.target_room or intent.location_hint or 'any'} "
              f"action={intent.action} return_home={intent.return_home} "
              f"({t_llm:.2f}s)")

        # 2. Detection match (LitePT closed-set)
        label = backend.resolve_label(intent.target_object, user_text)
        if label is None:
            status(f"'{intent.target_object}' 는 인식 가능한 물체가 아닙니다")
            print(f"[query] unresolvable target {intent.target_object!r} — "
                  f"classes: {', '.join(backend_classes())}")
            return
        room_name, room_type = backend.resolve_room(
            intent.target_room, intent.location_hint, user_text)
        if intent.scope == "building":
            room_name = room_type = None
        status(f"'{label}' 탐색 중 (LitePT detection)...")
        cands = backend.candidates(label, room_name, room_type)
        if not cands:
            status(f"'{label}' 를 찾지 못했습니다")
            return
        if (room_name or room_type) and not cands[0].room_match:
            status("요청한 방에는 없어 건물 전체에서 찾았습니다")
        print(f"[detect] {len(cands)} candidate(s) for '{label}'"
              + (f" (room={room_name or room_type})" if room_name or room_type
                 else ""))

        # 3. User confirm (Unity preview + buttons)
        idx = confirm_candidate(cands)
        if idx is None:
            return
        det = cands[idx]
        goal_world = det.center + np.array([0.0, 0.0, 0.5])  # hover above it
        print(f"[target] {det.describe()} -> goal={np.round(goal_world, 2)}")

        # 4. Path plan from the drone's current position
        status("경로 계산 중...")
        start_world = drone_world_pos()
        t2 = time.time()
        path, info, _ = planner.plan_path(
            points_world, start_world, goal_world, algo=args.algo,
            resolution=args.resolution, margin=args.margin, sample=args.sample,
            rrt_iter=args.rrt_iter, gm=gm)
        t_plan = time.time() - t2
        if path is None:
            status("경로 계산 실패")
            print(f"[plan] {args.algo} FAILED ({info.get('reason', '?')}, "
                  f"{t_plan:.2f}s) — no path saved.")
            return
        print(f"[plan] {args.algo}: {info['n_waypoints']} waypoints, "
              f"{info['length_m']:.2f} m ({t_plan:.2f}s)")

        # 5. Emit Tello SDK program
        ts = time.strftime("%Y%m%d_%H%M%S")
        program = sdk_export.build_tello_program(
            path, action=intent.action, return_home=intent.return_home,
            home_world=home, start_world=start_world, goal_world=goal_world,
            target_object=label, clip_prompt=intent.clip_prompt,
            query=user_text, algo=args.algo, building=args.building,
            speed=args.tello_speed, timestamp=ts)
        out_path = sdk_export.save_program(program, args.out_dir)
        print(f"[sdk] {len(program['commands'])} commands -> {out_path}")

        # 6. Fly in the simulator
        if bridge is not None and sim_tf is not None:
            wps_unity = sim_tf.mosaic_to_unity(np.asarray(path, dtype=float))
            length_u = float(np.linalg.norm(
                np.diff(wps_unity, axis=0), axis=1).sum())
            status(f"비행 중... ({info['length_m']:.1f} m)")
            print(f"[sim] flying {length_u:.1f} u at {args.sim_speed} u/s "
                  f"(~{length_u / max(1e-3, args.sim_speed):.0f}s) ...")
            res = follow_path.fly_mission(
                bridge, [tuple(p) for p in wps_unity],
                setpos_start=False,  # continuous mission: fly from where it is
                max_speed=float(args.sim_speed),
                rc_limit=int(args.sim_rc_limit),
                timeout_sec=(args.sim_timeout or None), on_status=status)
            status("도착 — 착륙 완료" if res.success
                   else f"비행 중단 ({res.reason})")
            print(f"[sim] {res.reason}: err={res.final_error_u:.2f}u "
                  f"collisions={res.collision_count} rc={res.rc_commands_sent} "
                  f"{res.duration_s:.0f}s")
            if not res.success:
                return  # keep last_goal where the mission actually is
        state["last_goal"] = goal_world

    def backend_classes():
        from webapp_llm_v2.litept_backend import INSTANCE_CLASSES
        return INSTANCE_CLASSES

    # ----------------------------------------------------------------- REPL
    print("\n" + "=" * 68)
    print("  webapp_llm_v2 — type a drone command (Korean/English).")
    print("  Unity: 후보 프리뷰에서 [이동]=비행 시작, [다음 후보]=후보 전환,")
    print("         C 키 = 1인칭/3인칭 카메라 전환.")
    print("  commands:  home            drone back to the launch point")
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
            teleport_home()
            print(f"[v2] drone reset to home {np.round(home, 2)}")
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
