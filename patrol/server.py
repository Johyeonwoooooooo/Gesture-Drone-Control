"""patrol — terminal NL → LitePT detection → user confirm → sim flight.

Two modes, chosen per query by the LLM (patrol.patrol_intent):

FIND (물체 찾기) — single-object search:

    natural-language command
        -> LLM intent parse       (patrol.llm_parser / patrol.remote_llm)
        -> LitePT precomputed detection match      (patrol.litept_backend)
        -> Unity camera previews the candidate; the user clicks [이동] to
           confirm or [다음 후보] to cycle candidates   (simulator/bridge)
        -> A* / RRT* path from the drone's position    (patrol.planner)
        -> Tello SDK command program written to out/   (patrol.sdk_export)
        -> the drone flies the path in the simulator   (simulator/bridge)

PATROL (구역 순찰) — "현우방만 탐색해줘":

    -> rooms resolved from aliases/type/floor   (patrol_intent + room_index)
        -> one preview/confirm of the plan          (Unity [이동]/[다음 후보])
        -> per room: A* leg -> 360° scan, detector ARMED only inside the room
        -> a person detection (UDP 9004, patrol.detect_events) triggers
           hover -> light on -> record photo -> notify   (patrol_mission)
        -> return home, land
        -> 순찰 보고서 md/html/json written to out/reports (patrol_report)

The 2D person detector and the Unity→Python photo transport run as a SEPARATE
process owned by another team member; docs/patrol-agent.md is the contract.

Detections come from `data/final_npy` (LitePT ScanNet-20 instance centers,
see litept_backend). Continuous mission: each query starts from the drone's
current simulator position (fallback: previous goal, then home).

Run from the repo root (see requirements.txt / README.md §1). The model can
load here, or live on another machine behind patrol/llm_serve.py — everything
else in this file is identical either way:

    python patrol/server.py --sim --unity-host 127.0.0.1 --llm-device cuda:1
    python patrol/server.py --sim --unity-host 127.0.0.1 \
        --llm-url http://166.104.223.32:8000/v1      # no torch needed here
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np

# Make `patrol.*` and `simulator.*` importable when run as a plain script.
_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parents[1]))  # repo root

# NOTE: patrol.llm_parser is imported lazily in main() — it pulls in torch, and
# under --llm-url this process is meant to run on a machine that has none.
from patrol import (patrol_intent, patrol_mission, patrol_report,  # noqa: E402
                    planner, room_index, sdk_export)
from patrol.detect_events import DetectionListener  # noqa: E402
from patrol.litept_backend import LitePTBackend  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir",
                    default=str(_THIS.parents[1] / "data" / "final_npy"),
                    help="LitePT output dir: detections.json + per-room npy.")
    ap.add_argument("--building", default="00809_Qpor2mEya8F",
                    help="Building id (transform lookup + program metadata).")
    ap.add_argument("--llm-model", default="Qwen/Qwen2.5-3B-Instruct")
    # With --llm-url the model is NOT loaded here — this process then needs no
    # torch at all, which is what lets the laptop run the pipeline next to
    # Unity while the GPU box only serves the LLM (patrol/llm_serve.py).
    ap.add_argument("--llm-url", default=None,
                    help="OpenAI-compatible endpoint, e.g. "
                         "http://166.104.223.32:8000/v1 . Omit to load the "
                         "model in-process (needs a local GPU + torch).")
    ap.add_argument("--llm-api-key", default=None,
                    help="Bearer token, if the LLM server requires one.")
    ap.add_argument("--llm-timeout", type=float, default=60.0,
                    help="Per-request timeout for --llm-url, seconds.")
    # Below here: in-process mode only, ignored with --llm-url.
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
    # Unity simulator link (simulator/bridge; see README.md §4)
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
    # Patrol mode (docs/patrol-agent.md)
    ap.add_argument("--patrol-port", type=int, default=9004,
                    help="UDP port the 2D detector pushes detection JSON to.")
    ap.add_argument("--patrol-labels", nargs="*", default=["person"],
                    help="Detection labels that trigger the stop/light/photo "
                         "reaction. Empty = accept every label.")
    ap.add_argument("--patrol-min-conf", type=float, default=0.0)
    ap.add_argument("--patrol-cooldown", type=float, default=3.0,
                    help="Seconds before the same label is recorded again.")
    ap.add_argument("--room-aliases", default=None,
                    help="Room nickname JSON. Default: patrol/room_aliases.json")
    ap.add_argument("--hover-height", type=float, default=1.2,
                    help="Scan hover height above the room floor (meters).")
    ap.add_argument("--scan-deg-per-sec", type=float, default=50.0)
    ap.add_argument("--scan-turns", type=float, default=1.0,
                    help="Full revolutions per room during the scan.")
    ap.add_argument("--max-rooms", type=int, default=12,
                    help="Cap on rooms per patrol (집 전체 순찰 safety).")
    ap.add_argument("--report-dir", default=str(_THIS.parent / "out" / "reports"))
    ap.add_argument("--no-light", action="store_true",
                    help="Do not send the `light on/off` verb on detection.")
    ap.add_argument("--no-patrol-confirm", action="store_true",
                    help="Start patrolling immediately, without the Unity "
                         "[이동] confirmation.")
    ap.add_argument("--viz-dir", default=None,
                    help="Write planned_path_3d.json / flight_trajectory_3d.json "
                         "here. Point at simulator/tello_simulator/Assets/Resources "
                         "to draw the route + detection markers in Unity.")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    print(f"[patrol] loading LitePT detections from {data_dir} ...")
    backend = LitePTBackend(data_dir)
    print(f"[patrol] {len(backend.detections)} detections, "
          f"{len(backend.room_dirs)} rooms")

    # One LLM handle for the whole session, remote or local — patrol_intent and
    # patrol_report reuse it rather than making their own.
    if args.llm_url:
        from patrol.remote_llm import RemoteLLMParser  # noqa: E402  (no torch)
        llm = RemoteLLMParser(args.llm_url, model_id=args.llm_model,
                              api_key=args.llm_api_key, timeout=args.llm_timeout)
        served = llm.ping()   # fail here, not on the user's first query
        print(f"[llm] server ok, models={served or '(no /models route)'}")
    else:
        from patrol.llm_parser import LocalLLMParser  # noqa: E402  (needs torch)
        llm_device = args.llm_device if args.llm_device_map is None else "cuda:0"
        llm = LocalLLMParser(model_id=args.llm_model, device=llm_device,
                             dtype=args.llm_dtype, device_map=args.llm_device_map)

    # ---------------- Unity simulator link (lazy: only with --sim) ----------
    bridge = None
    coord_transform = follow_path = None  # modules, bound under --sim
    if args.sim:
        if not args.unity_host:
            raise SystemExit("--sim requires --unity-host <ip-of-unity-machine>")
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
    rooms_index = room_index.build_room_index(backend, args.room_aliases)
    n_alias = sum(1 for r in rooms_index.values() if r.aliases)
    print(f"[patrol] room index: {len(rooms_index)} rooms ({n_alias} with 별칭)")

    print(f"[patrol] merging point clouds (stride={args.point_stride}) ...")
    t0 = time.time()
    points_world = backend.load_points(stride=args.point_stride)
    print(f"[patrol] voxelizing {len(points_world)} points "
          f"(res={args.resolution} margin={args.margin} sample={args.sample}) ...")
    gm = planner.voxelize(points_world, args.resolution, args.margin, args.sample)
    home = (np.asarray(args.home_xyz, dtype=float) if args.home_xyz is not None
            else backend.default_home())

    # Detection ingest starts DISARMED — nothing is recorded until the drone is
    # inside a patrol room (see patrol_mission.run_patrol).
    listener = DetectionListener(
        port=args.patrol_port, cooldown=args.patrol_cooldown,
        labels=tuple(args.patrol_labels or ()), min_conf=args.patrol_min_conf)
    try:
        listener.start()
        print(f"[patrol] detection ingest on udp/{args.patrol_port} "
              f"(labels={list(args.patrol_labels) or 'ALL'}) — disarmed")
    except OSError as e:
        print(f"[patrol] WARNING: cannot bind udp/{args.patrol_port} ({e}) — "
              "patrols will run without detection reactions")
        listener = None
    print(f"[patrol] scene ready: grid={gm.shape}, home={np.round(home, 2)} "
          f"({time.time()-t0:.1f}s)")

    state: Dict[str, object] = {"last_goal": None, "last_patrol": None}

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

    # ---------------------------------------------------------- patrol stage
    def confirm_patrol(rooms) -> bool:
        """Preview the patrol plan in Unity; [이동] starts it, [다음 후보]
        previews the next room in the planned order, timeout cancels."""
        if args.no_patrol_confirm or bridge is None or sim_tf is None:
            return True
        n = len(rooms)
        i = 0
        while True:
            room = rooms[i]
            tag = f"순찰 {i + 1}/{n} — {room.display}"
            status(f"순찰 계획 {i + 1}/{n}: {room.display} — "
                   f"[이동]=순찰 시작 / [다음 후보]=다음 구역 미리보기")
            goal = room_index.scan_pose(room, args.hover_height, gm)
            cu = sim_tf.mosaic_to_unity(goal)
            bridge.drain_events()
            bridge.preview(float(cu[0]), float(cu[1]), float(cu[2]), label=tag)
            ev = bridge.wait_for_event(args.confirm_timeout)
            if ev == "confirm":
                bridge.preview_off()
                return True
            if ev == "next":
                i = (i + 1) % n
                continue
            bridge.preview_off()
            status("확인 시간 초과 — 순찰을 취소했습니다")
            return False

    def run_patrol_query(user_text: str, intent) -> None:
        rooms, why = patrol_intent.resolve_rooms(
            intent, rooms_index, user_text, max_rooms=args.max_rooms)
        if not rooms:
            status("순찰할 구역을 특정하지 못했습니다")
            print("[patrol] 구역 미해석 — 사용 가능한 방:")
            print(room_index.room_directory_text(rooms_index))
            return

        start = drone_world_pos()
        rooms = room_index.order_rooms(rooms, start)
        print(f"[patrol] {why}: " + " -> ".join(r.display for r in rooms))
        status(f"순찰 계획: {len(rooms)}개 구역 ({why})")

        if bridge is None or sim_tf is None:
            print("[patrol] --sim 없이는 비행할 수 없습니다 (계획만 출력).")
            return
        if not confirm_patrol(rooms):
            return

        out_dir = patrol_report.report_dir_for(args.report_dir)
        cfg = patrol_mission.PatrolConfig(
            hover_height=args.hover_height,
            scan_deg_per_sec=args.scan_deg_per_sec,
            scan_turns=args.scan_turns,
            light_on_detect=not args.no_light,
            return_home=intent.return_home,
            speed=float(args.sim_speed), rc_limit=int(args.sim_rc_limit),
            algo=args.algo, leg_timeout=(args.sim_timeout or None),
            events_dir=out_dir / "events",
            viz_dir=Path(args.viz_dir) if args.viz_dir else None,
        )
        result = patrol_mission.run_patrol(
            bridge, sim_tf, points_world, gm, rooms, home, cfg,
            listener=listener, on_status=status,
            follow_path_mod=follow_path)

        status(f"순찰 완료 — 탐지 {len(result.events)}건, 보고서 작성 중...")
        report_path = patrol_report.build_report(
            result, rooms, user_text, out_dir, llm=llm, intent=intent)
        state["last_patrol"] = (result, rooms, user_text, out_dir)
        state["last_goal"] = None if result.returned_home else state["last_goal"]
        print(f"[patrol] {result.rooms_reached}/{len(rooms)} 구역 도달, "
              f"탐지 {len(result.events)}건, {result.distance_m:.1f} m, "
              f"{result.duration_s:.0f}s, 충돌 {result.collisions}회")
        if result.listener_stats:
            print(f"[patrol] detections {result.listener_stats}")
        print(f"[report] {report_path}/report.md  (+ report.html, report.json)")
        status(f"보고서 생성 완료 — 탐지 {len(result.events)}건")

    # ------------------------------------------------------------- per-query run
    def run_query(user_text: str) -> None:
        # 0. patrol or find?
        status(f"의도 분석 중... ({user_text})")
        p_intent = patrol_intent.parse_patrol(llm, user_text, rooms_index)
        print(f"[route] mode={p_intent.mode} rooms={p_intent.target_rooms} "
              f"types={p_intent.room_types} floors={p_intent.floors_kr} "
              f"scope={p_intent.scope!r}")
        if p_intent.is_patrol:
            run_patrol_query(user_text, p_intent)
            return

        # 1. LLM intent parse (object find)
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
        from patrol.litept_backend import INSTANCE_CLASSES
        return INSTANCE_CLASSES

    # ----------------------------------------------------------------- REPL
    print("\n" + "=" * 68)
    print("  patrol — type a drone command (Korean/English).")
    print("  Unity: 후보 프리뷰에서 [이동]=비행 시작, [다음 후보]=후보 전환,")
    print("         C 키 = 1인칭/3인칭 카메라 전환.")
    print("  예시:  거실 소파 찾아줘        (물체 찾기)")
    print("         현우방만 탐색해줘       (구역 순찰 + 보고서)")
    print("  commands:  home            drone back to the launch point")
    print("             rooms           list patrol-able rooms")
    print("             report          rebuild the last patrol report")
    print("             quit / exit     stop")
    print("=" * 68)

    while True:
        try:
            user_text = input("\nquery> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[patrol] bye.")
            break
        if not user_text:
            continue
        low = user_text.lower()
        if low in ("quit", "exit", "q"):
            print("[patrol] bye.")
            break
        if low == "home":
            state["last_goal"] = None
            teleport_home()
            print(f"[patrol] drone reset to home {np.round(home, 2)}")
            continue
        if low == "rooms":
            print(room_index.room_directory_text(rooms_index))
            continue
        if low == "report":
            last = state.get("last_patrol")
            if last is None:
                print("[patrol] 아직 순찰 기록이 없습니다.")
                continue
            # Rebuild in place: the detection photos already live in out/events.
            result, rooms, q, out_dir = last
            out = patrol_report.build_report(result, rooms, q, out_dir, llm=llm)
            print(f"[report] {out}/report.md")
            continue
        try:
            run_query(user_text)
        except Exception as e:  # keep the REPL alive on any per-query failure
            print(f"[patrol] error: {e}")
            import traceback
            traceback.print_exc()

    if listener is not None:
        listener.close()
    if bridge is not None:
        bridge.close()


if __name__ == "__main__":
    main()
