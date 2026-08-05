"""patrol — 자연어 한 줄 → 구역 순찰 → 보고서 (터미널 디버그 경로).

    "현우방만 탐색해줘"
        -> 순찰 구역 해석         (patrol_intent + room_index)
        -> 방마다 A* 구간 비행 + 360° 스캔     (patrol_mission)
        -> 사람 탐지 시 정지 -> 라이트 온 -> 사진 기록 -> 알림
        -> 복귀·착륙
        -> 순찰 보고서 md/html/json           (patrol_report)

확인 단계는 없다: 순찰 구역은 웹 평면도에서 이륙 전에 고른다.

방 정보는 `data/final_npy` 에서 읽는다 (LitePT 사전계산 결과, litept_backend).
연속 미션이라 각 쿼리는 드론의 **현재 시뮬 위치**에서 시작한다 (상태 수신 실패
시 직전 목표 → 홈 순으로 폴백).

The model is NEVER loaded here — it lives behind an OpenAI-compatible endpoint
(by default Ollama on this PC; `llm_server/serve.py` on the GPU box, or anything
else that speaks the same protocol). So this process needs no torch, which is
what lets it run next to Unity on the operator's PC:

    python patrol/server.py --sim --unity-host 127.0.0.1
    python patrol/server.py --sim --llm-url http://<GPU서버>:8000/v1 \
        --llm-model Qwen/Qwen2.5-3B-Instruct
    python patrol/server.py --sim --llm-url ""          # 오프라인 모드

Run from the repo root (see requirements.txt / README.md §1).

This REPL is the debug path. The web console drives the same pipeline through
the API server — do not run both at once, they would fight over the UDP bridge.
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

# 로그가 한국어라 윈도우 기본 cp949 로는 '—' 같은 글자에서 죽는다.
for _s in (sys.stdout, sys.stderr):
    _s.reconfigure(encoding="utf-8", errors="replace")

from patrol import (patrol_intent, patrol_mission, patrol_report,  # noqa: E402
                    planner, room_index)
from patrol.detect_events import DetectionListener  # noqa: E402
from patrol.litept_backend import LitePTBackend  # noqa: E402
from patrol.remote_llm import (DEFAULT_LLM_MODEL,  # noqa: E402  (stdlib only)
                               DEFAULT_LLM_URL, make_llm, resolve_llm)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir",
                    default=str(_THIS.parents[1] / "data" / "final_npy"),
                    help="LitePT output dir: detections.json + per-room npy.")
    ap.add_argument("--building", default="00809_Qpor2mEya8F",
                    help="Building id (transform lookup + program metadata).")
    ap.add_argument("--llm-url", default=DEFAULT_LLM_URL,
                    help="OpenAI-compatible endpoint. Defaults to Ollama on "
                         'this PC; "gpu" = the school GPU box (VPN), "" = '
                         "offline. The model never loads here.")
    ap.add_argument("--llm-model", default=DEFAULT_LLM_MODEL,
                    help="Model name to ask that endpoint for.")
    ap.add_argument("--llm-api-key", default=None,
                    help="Bearer token, if the LLM server requires one.")
    ap.add_argument("--llm-timeout", type=float, default=60.0,
                    help="Per-request timeout, seconds.")
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
    ap.add_argument("--home-xyz", type=float, nargs=3, default=None,
                    metavar=("X", "Y", "Z"),
                    help="World-meter launch point. Default: first room's "
                         "centroid, 1 m above its floor.")
    # Unity simulator link (simulator/bridge; see README.md §4)
    ap.add_argument("--sim", action="store_true",
                    help="Fly in the Unity Tello simulator.")
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
    # Patrol mode
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
    ap.add_argument("--scan-mode", default="auto",
                    choices=["auto", "unity", "rc"],
                    help="Who spins and detects: unity (PatrolPersonDetection "
                         "+ YOLO), rc (we spin, UDP 9004 detector), or auto — "
                         "try unity once, fall back to rc for the mission.")
    ap.add_argument("--max-rooms", type=int, default=12,
                    help="Cap on rooms per patrol (집 전체 순찰 safety).")
    ap.add_argument("--report-dir", default=str(_THIS.parent / "out" / "reports"))
    ap.add_argument("--no-light", action="store_true",
                    help="Do not send the `light on/off` verb on detection.")
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

    # One LLM handle for the whole session — patrol_intent and patrol_report
    # reuse it rather than making their own.
    # `--llm-url ""` picks the offline parser, same switch as api_server.py.
    llm_url, llm_model, llm_key = resolve_llm(
        args.llm_url, args.llm_model, args.llm_api_key)
    llm = make_llm(llm_url, model_id=llm_model,
                   api_key=llm_key, timeout=args.llm_timeout)
    if llm_url:
        served = llm.ping()   # fail here, not on the user's first query
        print(f"[llm] server ok, models={served or '(no /models route)'}")

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

    # ---------------------------------------------------------- patrol stage
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

        out_dir = patrol_report.report_dir_for(args.report_dir)
        cfg = patrol_mission.PatrolConfig(
            hover_height=args.hover_height,
            scan_deg_per_sec=args.scan_deg_per_sec,
            scan_turns=args.scan_turns,
            scan_mode=args.scan_mode,
            labels=tuple(l.lower() for l in (args.patrol_labels or ())),
            min_conf=args.patrol_min_conf,
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
        status(f"의도 분석 중... ({user_text})")
        p_intent = patrol_intent.parse_patrol(llm, user_text, rooms_index)
        print(f"[route] rooms={p_intent.target_rooms} "
              f"types={p_intent.room_types} floors={p_intent.floors_kr} "
              f"scope={p_intent.scope!r}")
        run_patrol_query(user_text, p_intent)

    # ----------------------------------------------------------------- REPL
    print("\n" + "=" * 68)
    print("  patrol — type a drone command (Korean/English).")
    print("  Unity: C 키 = 1인칭/3인칭 카메라 전환.")
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
