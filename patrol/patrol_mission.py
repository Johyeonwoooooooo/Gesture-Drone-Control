"""Patrol mission executor — fly the rooms, scan them, react to detections.

One mission = for each room in order:

    plan A* leg  ->  fly it (no detection in transit)
                 ->  scan the room 360°, reacting to what it finds
    ... then a final leg back home, then land.

The scan itself belongs to Unity: it spins, captures its own camera frames and
runs YOLO over TCP 9100 (`PatrolPersonDetection.cs`), then reports `detect` and
`scan_done` events. This side sends `scan` and reacts. Older builds have no
`scan` verb — and since unknown verbs are acked "ok", the only way to find out
is to ask and see whether anything answers, so `scan_mode="auto"` probes once
and falls back to driving the spin with `rc` + the UDP 9004 listener.

Detection reaction (the flow's "탐지 → 정지 → 라이트 온 → 사진"):

    rc 0 0 0 0            hover in place, abandoning the spin
    light on              (Unity ignores the verb today)
    settle                give the light a frame to land
    record the photo      Unity/the detector captured it and sent image_path
    notify                Unity banner + terminal
    light off, resume     finish the remaining rotation

Why not `follow_path.fly_mission`? It does takeoff→follow→land per call, and
`TelloSimulator.cs` only applies rc (including yaw) while `isFlying`. Landing
between rooms would make the 360° scan silently do nothing. So the mission owns
takeoff/land itself and calls the lower-level `follow_path.follow_path` per leg.
"""
from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

try:
    from patrol import planner
    from patrol.detect_events import DetectionEvent, DetectionListener
    from patrol.room_index import RoomInfo, scan_pose
except ImportError:  # plain-script import path
    import planner  # type: ignore
    from detect_events import DetectionEvent, DetectionListener  # type: ignore
    from room_index import RoomInfo, scan_pose  # type: ignore

# TelloSimulator.cs `rotationSpeed` — deg/s commanded by rc yaw = 100.
UNITY_ROTATION_SPEED = 100.0


@dataclass
class PatrolConfig:
    hover_height: float = 1.2        # meters above the room floor
    scan_deg_per_sec: float = 50.0
    scan_turns: float = 1.0          # full revolutions per room
    scan_mode: str = "auto"          # unity | rc | auto (probe, then fall back)
    scan_probe_sec: float = 3.0      # no event by then => build has no `scan`
    labels: tuple = ("person",)      # accepted detect labels ("" = all)
    min_conf: float = 0.0
    settle_sec: float = 0.8          # pause after light-on before recording
    light_on_detect: bool = True
    light_off_after: bool = True
    max_events_per_room: int = 5
    return_home: bool = True
    speed: float = 6.0               # Unity u/s (= 1.2 m/s at house scale 5)
    rc_limit: int = 60               # >= speed*100/15, or rc clipping caps it
    algo: str = "astar"
    leg_timeout: Optional[float] = None
    abort_on_collision: bool = False
    events_dir: Optional[Path] = None   # where detection photos are copied
    viz_dir: Optional[Path] = None      # Unity Resources/ for the path renderers


@dataclass
class RoomVisit:
    room_name: str
    display: str
    reached: bool
    reason: str                      # arrived | timeout | plan_failed | ...
    scan_completed: bool
    scan_degrees: float
    events: int
    leg_length_m: float
    duration_s: float


@dataclass
class PatrolResult:
    started_at: float = 0.0
    finished_at: float = 0.0
    rooms: List[RoomVisit] = field(default_factory=list)
    events: List[DetectionEvent] = field(default_factory=list)
    distance_m: float = 0.0
    collisions: int = 0
    returned_home: bool = False
    aborted_reason: str = ""
    planned_path_world: List[List[float]] = field(default_factory=list)   # mosaic m
    planned_path_unity: List[List[float]] = field(default_factory=list)   # Unity u
    trajectory_unity: List[List[float]] = field(default_factory=list)
    listener_stats: Dict[str, int] = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return max(0.0, self.finished_at - self.started_at)

    @property
    def rooms_reached(self) -> int:
        return sum(1 for r in self.rooms if r.reached)


# --------------------------------------------------------------------- scanning

def _yaw_delta(a: float, b: float) -> float:
    """Signed shortest angular difference b-a in degrees, wrap-safe."""
    return (b - a + 180.0) % 360.0 - 180.0


def scan_room(bridge, listener: Optional[DetectionListener], cfg: PatrolConfig,
              on_status: Callable[[str], None],
              on_event: Optional[Callable[[DetectionEvent], None]] = None,
              ) -> tuple[bool, float, int, bool]:
    """Scan the room the drone is hovering in.

    -> (completed, degrees, n_events, unity_did_it)

    Preferred path: Unity spins itself and runs the detector on its own camera
    frames (`scan` verb → `detect`/`scan_done` events). It owns the frames, so
    its boxes are real pixel coordinates rather than something we guessed.

    Fallback: we spin it with `rc` and take detections from the UDP 9004
    listener. Used when the Unity build predates the `scan` verb — which we can
    only find out by asking, since unknown verbs are acked "ok" and dropped.
    `cfg.scan_mode` pins the choice if you already know which build you have.
    """
    if cfg.scan_mode in ("unity", "auto"):
        bridge.drain_events()
        bridge.start_scan(cfg.scan_deg_per_sec, cfg.scan_turns)
        done, degrees, events, answered = _await_unity_scan(
            bridge, cfg, on_status, on_event)
        if answered:
            return done, degrees, events, True
        if cfg.scan_mode == "unity":
            on_status("스캔 실패: Unity가 scan 에 응답하지 않음")
            return False, 0.0, 0, True
        on_status("Unity 쪽 scan 응답 없음 — rc 회전으로 대체")
        bridge.stop_scan()

    done, degrees, events = _scan_by_rc(bridge, listener, cfg, on_status, on_event)
    return done, degrees, events, False


def _await_unity_scan(bridge, cfg: PatrolConfig,
                      on_status: Callable[[str], None],
                      on_event: Optional[Callable[[DetectionEvent], None]],
                      ) -> tuple[bool, float, int, bool]:
    """Consume detect/scan_done while Unity sweeps. -> (..., answered).

    `answered` is False when no `scan_started` came back within
    `scan_probe_sec`: that build has no `scan` verb (it acked "ok" and dropped
    the packet), so the caller falls back.

    It has to be an explicit ack and not just "any event". A clean room emits
    nothing at all until `scan_done` — at 50°/s that is 7 s away, so a probe
    waiting for any event would call every quiet room a missing verb.
    """
    target_deg = 360.0 * max(0.1, cfg.scan_turns)
    full_timeout = 3.0 * target_deg / max(1.0, cfg.scan_deg_per_sec) + 10.0
    answered = False
    events = 0
    t0 = time.time()

    while True:
        waited = time.time() - t0
        if not answered and waited > cfg.scan_probe_sec:
            return False, 0.0, 0, False
        if waited > full_timeout:
            on_status(f"스캔 타임아웃 (Unity, {waited:.0f}s)")
            bridge.stop_scan()
            return False, 0.0, events, True

        ev = bridge.wait_for_event(0.2)
        if ev is None:
            continue
        kind = ev.get("event")
        if kind == "scan_started":
            answered = True
            continue
        if not answered and kind in ("scan_done", "detect"):
            # Older stub/build that answers without acking first — still proof
            # the verb exists.
            answered = True
        if kind == "scan_done":
            return True, float(ev.get("degrees", target_deg)), events, True
        if kind != "detect":
            continue
        if events >= cfg.max_events_per_room:
            continue
        det = DetectionEvent.from_payload(ev)
        if det is None or not _label_wanted(det, cfg):
            continue
        if on_event is not None:
            on_event(det)
        events += 1
        if events == cfg.max_events_per_room:
            on_status("이 방의 탐지 기록 상한 도달 — 스캔 계속")


def _label_wanted(det: DetectionEvent, cfg: PatrolConfig) -> bool:
    if det.conf < cfg.min_conf:
        return False
    return not cfg.labels or det.label in cfg.labels


def _scan_by_rc(bridge, listener: Optional[DetectionListener], cfg: PatrolConfig,
                on_status: Callable[[str], None],
                on_event: Optional[Callable[[DetectionEvent], None]] = None,
                ) -> tuple[bool, float, int]:
    """Spin in place with rc, watching UDP 9004. -> (completed, degrees, n).

    Rotation is closed on the simulator's own reported yaw (accumulated
    wrap-safe deltas), not on elapsed time, so a slow frame rate or the C#
    SmoothDamp ramp cannot cut the sweep short.
    """
    target_deg = 360.0 * max(0.1, cfg.scan_turns)
    rc_yaw = int(max(1, min(100, round(cfg.scan_deg_per_sec * 100.0
                                       / UNITY_ROTATION_SPEED))))
    timeout = 3.0 * target_deg / max(1.0, cfg.scan_deg_per_sec) + 10.0

    state = bridge.wait_for_state(timeout=3.0)
    if state is None:
        on_status("스캔 실패: 시뮬레이터 상태 수신 없음")
        return False, 0.0, 0

    last_yaw = state.yaw
    turned = 0.0
    events = 0
    t0 = time.time()
    dt = 0.05
    try:
        while turned < target_deg:
            if time.time() - t0 > timeout:
                on_status(f"스캔 타임아웃 ({turned:.0f}°/{target_deg:.0f}°)")
                bridge.send_rc(0, 0, 0, 0)
                return False, turned, events

            new_events = listener.pop_all() if listener is not None else []
            if new_events and on_event is not None:
                bridge.send_rc(0, 0, 0, 0)          # 1. 정지 (hover)
                for ev in new_events[:max(0, cfg.max_events_per_room - events)]:
                    on_event(ev)
                    events += 1
                if events >= cfg.max_events_per_room:
                    on_status("이 방의 탐지 기록 상한 도달 — 스캔 계속")

            bridge.send_rc(0, 0, 0, rc_yaw)
            time.sleep(dt)

            s = bridge.get_latest_state()
            if s is not None:
                turned += abs(_yaw_delta(last_yaw, s.yaw))
                last_yaw = s.yaw
        return True, turned, events
    finally:
        bridge.send_rc(0, 0, 0, 0)


# ------------------------------------------------------------------- reactions

def react_to_detection(bridge, ev: DetectionEvent, room: RoomInfo,
                       cfg: PatrolConfig, sim_tf, on_status: Callable[[str], None],
                       index: int) -> DetectionEvent:
    """정지 → 라이트 온 → (사진 기록) → 알림. Returns the annotated event."""
    bridge.send_rc(0, 0, 0, 0)                      # 1. hover (already sent, idempotent)
    if cfg.light_on_detect:
        try:
            bridge.set_light(True)                  # 2. light on
        except Exception:
            pass
    time.sleep(max(0.0, cfg.settle_sec))            # 3. let the light land

    state = bridge.get_latest_state()
    if state is not None:
        ev.drone_unity = [state.x, state.y, state.z]
        if sim_tf is not None:
            try:
                w = sim_tf.unity_to_mosaic(np.array(ev.drone_unity, dtype=float))
                ev.drone_world = [float(v) for v in w]
            except Exception:
                ev.drone_world = None
    ev.room_name = room.room_name
    ev.room_display = room.display

    ev.saved_image = _save_photo(ev, cfg.events_dir, index)   # 4. 사진 기록
    photo = "사진 저장됨" if ev.saved_image else "사진 없음"
    on_status(f"⚠ {room.display}에서 {ev.label} 탐지 "                # 5. 알림
              f"(신뢰도 {ev.conf:.2f}) — {photo}")

    if cfg.light_on_detect and cfg.light_off_after:
        try:
            bridge.set_light(False)                 # 6. resume
        except Exception:
            pass
    return ev


def _save_photo(ev: DetectionEvent, events_dir: Optional[Path],
                index: int) -> Optional[str]:
    """Copy the detector's photo into the report folder. Never raises —
    a missing/unreadable image must not abort a patrol."""
    if not ev.image_path or events_dir is None:
        return None
    src = Path(ev.image_path)
    if not src.is_file():
        return None
    try:
        events_dir.mkdir(parents=True, exist_ok=True)
        dst = events_dir / f"evt_{index:03d}{src.suffix.lower() or '.jpg'}"
        shutil.copyfile(src, dst)
        return str(dst)
    except OSError:
        return None


# ---------------------------------------------------------------------- mission

def run_patrol(bridge, sim_tf, points_world: np.ndarray, gm: planner.GridMeta,
               rooms: Sequence[RoomInfo], home_world, cfg: PatrolConfig,
               listener: Optional[DetectionListener] = None,
               on_status: Callable[[str], None] = print,
               on_progress: Optional[Callable[[str, dict], None]] = None,
               follow_path_mod=None) -> PatrolResult:
    """Fly the whole patrol. Requires an active `--sim` bridge + transform.

    `on_status` is prose for a human (terminal + Unity banner). `on_progress`
    is the machine-readable twin — `(kind, payload)` per milestone, which the
    API server turns into the poll feed the web console reads. Keeping them
    separate means the web never has to parse Korean sentences to know where
    the drone is.

    Kinds: mission_start, leg_start, arrived, leg_failed, scan_start,
    scan_done, detect, returning, landed, mission_end.
    """
    if follow_path_mod is None:  # injected by the server (repo-root sys.path)
        from simulator.bridge import follow_path as follow_path_mod  # type: ignore

    def progress(kind: str, **payload) -> None:
        if on_progress is not None:
            try:
                on_progress(kind, payload)
            except Exception:      # a listener must never abort a flight
                pass

    res = PatrolResult(started_at=time.time())
    home_world = np.asarray(home_world, dtype=float)
    pos_world = _drone_world(bridge, sim_tf, home_world)

    def fly_leg(goal_world, label: str) -> tuple[bool, str, float]:
        """Plan + fly one leg. -> (reached, reason, length_m)."""
        nonlocal pos_world
        path, info, _ = planner.plan_path(
            points_world, pos_world, goal_world, algo=cfg.algo, gm=gm)
        if path is None:
            on_status(f"{label}: 경로 계산 실패 ({info.get('reason', '?')})")
            return False, "plan_failed", 0.0
        res.planned_path_world.extend([[float(v) for v in p] for p in path])
        wps_unity = sim_tf.mosaic_to_unity(np.asarray(path, dtype=float))
        res.planned_path_unity.extend([[float(v) for v in p] for p in wps_unity])
        on_status(f"이동 중: {label} ({info['length_m']:.1f} m)")
        fr = follow_path_mod.follow_path(
            bridge, [tuple(p) for p in wps_unity],
            max_speed=cfg.speed, rc_limit=cfg.rc_limit,
            timeout_sec=cfg.leg_timeout,
            abort_on_collision=cfg.abort_on_collision,
            on_status=lambda t: on_status(f"  {t}"))
        res.collisions += fr.collision_count
        res.trajectory_unity.extend([list(p) for p in fr.trajectory_unity])
        res.distance_m += float(info["length_m"]) if fr.success else 0.0
        pos_world = _drone_world(bridge, sim_tf, goal_world)
        return fr.success, fr.reason, float(info["length_m"])

    try:
        if bridge.initialize_sdk() == "timeout":
            res.aborted_reason = "simulator_unreachable"
            on_status("시뮬레이터에 연결할 수 없습니다 (Play 모드인가요?)")
            return _finish(res, listener, cfg, progress)

        on_status(f"순찰 시작 — {len(rooms)}개 구역")
        progress("mission_start",
                 rooms=[{"room": r.room_name, "display": r.display,
                         "floor": r.floor} for r in rooms],
                 returnHome=bool(cfg.return_home))
        bridge.takeoff()                       # ← exactly once for the mission
        if bridge.wait_for_state(timeout=5.0) is None:
            res.aborted_reason = "state_lost"
            on_status("시뮬레이터 상태 스트림 없음 (UDP 9002 확인)")
            return _finish(res, listener, cfg, progress)
        pos_world = _drone_world(bridge, sim_tf, pos_world)

        for i, room in enumerate(rooms, 1):
            t_room = time.time()
            goal = scan_pose(room, cfg.hover_height, gm)
            label = f"[{i}/{len(rooms)}] {room.display}"
            progress("leg_start", room=room.room_name, display=room.display,
                     order=i, of=len(rooms))
            reached, reason, leg_m = fly_leg(goal, label)
            if not reached:
                res.rooms.append(RoomVisit(
                    room.room_name, room.display, False, reason, False, 0.0, 0,
                    leg_m, time.time() - t_room))
                on_status(f"{label} 도달 실패 ({reason}) — 다음 구역으로")
                progress("leg_failed", room=room.room_name, reason=reason)
                continue
            progress("arrived", room=room.room_name, display=room.display,
                     legMeters=round(leg_m, 2))

            # Detection is only accepted INSIDE the patrol room.
            if listener is not None:
                listener.arm(room.room_name)
            room_events: List[DetectionEvent] = []

            def _on_event(ev: DetectionEvent, _room=room) -> None:
                annotated = react_to_detection(
                    bridge, ev, _room, cfg, sim_tf, on_status,
                    len(res.events) + 1)
                res.events.append(annotated)
                room_events.append(annotated)
                progress("detect", room=_room.room_name,
                         display=_room.display, n=len(res.events),
                         label=annotated.label, conf=annotated.conf,
                         box=annotated.box_pct,
                         image=annotated.saved_image or annotated.image_path)

            on_status(f"{label} 도착 — 360° 스캔 중...")
            progress("scan_start", room=room.room_name, display=room.display)
            done, degrees, n_ev, by_unity = scan_room(
                bridge, listener, cfg, on_status, on_event=_on_event)
            if cfg.scan_mode == "auto" and not by_unity:
                # One probe per mission, not per room: a build without the verb
                # would otherwise cost scan_probe_sec in every single room.
                cfg.scan_mode = "rc"
            if listener is not None:
                listener.disarm()
            res.rooms.append(RoomVisit(
                room.room_name, room.display, True, reason, done, degrees,
                len(room_events), leg_m, time.time() - t_room))
            on_status(f"{label} 스캔 완료 ({degrees:.0f}°, 탐지 {len(room_events)}건)")
            progress("scan_done", room=room.room_name, display=room.display,
                     degrees=round(degrees, 1), detections=len(room_events),
                     completed=bool(done))

        if cfg.return_home:
            progress("returning")
            reached, _, _ = fly_leg(home_world, "복귀 지점(home)")
            res.returned_home = reached
        progress("landed")
        return _finish(res, listener, cfg, progress)
    except Exception as e:                       # never leave the drone driving
        res.aborted_reason = f"{type(e).__name__}: {e}"
        on_status(f"순찰 중단: {e}")
        return _finish(res, listener, cfg, progress)
    finally:
        try:
            bridge.send_rc(0, 0, 0, 0)
            if cfg.light_on_detect:
                bridge.set_light(False)
            time.sleep(0.3)
            bridge.land()
        except Exception:
            pass


def _finish(res: PatrolResult, listener: Optional[DetectionListener],
            cfg: PatrolConfig, progress=None) -> PatrolResult:
    res.finished_at = time.time()
    if listener is not None:
        res.listener_stats = listener.stats()
        listener.disarm()
    if cfg.viz_dir is not None:
        write_viz(res, cfg.viz_dir)
    if progress is not None:
        progress("mission_end",
                 flownMeters=round(res.distance_m, 1),
                 durationSec=round(res.duration_s, 1),
                 roomsReached=res.rooms_reached,
                 roomsPlanned=len(res.rooms),
                 detections=len(res.events),
                 collisions=res.collisions,
                 returnedHome=bool(res.returned_home),
                 abortedReason=res.aborted_reason)
    return res


def _drone_world(bridge, sim_tf, fallback) -> np.ndarray:
    """Live simulator position in world meters, else `fallback`."""
    if bridge is not None and sim_tf is not None:
        s = bridge.get_latest_state()
        if s is not None:
            return sim_tf.unity_to_mosaic(np.array([s.x, s.y, s.z], dtype=float))
    return np.asarray(fallback, dtype=float)


# ------------------------------------------------------------ Unity path viz

def write_viz(res: PatrolResult, viz_dir: Path) -> None:
    """Feed the two Unity renderers that already exist but have had no producer
    since the jiyun-simul branch: `PlannedPathRenderer.cs` reads
    planned_path_3d.json, `FlightReportRenderer.cs` reads
    flight_trajectory_3d.json (+ red spheres at `intrusions_world`).

    Point --viz-dir at simulator/tello_simulator/Assets/Resources to see the
    patrol route and the detection points drawn in the scene — no C# changes.

    Both renderers use `useWorldSpace = true` / `transform.position`, so every
    coordinate here must be UNITY units, not mosaic meters. They also parse with
    JsonUtility into `SerializableVector3`, so points must be {x,y,z} OBJECTS —
    a bare [x,y,z] array is silently dropped.
    """
    def vecs(points) -> List[dict]:
        return [{"x": float(p[0]), "y": float(p[1]), "z": float(p[2])}
                for p in points if p is not None and len(p) >= 3]

    try:
        viz_dir = Path(viz_dir)
        viz_dir.mkdir(parents=True, exist_ok=True)
        path = vecs(res.planned_path_unity)
        empty = {"x": 0.0, "y": 0.0, "z": 0.0}
        with open(viz_dir / "planned_path_3d.json", "w", encoding="utf-8") as f:
            json.dump({"planner": "patrol-astar", "path_world": path,
                       "start_world": path[0] if path else empty,
                       "goal_world": path[-1] if path else empty}, f)
        intrusions = vecs([e.drone_unity for e in res.events if e.drone_unity])
        with open(viz_dir / "flight_trajectory_3d.json", "w", encoding="utf-8") as f:
            json.dump({"planner": "patrol-astar",
                       "intrusion_steps": len(intrusions),
                       "trajectory_world": vecs(res.trajectory_unity),
                       "path_world": path,
                       "intrusions_world": intrusions}, f)
    except (OSError, TypeError, ValueError):
        pass  # visualization is never worth failing a mission over
