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
import math
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
    flight: str = "pid"              # 추종기: "pid" | "rl" (경로는 둘 다 A*)
    data_dir: Optional[Path] = None  # flight="rl" 일 때 장애물 점군 출처
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
    # arrived | timeout | collision | ... — `plan_failed` 은 더 이상 안 나온다.
    # 계획은 `plan_leg` 가 3단으로 폴백해서 항상 경로를 돌려주므로, 실패는
    # 이제 계획이 아니라 추종 단계에서만 생긴다.
    reason: str
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
              should_abort: Optional[Callable[[], bool]] = None,
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
            bridge, cfg, on_status, on_event, should_abort)
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
                      should_abort: Optional[Callable[[], bool]] = None,
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
        # 스캔 한 바퀴는 50°/s 에서 7초다. 중단을 여기서 안 보면 그 7초가 통째로
        # 재시작 지연이 되므로, `answered=True` 로 돌려줘 rc 폴백은 타지 않게 한다.
        if should_abort is not None and should_abort():
            bridge.stop_scan()
            return False, 0.0, events, True
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


# ------------------------------------------------------------------- planning

def clearance_ladder(base_res: float) -> list[tuple[float, int]]:
    """`(resolution, margin)` 단계들. 여유가 넓은 쪽 → 좁은 쪽 순서.

    장애물 팽창 여유(clearance) = `resolution × margin` [m] 이다. 기본 0.15 m
    격자 기준으로:

        (0.15, 2) -> 0.30 m 여유   폭 0.75 m 이상인 통로만 통과
        (0.15, 1) -> 0.15 m        폭 0.45 m 이상   ← 두 진입점이 만드는 격자
        (0.075, 1) -> 0.075 m      폭 0.225 m 이상
        (0.05, 1) -> 0.05 m        폭 0.15 m 이상

    **해상도를 낮출지언정 `margin` 은 0으로 내리지 않는다.** 팽창을 빼면 좁은
    통로가 열려서 계획은 성공하지만, 경로가 벽 표면에 딱 붙어버려 드론이 그대로
    긁고 지나간다 — 없애려는 그 충돌이 계획 단계에서 만들어지는 셈이다. 대신
    해상도를 반/삼분의 일로 줄이면 필요한 통로 폭도 같은 비율로 줄면서 여유는
    (작아질지언정) 남는다.
    """
    return [(base_res, 2), (base_res, 1), (base_res / 2.0, 1),
            (base_res / 3.0, 1)]


def plan_leg(points_world: np.ndarray, gm: planner.GridMeta,
             start_world, goal_world, algo: str = "astar",
             grid_cache: Optional[dict] = None) -> tuple[list, dict]:
    """Plan one leg, from the safest grid down. Never returns None.

    건물 전체가 하나의 복셀 격자이고 팽창이 좁은 방·문틈을 막아버려 A* 가
    `no path` / `max_iters exceeded` 로 죽는 구간이 있다 (patrol/README.md
    §한계, 안방 `000_003`). 미션 전체 설정을 흔드는 대신 **막힌 그 구간만**
    `clearance_ladder()` 를 한 칸씩 내려가며 다시 푼다. 넉넉한 여유로 먼저
    풀어보고, 안 되는 구간에서만 조인다 — 벽에서 멀리 도는 경로가 기본이 된다.

    사다리를 다 내려가도 못 풀면 **목표를 포기하지 목표까지 뚫고 가지는 않는다**:
    호출자의 격자(여유 그대로)에서 `best_effort` A* 를 돌려 도달 가능한 지점 중
    목표에 가장 가까운 곳까지 간다(`fallback="nearest"`). 00809 의 `002_021`·
    `002_022` 처럼 드론이 들어갈 수 없는 좁은 공간이 목표일 때 이게 걸린다 —
    벽을 통과하는 대신 문 앞까지 가서 거기서 스캔한다.

    직선(`"direct"`)은 출발점조차 자유공간이 아닐 때만 나오는 마지막 안전망이다.

    info 에 남는 것: `clearance_m`(이 경로가 확보한 여유), `fallback`
    (`None` = 가장 안전한 단계에서 풀림 | `"tightened"` | `"nearest"` |
    `"direct"`), `gap_to_goal_m`(`"nearest"` 일 때 목표까지 남은 거리),
    `first_reason`(첫 단계가 실패한 이유), `rung`(사다리에서 몇 번째).

    `grid_cache` 는 단계별 격자를 미션 내내 재사용하는 통. 0.05 m 격자는 만드는
    데 몇 초 걸리므로 구간마다 다시 만들면 안 된다.
    """
    if grid_cache is None:
        grid_cache = {}
    rungs = clearance_ladder(gm.resolution)
    # 호출자가 이미 만들어 넘긴 격자를 그 자리에 꽂아 둔다 (다시 만들지 않도록).
    grid_cache.setdefault((round(gm.resolution, 4), 1), gm)

    first_reason = None
    reason = "?"
    for rung, (res, margin) in enumerate(rungs):
        key = (round(res, 4), margin)
        grid = grid_cache.get(key)
        if grid is None:
            # 점은 이미 point_stride 로 솎여 들어오므로 sample=1
            # (두 진입점의 --sample 기본값과 같다).
            grid = planner.voxelize(points_world, res, margin, 1)
            grid_cache[key] = grid
        # 이 단계에서 목표가 아예 도달 불가면 A* 를 돌리지 않는다. 실패하는
        # A* 가 제일 비싸다 — 도달 가능한 복셀을 전부 펼쳐봐야 "없다"고 말할 수
        # 있기 때문이다(실측 5~9 s). 연결 성분 라벨은 그걸 한 번에 답한다.
        if not planner.connected(grid, start_world, goal_world):
            if first_reason is None:
                first_reason = "unreachable at this clearance"
            reason = "unreachable at this clearance"
            continue
        # 첫 단계만 호출자가 고른 algo 를 쓴다. 재시도는 A* 로 못 박는다 —
        # RRT* 는 좁은 통로를 확률적으로 놓치므로 폴백에 어울리지 않는다.
        path, info, _ = planner.plan_path(
            points_world, start_world, goal_world,
            algo=(algo if rung == 0 else "astar"), gm=grid)
        if path is not None:
            info["clearance_m"] = round(res * margin, 4)
            info["rung"] = rung
            info["fallback"] = None if rung == 0 else "tightened"
            info["first_reason"] = first_reason
            return path, info
        reason = info.get("reason", "?")
        if first_reason is None:
            first_reason = reason

    # 어느 여유로도 목표에 못 닿는다 = 드론이 들어갈 수 없는 공간이다. 뚫고
    # 가는 대신 갈 수 있는 데까지만 간다. 여유는 호출자 격자 그대로 지킨다.
    path, info, _ = planner.plan_path(
        points_world, start_world, goal_world, algo="astar", gm=gm,
        best_effort=True)
    if path is not None and len(path) >= 2:
        info["clearance_m"] = round(gm.resolution, 4)
        info["rung"] = len(rungs)
        info["fallback"] = "nearest"
        info["first_reason"] = first_reason
        return path, info

    # 출발점 자체가 막혀 있는 경우만 여기까지 온다.
    path = [np.asarray(start_world, dtype=float),
            np.asarray(goal_world, dtype=float)]
    return path, {"algo": "direct", "fallback": "direct",
                  "clearance_m": 0.0, "rung": len(rungs) + 1,
                  "first_reason": first_reason,
                  "reason": reason,
                  "length_m": planner.path_length(path),
                  "n_waypoints": 2}


# ---------------------------------------------------------------------- mission

def run_patrol(bridge, sim_tf, points_world: np.ndarray, gm: planner.GridMeta,
               rooms: Sequence[RoomInfo], home_world, cfg: PatrolConfig,
               listener: Optional[DetectionListener] = None,
               on_status: Callable[[str], None] = print,
               on_progress: Optional[Callable[[str, dict], None]] = None,
               follow_path_mod=None,
               should_abort: Optional[Callable[[], bool]] = None) -> PatrolResult:
    """Fly the whole patrol. Requires an active `--sim` bridge + transform.

    `on_status` is prose for a human (terminal + Unity banner). `on_progress`
    is the machine-readable twin — `(kind, payload)` per milestone, which the
    API server turns into the poll feed the web console reads. Keeping them
    separate means the web never has to parse Korean sentences to know where
    the drone is.

    Kinds: mission_start, leg_start, arrived, leg_failed, scan_start,
    scan_done, detect, returning, landed, mission_end.

    `should_abort` 는 사용자가 순찰을 중단했는지 묻는다. 구간 경계마다, 그리고
    비행·스캔 루프 **안에서** 확인한다 — 경계에서만 보면 한 구간이 타임아웃
    (최대 600초)까지 도는 동안 아무도 못 멈춘다. 중단이 확인되면 `aborted_reason`
    을 "aborted" 로 두고 정상 종료 경로(`_finish`)로 빠진다: 보고서는 그대로
    나오고, 스레드가 즉시 끝나므로 다음 순찰을 바로 시작할 수 있다.
    """
    def aborted() -> bool:
        return should_abort is not None and should_abort()
    if follow_path_mod is None:  # injected by the server (repo-root sys.path)
        from simulator.bridge import follow_path as follow_path_mod  # type: ignore

    # 강화학습 추종기는 쓸 때만 올린다 (장애물 KDTree 구성이 몇 초 걸린다).
    rl = None
    if cfg.flight == "rl":
        from simulator.bridge import follow_rl as follow_rl_mod  # type: ignore
        world = follow_rl_mod.GeoWorld.load(cfg.data_dir)
        rl = (follow_rl_mod, world, follow_rl_mod.NumpyActor())
        on_status(f"추종기: 강화학습 정책 (장애물 점 {len(world.coord):,}개)")

    def progress(kind: str, **payload) -> None:
        if on_progress is not None:
            try:
                on_progress(kind, payload)
            except Exception:      # a listener must never abort a flight
                pass

    res = PatrolResult(started_at=time.time())
    home_world = np.asarray(home_world, dtype=float)
    pos_world = _drone_world(bridge, sim_tf, home_world)

    grid_cache: dict = {}       # plan_leg 의 단계별 격자, 미션 내내 재사용

    def fly_leg(goal_world, label: str) -> tuple[bool, str, float]:
        """Plan + fly one leg. -> (reached, reason, length_m)."""
        nonlocal pos_world
        path, info = plan_leg(points_world, gm, pos_world, goal_world,
                              algo=cfg.algo, grid_cache=grid_cache)
        fallback = info.get("fallback")
        if fallback == "tightened":
            r0, m0 = clearance_ladder(gm.resolution)[0]
            on_status(f"{label}: 여유 {r0 * m0:.2f} m 로는 막힘"
                      f"({info['first_reason']}) → 여유 "
                      f"{info['clearance_m']:.2f} m 로 좁혀 그 구간만 재계획")
        elif fallback == "nearest":
            on_status(f"{label}: 드론이 들어갈 수 없는 공간 "
                      f"({info['first_reason']}) → 목표 "
                      f"{info.get('gap_to_goal_m', 0):.1f} m 앞까지만 접근")
        elif fallback == "direct":
            on_status(f"{label}: 출발점이 막혀 있음({info['first_reason']}) "
                      f"→ 직선으로 이동 (벽 통과 가능)")
        res.planned_path_world.extend([[float(v) for v in p] for p in path])
        wps_unity = sim_tf.mosaic_to_unity(np.asarray(path, dtype=float))
        res.planned_path_unity.extend([[float(v) for v in p] for p in wps_unity])
        on_status(f"이동 중: {label} ({info['length_m']:.1f} m)")
        fr = None
        if rl is not None:
            rl_mod, rl_world, rl_actor = rl
            fr = rl_mod.follow_rl(
                bridge, np.asarray(path, dtype=float), sim_tf, rl_world, rl_actor,
                max_speed=cfg.speed, rc_limit=cfg.rc_limit,
                timeout_sec=cfg.leg_timeout,
                abort_on_collision=cfg.abort_on_collision,
                on_status=lambda t: on_status(f"  {t}"),
                should_abort=should_abort)
            if not fr.success and fr.reason != "aborted":
                # 정책이 막히면 미션을 버리지 않고 PID 로 마저 간다. 남은 구간은
                # 현재 위치에서 다시 계획된 게 아니라 원래 경로라, 이미 지나온
                # 앞부분은 PID 가 알아서 건너뛴다(도착 판정 반경).
                on_status(f"  RL 실패({fr.reason}) → PID 로 폴백")
        # 중단이면 폴백하지 않는다 — 멈추라고 했는데 다른 추종기로 다시 날면 안 된다.
        if (fr is None or not fr.success) and not (fr is not None
                                                   and fr.reason == "aborted"):
            fr = follow_path_mod.follow_path(
                bridge, [tuple(p) for p in wps_unity],
                max_speed=cfg.speed, rc_limit=cfg.rc_limit,
                timeout_sec=cfg.leg_timeout,
                abort_on_collision=cfg.abort_on_collision,
                on_status=lambda t: on_status(f"  {t}"),
                should_abort=should_abort)
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
            if aborted():
                res.aborted_reason = "aborted"
                on_status("순찰 중단됨 — 남은 구역은 건너뜁니다")
                break
            t_room = time.time()
            goal = scan_pose(room, cfg.hover_height, gm)
            label = f"[{i}/{len(rooms)}] {room.display}"
            progress("leg_start", room=room.room_name, display=room.display,
                     order=i, of=len(rooms))
            reached, reason, leg_m = fly_leg(goal, label)
            if reason == "aborted":
                res.aborted_reason = "aborted"
                res.rooms.append(RoomVisit(
                    room.room_name, room.display, False, reason, False, 0.0, 0,
                    leg_m, time.time() - t_room))
                on_status("순찰 중단됨 — 이동 중 정지")
                progress("leg_failed", room=room.room_name, reason=reason)
                break
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
                bridge, listener, cfg, on_status, on_event=_on_event,
                should_abort=should_abort)
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

        # 중단이면 복귀도 하지 않는다. 멈추라고 한 직후에 집까지 한 구간을 더
        # 나는 건 사용자가 기대하는 동작이 아니고, 그 구간이 끝날 때까지 스레드가
        # 살아 있어 재시작도 그만큼 늦어진다.
        if cfg.return_home and not aborted():
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


# ------------------------------------------------------------------ self-check

def _demo() -> None:
    """`python -m patrol.patrol_mission` — plan_leg 의 3단 폴백을 합성 씬으로 확인.

    시뮬레이터도 Unity도 필요 없다. 방 하나를 벽으로 갈라놓고 문틈 너비만 바꿔
    1번(그냥 통과) / 2번(팽창이 문틈을 막음) / 3번(문이 아예 없음)을 각각 태운다.
    """
    # 로그가 전부 한국어인데 윈도우 기본이 cp949 라 '—' 하나에 죽는다.
    import sys
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass

    res_m = 0.15

    # 문을 한쪽으로 치우쳐 둔다 — start/goal 직선 위에 두면 A* 가 푼 경로와 3번의
    # 직선 폴백이 길이까지 똑같아져서, 검사가 둘을 구분하지 못한다.
    gap_center = 0.8

    def room_points(gap_half: float) -> np.ndarray:
        """4×3×1.5 m 방 + x=0 칸막이. `gap_half`=0 이면 문틈 없음."""
        step = 0.05
        pts = []
        ax = np.arange(-2.0, 2.0 + step, step)
        ay = np.arange(-1.5, 1.5 + step, step)
        az = np.arange(0.0, 1.5 + step, step)
        # 바깥 벽 4면 (격자 범위를 정하고 밖으로 새는 우회로를 막는다)
        for y in (ay[0], ay[-1]):
            gx, gz = np.meshgrid(ax, az, indexing="ij")
            pts.append(np.column_stack([gx.ravel(), np.full(gx.size, y), gz.ravel()]))
        for x in (ax[0], ax[-1]):
            gy, gz = np.meshgrid(ay, az, indexing="ij")
            pts.append(np.column_stack([np.full(gy.size, x), gy.ravel(), gz.ravel()]))
        # 바닥·천장
        for z in (az[0], az[-1]):
            gx, gy = np.meshgrid(ax, ay, indexing="ij")
            pts.append(np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, z)]))
        # 칸막이: x=0, 문틈만 비움
        gy, gz = np.meshgrid(ay, az, indexing="ij")
        gy, gz = gy.ravel(), gz.ravel()
        keep = (np.abs(gy - gap_center) >= gap_half if gap_half > 0
                else np.ones(gy.shape, bool))
        pts.append(np.column_stack([np.zeros(keep.sum()), gy[keep], gz[keep]]))
        return np.vstack(pts)

    start, goal = (-1.2, 0.0, 0.75), (1.2, 0.0, 0.75)

    def min_clearance(path, pts) -> float:
        """경로 위 어느 점이든 가장 가까운 장애물 점까지의 거리 [m].
        계획이 약속한 여유를 실제로 지켰는지 보는 유일한 잣대."""
        dense = []
        for a, b in zip(path[:-1], path[1:]):
            a, b = np.asarray(a, float), np.asarray(b, float)
            n = max(2, int(np.linalg.norm(b - a) / 0.02))
            dense.append(a + (b - a) * np.linspace(0, 1, n)[:, None])
        dense = np.vstack(dense)
        # 점군이 작아 브루트포스로 충분하다 (scipy 없이 돌게).
        best = math.inf
        for chunk in np.array_split(dense, max(1, len(dense) // 200)):
            d = np.linalg.norm(pts[None, :, :] - chunk[:, None, :], axis=2)
            best = min(best, float(d.min()))
        return best

    # 1) 문이 넓으면(±0.6 m) 가장 안전한 단계(여유 0.30 m)에서 바로 풀린다.
    pts = room_points(0.60)
    gm_wide = planner.voxelize(pts, res_m, 1, 1)
    path, info = plan_leg(pts, gm_wide, start, goal)
    assert info["fallback"] is None and info["rung"] == 0, info
    assert info["clearance_m"] == 0.30, info
    # 문이 y=+0.8 로 치우쳐 있으니 직선 2.4 m 보다 길어야 = 진짜로 돌아갔어야 한다.
    assert 2.5 < info["length_m"] < math.inf, info
    c1 = min_clearance(path, pts)
    print(f"  1) 넓은 문(±0.60)  rung={info['rung']} 여유 "
          f"{info['clearance_m']:.3f} m, {info['length_m']:.2f} m, "
          f"실측 최소거리 {c1:.3f} m")

    # 2) 문틈 ±0.15 m — 여유 0.30 도 0.15 도 막힌다. 사다리를 내려가 풀되,
    #    여유가 0 이 되지는 않아야 한다. 이게 이번 변경의 핵심 주장이다.
    pts = room_points(0.15)
    gm_mid = planner.voxelize(pts, res_m, 1, 1)
    assert planner.astar(gm_mid, np.array(start), np.array(goal))[0] is None, \
        "합성 씬이 의도와 다르다 — 팽창 1셀로도 문틈이 안 막혔다"
    # 같은 격자·같은 질의라도 best_effort 를 켜면 None 대신 '갈 수 있는 데까지'가
    # 나온다. 기본값(False)이 여전히 None 인 건 바로 위 assert 가 지킨다.
    pp, pi = planner.astar(gm_mid, np.array(start), np.array(goal), best_effort=True)
    assert pp is not None and pi["partial"], pi
    assert max(float(q[0]) for q in pp) < 0.0, "best_effort 가 벽을 넘었다"
    cache: dict = {}
    path, info = plan_leg(pts, gm_mid, start, goal, grid_cache=cache)
    assert info["fallback"] == "tightened", info
    assert info["rung"] >= 2, info
    assert info["clearance_m"] > 0.0, "여유 0 으로 푸는 단계가 있으면 안 된다"
    assert 2.5 < info["length_m"] < math.inf, info
    c2 = min_clearance(path, pts)
    assert c2 > 0.03, f"경로가 벽에 붙었다 (최소거리 {c2:.3f} m)"
    print(f"  2) 좁은 문틈(±0.15) rung={info['rung']} 여유 "
          f"{info['clearance_m']:.3f} m ({info['first_reason']}), "
          f"{info['length_m']:.2f} m, 실측 최소거리 {c2:.3f} m")

    # 두 번째 호출은 캐시를 재사용한다 (격자를 다시 만들지 않는다).
    keys, objs = set(cache), dict(cache)
    plan_leg(pts, gm_mid, start, goal, grid_cache=cache)
    assert set(cache) == keys and all(cache[k] is objs[k] for k in keys), \
        "grid_cache 가 재사용되지 않았다"

    # 3) 문이 아예 없으면 어느 단계로도 못 푼다. 뚫고 가면 안 되고, 벽 앞까지만
    #    가야 한다 — 이번 변경에서 제일 중요한 성질이다.
    pts = room_points(0.0)
    gm_sealed = planner.voxelize(pts, res_m, 1, 1)
    path, info = plan_leg(pts, gm_sealed, start, goal)
    assert info["fallback"] == "nearest", info
    assert info["clearance_m"] > 0.0, info
    c3 = min_clearance(path, pts)
    assert c3 > 0.03, f"벽 앞까지만 가야 하는데 붙었다 (최소거리 {c3:.3f} m)"
    # 경로가 칸막이(x=0)를 넘지 않아야 한다 = 뚫고 가지 않았다.
    assert max(float(p[0]) for p in path) < 0.0, \
        f"막힌 벽을 통과했다 (최대 x={max(float(p[0]) for p in path):.2f})"
    print(f"  3) 막힌 벽         rung={info['rung']} fallback=nearest, "
          f"목표 {info['gap_to_goal_m']:.2f} m 앞까지, "
          f"실측 최소거리 {c3:.3f} m, 벽 안 넘음")

    print("plan_leg: 사다리 전 단계 통과 — 경로 None 없음, "
          "여유 0 인 A* 경로 없음, 막힌 목표는 뚫지 않고 앞에서 멈춤")


if __name__ == "__main__":
    _demo()
