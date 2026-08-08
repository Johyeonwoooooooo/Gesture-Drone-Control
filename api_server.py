"""api_server.py — the web console's backend. Runs next to Unity, not on the GPU box.

The HAUNTED OPS console (`web/`) picks patrol areas on a floor plan and draws
what happens; everything behind it lives here. One process, because there is
one drone and one UDP link to it — two backends cannot both hold 9000/9002, and
the console calls `api/drone` and `plan` as RELATIVE paths, so the static files
and the API have to share an origin anyway.

    python api_server.py                                      # 로컬 Ollama (기본)
    python api_server.py --llm-url gpu                        # 학교 GPU 서버 (VPN)
    python api_server.py --llm-url ""                         # LLM 없이 테스트

LLM은 기본적으로 **이 PC의 Ollama** (`qwen2.5:3b-instruct`) 를 쓴다. 순찰 한 번에
LLM 호출이 2회뿐이라 3B면 노트북에서 충분하다. 학교 GPU 서버를 쓰려면
`--llm-url gpu` — 주소·모델 이름은 `remote_llm.GPU_LLM_URL/GPU_LLM_MODEL` 에
있고, 토큰은 `PATROL_LLM_API_KEY` 환경변수로 넘기면 명령줄에 안 남는다.
**교내망 밖에서는 8000·22 가 막혀 있어 한양대 SSL VPN 을 켜야 닿는다.**
프로토콜이 OpenAI 호환이라 어느 쪽이든 클라이언트는 같다.

`--llm-url ""` 로 띄우면 오프라인 모드다 — LLM 호출만 빠지고 나머지(방 인덱스,
플래너, 미션 루프, 보고서, 웹 콘솔)는 그대로 돈다. `/api/intent` 는 별칭·층·방
종류 키워드로 구역을 찾고(모델이 제안하는 방 코드 한 단계만 못 쓴다), 보고서
요약문은 템플릿 문장으로 나온다. Unity 도 마찬가지로 없으면 없는 대로 뜬다 —
드론 위치가 `source:"home"` 으로 고정될 뿐이다.

Routes the console already calls (do not change their shape):

    GET  /api/drone     x,y,z,source,flying,connected      polled every 1 s
    GET  /api/status    ready/engine/rooms/mission
    POST /plan          {start,goal} -> one leg, for the briefing drawing
    GET  /  , /<path>   web/ static

Routes it is meant to grow into — the seams the console already left open:

    POST /api/intent            "2층 전부 순찰해줘" -> room ids   (runSearch)
    POST /api/patrol/start      the launch payload -> fly it     (startMission)
    GET  /api/patrol/events     progress feed, `since` cursor    (진행 로그 입구)
    POST /api/patrol/abort      stop and land
    GET  /api/patrol/report/... the recorded mission, for 기록

The feed is polled, not pushed: the console has no WebSocket or EventSource
anywhere, only `fetch`. A `since` cursor over a monotonic sequence means a slow
or backgrounded tab misses nothing.

Room ids differ only in spelling — we say "002_012", the web says "012" — so
the conversion happens at this boundary and nowhere else (room_index.web_room_id).
Coordinates need no conversion at all: both sides are Z-up world meters and the
room boxes agree to the centimetre.
"""
from __future__ import annotations

import argparse
import collections
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent))

# 콘솔 로그가 한국어라 윈도우 기본 cp949 로는 '—' 같은 글자에서 죽는다.
for _s in (sys.stdout, sys.stderr):
    _s.reconfigure(encoding="utf-8", errors="replace")

from patrol import (patrol_intent, patrol_mission, patrol_report,  # noqa: E402
                    planner, room_index)
from patrol.litept_backend import LitePTBackend  # noqa: E402
from patrol.remote_llm import (DEFAULT_LLM_MODEL, DEFAULT_LLM_URL,  # noqa: E402
                               RemoteLLMError, make_llm, resolve_llm)
from patrol.room_index import RoomInfo  # noqa: E402

WEB_DIR = _THIS.parent / "web"


# --------------------------------------------------------------- progress feed

class EventLog:
    """Monotonic, bounded, thread-safe. The console polls `since`."""

    def __init__(self, maxlen: int = 500) -> None:
        self._lock = threading.Lock()
        self._events: collections.deque = collections.deque(maxlen=maxlen)
        self._seq = 0

    def append(self, kind: str, payload: Optional[dict] = None) -> int:
        with self._lock:
            self._seq += 1
            self._events.append({"seq": self._seq, "kind": kind,
                                 "t": time.time(), **(payload or {})})
            return self._seq

    def since(self, seq: int) -> List[dict]:
        with self._lock:
            return [e for e in self._events if e["seq"] > seq]

    @property
    def seq(self) -> int:
        with self._lock:
            return self._seq

    def clear(self) -> None:
        with self._lock:
            self._events.clear()      # `seq` deliberately keeps counting:
                                      # a client mid-poll must not see it reset


# ----------------------------------------------------------------------- scene

class Scene:
    """Everything loaded once at startup, plus the live simulator link."""

    def __init__(self, args) -> None:
        self.args = args
        data_dir = Path(args.data_dir)
        print(f"[api] loading detections from {data_dir} ...")
        self.backend = LitePTBackend(data_dir)
        self.rooms: Dict[str, RoomInfo] = room_index.build_room_index(
            self.backend, args.room_aliases)
        print(f"[api] {len(self.backend.detections)} detections, "
              f"{len(self.rooms)} rooms")

        t0 = time.time()
        self.points_world = self.backend.load_points(stride=args.point_stride)
        self.gm = planner.voxelize(self.points_world, args.resolution,
                                   args.margin, args.sample)
        # patrol_mission.plan_leg 가 좁은 통로용으로 만드는 팽창 0 격자.
        # 만드는 데 몇 초 걸리므로 프로세스 내내 재사용한다.
        self.grid_cache: Dict[tuple, planner.GridMeta] = {}
        self.home = self.backend.default_home()
        print(f"[api] scene ready: grid={self.gm.shape}, "
              f"home={np.round(self.home, 2)} ({time.time() - t0:.1f}s)")

        llm_url, llm_model, llm_key = resolve_llm(
            args.llm_url, args.llm_model, args.llm_api_key)
        self.llm = make_llm(llm_url, model_id=llm_model,
                            api_key=llm_key, timeout=args.llm_timeout)
        self.offline_llm = not llm_url
        if not self.offline_llm:
            try:
                served = self.llm.ping()   # fail here, not on the first query
            except RemoteLLMError as e:
                raise SystemExit(
                    f'[llm] {e}\n[llm] LLM 없이 나머지를 테스트하려면 '
                    f'--llm-url "" 로 실행하세요.') from e
            print(f"[llm] server ok, models={served or '(no /models route)'}")

        from simulator.bridge import coord_transform, follow_path
        from simulator.bridge.unity_bridge import UnityTelloBridge
        self.follow_path = follow_path
        self.sim_tf = coord_transform.load_building_transform(args.building)
        self.bridge = UnityTelloBridge(args.unity_host, args.unity_port,
                                       args.unity_local_port,
                                       args.unity_state_port)
        self.bridge.connect()
        reply = self.bridge.initialize_sdk()
        print(f"[sim] Unity {args.unity_host}:{args.unity_port} -> {reply!r}"
              + ("  (Play 모드인가요?)" if reply == "timeout" else ""))

    # ------------------------------------------------------------- geometry --
    def drone_pose(self) -> tuple[np.ndarray, str, bool]:
        """(world meters, 'sim'|'home', flying)."""
        s = self.bridge.get_latest_state()
        if s is not None:
            p = self.sim_tf.unity_to_mosaic(
                np.array([s.x, s.y, s.z], dtype=float))
            return p, "sim", bool(s.flying)
        return self.home.copy(), "home", False

    def room_at(self, world) -> Optional[RoomInfo]:
        """Smallest room containing the point — **z matters**.

        Floors sit on top of each other, so xy alone matches a stack of rooms
        and the smallest of those is usually on the wrong floor. Prefer the
        ones whose z range contains the point; if the console sent a z that
        misses every box, fall back to the xy hit with the nearest floor.
        """
        x, y, z = (float(world[0]), float(world[1]), float(world[2]))
        hits = [r for r in self.rooms.values()
                if r.bbox_min[0] <= x <= r.bbox_max[0]
                and r.bbox_min[1] <= y <= r.bbox_max[1]]
        if not hits:
            return None
        # The floor you would be standing on: the highest one still at or below
        # the point. Tall rooms (the 1층 거실 reaches into the 2층 ceiling) span
        # the floor above, so "z is inside the box" alone is not enough.
        under = [r for r in hits if float(r.floor_z) <= z + 0.5]
        if under:
            top = max(float(r.floor_z) for r in under)
            same = [r for r in under if float(r.floor_z) >= top - 0.5]
            return min(same, key=lambda r: float(np.prod(r.size_xy)))
        return min(hits, key=lambda r: abs(float(r.floor_z) - z))

    def snap_goal(self, goal_world) -> np.ndarray:
        """Turn a console goal into a pose the drone can actually hold.

        The console sends the room's bbox mid-height as z, which is roughly a
        metre above where we hover and can land inside furniture. When the
        point falls in a known room, our own scan pose wins.
        """
        goal = np.asarray(goal_world, dtype=float)
        room = self.room_at(goal)
        if room is None:
            return goal
        return room_index.scan_pose(room, self.args.hover_height, self.gm)


# --------------------------------------------------------------------- mission

class MissionRunner:
    """One patrol at a time — there is one drone."""

    def __init__(self, scene: Scene, log: EventLog) -> None:
        self.scene = scene
        self.log = log
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self.state = "idle"          # idle | running | done | error
        self.mission_id = ""
        self.abort_requested = False
        self.last_result = None
        self.last_report_dir: Optional[Path] = None

    @property
    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # 중단된 순찰이 스스로 정리하고 끝나기를 기다려 주는 시간. 비행 루프는 한
    # 틱(0.05초) 안에 빠져나오지만 그 뒤 보고서를 쓰는데 요약문이 LLM 호출이라
    # 몇 초 걸린다. 그동안 스레드가 살아 있어 `busy` 가 True 이므로, 이걸 안
    # 기다리면 사용자가 중단 직후 누른 재시작이 409 로 튕긴다.
    ABORT_WIND_DOWN_SEC = 8.0

    def start(self, rooms: List[RoomInfo], return_home: bool,
              cmd: str = "") -> str:
        with self._lock:
            if self.busy and self.abort_requested and self._thread is not None:
                self._thread.join(timeout=self.ABORT_WIND_DOWN_SEC)
            if self.busy:
                raise HTTPException(409, "순찰이 이미 진행 중입니다")
            self.mission_id = time.strftime("%Y%m%d_%H%M%S")
            self.state = "running"
            self.abort_requested = False
            self.log.clear()
            self._thread = threading.Thread(
                target=self._run, args=(rooms, return_home, cmd), daemon=True)
            self._thread.start()
            return self.mission_id

    def abort(self) -> None:
        self.abort_requested = True
        self.log.append("abort_requested")
        try:
            self.scene.bridge.stop_scan()
            self.scene.bridge.send_rc(0, 0, 0, 0)
            self.scene.bridge.land()
        except Exception:
            pass

    def _run(self, rooms: List[RoomInfo], return_home: bool,
             cmd: str = "") -> None:
        sc, a = self.scene, self.scene.args
        out_dir = patrol_report.report_dir_for(a.report_dir)
        cfg = patrol_mission.PatrolConfig(
            hover_height=a.hover_height,
            scan_deg_per_sec=a.scan_deg_per_sec,
            scan_turns=a.scan_turns,
            scan_mode=a.scan_mode,
            labels=tuple(l.lower() for l in (a.patrol_labels or ())),
            min_conf=a.patrol_min_conf,
            light_on_detect=not a.no_light,
            return_home=return_home,
            speed=float(a.sim_speed), rc_limit=int(a.sim_rc_limit),
            algo=a.algo, leg_timeout=(a.sim_timeout or None),
            flight=a.flight, data_dir=Path(a.data_dir),
            events_dir=out_dir / "events",
            viz_dir=Path(a.viz_dir) if a.viz_dir else None,
        )
        try:
            result = patrol_mission.run_patrol(
                sc.bridge, sc.sim_tf, sc.points_world, sc.gm, rooms,
                sc.home, cfg,
                on_status=lambda t: self.log.append("status", {"text": t}),
                on_progress=self.log.append,
                follow_path_mod=sc.follow_path,
                should_abort=lambda: self.abort_requested)
            self.last_result = result
            self.last_report_dir = out_dir
            # `cmd` 는 요약 프롬프트의 `사용자_명령` 이 된다 — 콘솔이 실제로 받은
            # 한국어 문장이 들어가야 요약이 "무엇을 시켰는지"를 안다.
            patrol_report.build_report(result, rooms, cmd or "web console",
                                       out_dir, llm=sc.llm)
            self.log.append("report_ready", {"missionId": self.mission_id})
            self.state = "done"
        except Exception as e:
            self.state = "error"
            self.log.append("error", {"message": f"{type(e).__name__}: {e}"})


# ------------------------------------------------------------------------ app

class IntentRequest(BaseModel):
    text: str


class PlanRequest(BaseModel):
    goal: List[float]
    start: Optional[List[float]] = None


class StartRequest(BaseModel):
    # The console's launch payload. `targets[].room` is the web id ("012") and
    # the array order IS the visit order — we do not re-sort it.
    targets: List[Dict[str, Any]]
    returnHome: bool = True
    # 사용자가 콘솔에 친 문장. 보고서 요약 프롬프트의 `사용자_명령` 으로 들어간다.
    # 선택 필드 — 안 보내면 예전처럼 "web console" 로 적힌다.
    cmd: str = ""


def build_app(scene: Scene) -> FastAPI:
    log = EventLog()
    mission = MissionRunner(scene, log)
    app = FastAPI(title="patrol api", version="1",
                  description=__doc__.split("\n\n")[0])

    # ---------------------------------------------------------- console-facing
    @app.get("/api/drone")
    def api_drone() -> dict:
        pos, source, flying = scene.drone_pose()
        return {"x": round(float(pos[0]), 3), "y": round(float(pos[1]), 3),
                "z": round(float(pos[2]), 3), "source": source,
                "flying": flying, "connected": True}

    @app.get("/api/status")
    def api_status() -> dict:
        return {
            "ready": True,
            "engine": _engine(scene),
            "building": scene.args.building,
            "rooms": len(scene.rooms),
            "model": scene.llm.model_id,
            "llmOffline": scene.offline_llm,
            "mission": {"state": mission.state, "id": mission.mission_id,
                        "busy": mission.busy, "seq": log.seq},
        }

    @app.post("/plan")
    def api_plan(req: PlanRequest) -> dict:
        start = (np.asarray(req.start, dtype=float) if req.start
                 else scene.drone_pose()[0])
        goal = scene.snap_goal(req.goal)
        t0 = time.time()
        # 미션과 같은 플래너를 탄다 (patrol_mission.plan_leg): 여유가 넉넉한
        # 격자부터 시도하고 막힌 구간만 조여서 다시 푼다. 경로가 None 인 경우는
        # 없어서 브리핑이 "일부 구간 실패"로 비지 않는다. 둘이 갈라지면 브리핑에
        # 그려진 경로와 실제로 나는 경로가 달라진다.
        path, info = patrol_mission.plan_leg(
            scene.points_world, scene.gm, start, goal,
            algo=scene.args.algo, grid_cache=scene.grid_cache)
        ms = int((time.time() - t0) * 1000)
        return {
            "engine": _engine(scene), "success": True,
            "fallback": info.get("fallback"),   # None | tightened | direct
            "clearanceM": info.get("clearance_m"),
            "steps": int(info["n_waypoints"]), "bumps": 0,
            "dist": round(float(np.linalg.norm(goal - path[-1])), 3),
            "flown": round(float(info["length_m"]), 2), "ms": ms,
            "start": _xyz(start), "goal": _xyz(goal),
            "path": [_xyz(p) for p in path],
        }

    @app.get("/api/rooms")
    def api_rooms() -> dict:
        """Every room, in the console's spelling, with the names we know.

        The console's own `LABELS` table is hardcoded and disagrees with
        `room_aliases.json` — it calls 012 "채원의 금고..." where the aliases
        (and therefore every natural-language query) call it 현우방. This is the
        one list, so the display names can come from here instead.
        """
        out = {}
        for name, r in sorted(scene.rooms.items()):
            out[room_index.web_room_id(name)] = {
                "roomName": name, "floor": r.floor, "type": r.room_type,
                "typeKr": r.type_kr, "display": r.display,
                "aliases": list(r.aliases),
                "center": _xyz(r.centroid),
                "scanPose": _xyz(room_index.scan_pose(
                    r, scene.args.hover_height, scene.gm)),
            }
        return {"building": scene.args.building, "rooms": out}

    # ------------------------------------------------------------------ intent
    @app.post("/api/intent")
    def api_intent(req: IntentRequest) -> dict:
        """Natural language -> room ids the console can hand to setTargets().

        Returns web-spelled ids in visit order, plus the names we know them by,
        because the console's own labels are hardcoded and disagree with the
        aliases the LLM resolved against (see API.md). An empty `rooms` means
        the text named no area we could pin down — `why` says so.
        """
        text = (req.text or "").strip()
        if not text:
            raise HTTPException(400, "text 가 비어 있습니다")
        intent = patrol_intent.parse_patrol(scene.llm, text, scene.rooms)
        rooms, why = patrol_intent.resolve_rooms(
            intent, scene.rooms, text, max_rooms=scene.args.max_rooms)
        # The order the user spoke wins over distance, for the rooms it names;
        # everything else is filled in greedily behind it. The console takes this
        # array as the visit order, so this is the only place order is decided.
        pinned = patrol_intent.resolve_order(intent, rooms)
        rooms = room_index.order_rooms(rooms, scene.drone_pose()[0], pinned)
        if pinned:
            why += f", 순서 지정 {len(pinned)}곳"
        if not rooms and scene.offline_llm:
            # Offline it is the keyword tiers alone, so say which lever is
            # missing instead of letting it read as "the model didn't get it".
            why += " (오프라인 모드 — 방 이름/별칭이나 \"N층\"을 넣어 주세요)"
        return {
            "why": why,
            "rooms": [room_index.web_room_id(r.room_name) for r in rooms],
            "names": {room_index.web_room_id(r.room_name): r.display
                      for r in rooms},
            "returnHome": bool(intent.return_home),
        }

    # ------------------------------------------------------------------ patrol
    @app.post("/api/patrol/start")
    def api_patrol_start(req: StartRequest) -> dict:
        rooms: List[RoomInfo] = []
        unknown: List[str] = []
        for t in req.targets:
            web_id = str(t.get("room", "")).strip()
            info = room_index.by_web_room_id(scene.rooms, web_id)
            if info is None:
                unknown.append(web_id)
            else:
                rooms.append(info)
        if unknown:
            raise HTTPException(400, f"모르는 구역: {', '.join(unknown)}")
        if not rooms:
            raise HTTPException(400, "targets 가 비어 있습니다")
        # Read the cursor BEFORE the mission thread can append to it. The
        # console polls `since=seq`, so a seq read afterwards would silently
        # skip whatever the thread already emitted — mission_start included.
        seq0 = log.seq
        mission_id = mission.start(rooms, req.returnHome,
                                   (req.cmd or "").strip())
        return {"missionId": mission_id,
                "rooms": [r.room_name for r in rooms],
                "seq": seq0}

    @app.get("/api/patrol/events")
    def api_patrol_events(since: int = 0) -> dict:
        return {"seq": log.seq, "state": mission.state,
                "missionId": mission.mission_id,
                "events": log.since(since)}

    @app.post("/api/patrol/abort")
    def api_patrol_abort() -> dict:
        if not mission.busy:
            return {"ok": False, "reason": "진행 중인 순찰이 없습니다"}
        mission.abort()
        return {"ok": True}

    @app.get("/api/patrol/report/{mission_id}")
    def api_patrol_report(mission_id: str) -> dict:
        d = mission.last_report_dir
        if d is None or mission_id != mission.mission_id:
            raise HTTPException(404, "그 순찰의 보고서가 없습니다")
        f = d / "report.json"
        if not f.is_file():
            raise HTTPException(404, "보고서가 아직 생성되지 않았습니다")
        import json
        # utf-8 필수 — 보고서는 한국어이고 윈도우 기본 인코딩은 cp949 다.
        return json.loads(f.read_text(encoding="utf-8"))

    # ------------------------------------------------------------ web/ static
    if WEB_DIR.is_dir():
        @app.get("/")
        def index() -> RedirectResponse:
            return RedirectResponse("/HAUNTED%20OPS.dc.html")

        @app.get("/{path:path}")
        def static_file(path: str) -> FileResponse:
            target = (WEB_DIR / path).resolve()
            if not str(target).startswith(str(WEB_DIR.resolve())):
                raise HTTPException(403, "경로를 벗어났습니다")
            if not target.is_file():
                raise HTTPException(404, path)
            return FileResponse(target)
    else:
        @app.get("/")
        def no_web() -> dict:
            return {"detail": f"{WEB_DIR} 가 없습니다 — API 만 제공합니다. "
                              f"규격은 /docs 또는 API.md"}

    return app


def _engine(scene: Scene) -> str:
    return f"{scene.args.algo.upper()} · voxel {scene.args.resolution} m"


def _xyz(p) -> List[float]:
    return [round(float(v), 3) for v in np.asarray(p, dtype=float)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    # scene
    ap.add_argument("--data-dir",
                    default=str(_THIS.parent / "data" / "final_npy"))
    ap.add_argument("--building", default="00809_Qpor2mEya8F")
    ap.add_argument("--room-aliases", default=None)
    ap.add_argument("--point-stride", type=int, default=4)
    ap.add_argument("--resolution", type=float, default=0.15)
    ap.add_argument("--margin", type=int, default=1)
    ap.add_argument("--sample", type=int, default=1)
    ap.add_argument("--algo", default="astar", choices=["astar", "rrt"])
    ap.add_argument("--flight", default="pid", choices=["pid", "rl"],
                    help="구간 추종기. pid=기존 PID, rl=강화학습 정책(A* 경로를 따라가되 주변은 정책이 본다). rl 실패 시 자동으로 pid 폴백.")
    ap.add_argument("--rrt-iter", type=int, default=8000)
    # llm (never loaded here — llm_server/ holds the model).
    # Optional on purpose: without it the server runs offline (see the module
    # docstring) so the console and the mission loop can be tested alone.
    ap.add_argument("--llm-url", default=DEFAULT_LLM_URL,
                    help="OpenAI 호환 엔드포인트. 기본은 이 PC의 Ollama. "
                         '"gpu" = 학교 GPU 서버(VPN 필요), '
                         '빈 문자열("")이면 오프라인 모드')
    ap.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    ap.add_argument("--llm-api-key", default=None)
    ap.add_argument("--llm-timeout", type=float, default=60.0)
    # unity link — localhost now that the pipeline runs beside the simulator
    ap.add_argument("--unity-host", default="127.0.0.1")
    ap.add_argument("--unity-port", type=int, default=9000)
    ap.add_argument("--unity-local-port", type=int, default=9001)
    ap.add_argument("--unity-state-port", type=int, default=9002)
    # 6.0 u/s = 1.2 m/s (집이 scale 5). rc-limit 는 speed*100/15 이상이어야
    # 한다 — 아니면 rc 클리핑이 속도를 대신 깎는다.
    ap.add_argument("--sim-speed", type=float, default=6.0)
    ap.add_argument("--sim-rc-limit", type=int, default=60)
    ap.add_argument("--sim-timeout", type=float, default=0.0)
    # patrol
    ap.add_argument("--hover-height", type=float, default=1.2)
    ap.add_argument("--scan-deg-per-sec", type=float, default=50.0)
    ap.add_argument("--scan-turns", type=float, default=1.0)
    ap.add_argument("--scan-mode", default="auto",
                    choices=["auto", "unity", "rc"])
    ap.add_argument("--patrol-labels", nargs="*", default=["person"])
    ap.add_argument("--patrol-min-conf", type=float, default=0.0)
    ap.add_argument("--max-rooms", type=int, default=12)
    ap.add_argument("--no-light", action="store_true")
    ap.add_argument("--report-dir",
                    default=str(_THIS.parent / "patrol" / "out" / "reports"))
    ap.add_argument("--viz-dir", default=None)
    args = ap.parse_args()

    import uvicorn
    app = build_app(Scene(args))
    print(f"[api] http://{args.host}:{args.port}   (규격: /docs)")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
