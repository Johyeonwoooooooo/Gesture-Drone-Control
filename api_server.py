"""api_server.py — the web console's backend. Runs next to Unity, not on the GPU box.

The HAUNTED OPS console (`web/`) picks patrol areas on a floor plan and draws
what happens; everything behind it lives here. One process, because there is
one drone and one UDP link to it — two backends cannot both hold 9000/9002, and
the console calls `api/drone` and `plan` as RELATIVE paths, so the static files
and the API have to share an origin anyway.

    python api_server.py --llm-url http://<GPU서버>:8000/v1    # + --api-key

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

from patrol import (patrol_intent, patrol_mission, patrol_report,  # noqa: E402
                    planner, room_index)
from patrol.litept_backend import LitePTBackend  # noqa: E402
from patrol.remote_llm import RemoteLLMParser  # noqa: E402
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
        self.home = self.backend.default_home()
        print(f"[api] scene ready: grid={self.gm.shape}, "
              f"home={np.round(self.home, 2)} ({time.time() - t0:.1f}s)")

        self.llm = RemoteLLMParser(args.llm_url, model_id=args.llm_model,
                                   api_key=args.llm_api_key,
                                   timeout=args.llm_timeout)
        served = self.llm.ping()
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

    def start(self, rooms: List[RoomInfo], return_home: bool) -> str:
        with self._lock:
            if self.busy:
                raise HTTPException(409, "순찰이 이미 진행 중입니다")
            self.mission_id = time.strftime("%Y%m%d_%H%M%S")
            self.state = "running"
            self.abort_requested = False
            self.log.clear()
            self._thread = threading.Thread(
                target=self._run, args=(rooms, return_home), daemon=True)
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

    def _run(self, rooms: List[RoomInfo], return_home: bool) -> None:
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
            events_dir=out_dir / "events",
            viz_dir=Path(a.viz_dir) if a.viz_dir else None,
        )
        try:
            result = patrol_mission.run_patrol(
                sc.bridge, sc.sim_tf, sc.points_world, sc.gm, rooms,
                sc.home, cfg,
                on_status=lambda t: self.log.append("status", {"text": t}),
                on_progress=self.log.append,
                follow_path_mod=sc.follow_path)
            self.last_result = result
            self.last_report_dir = out_dir
            patrol_report.build_report(result, rooms, "web console", out_dir,
                                       llm=sc.llm)
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
            "mission": {"state": mission.state, "id": mission.mission_id,
                        "busy": mission.busy, "seq": log.seq},
        }

    @app.post("/plan")
    def api_plan(req: PlanRequest) -> dict:
        start = (np.asarray(req.start, dtype=float) if req.start
                 else scene.drone_pose()[0])
        goal = scene.snap_goal(req.goal)
        t0 = time.time()
        path, info, _ = planner.plan_path(
            scene.points_world, start, goal, algo=scene.args.algo,
            gm=scene.gm, rrt_iter=scene.args.rrt_iter)
        ms = int((time.time() - t0) * 1000)
        if path is None:
            return {"engine": _engine(scene), "success": False,
                    "error": info.get("reason", "no path"),
                    "steps": 0, "bumps": 0, "dist": 0.0, "flown": 0.0,
                    "ms": ms, "start": _xyz(start), "goal": _xyz(goal),
                    "path": []}
        return {
            "engine": _engine(scene), "success": True,
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
        rooms = room_index.order_rooms(rooms, scene.drone_pose()[0])
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
        mission_id = mission.start(rooms, req.returnHome)
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
        return json.loads(f.read_text())

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
    ap.add_argument("--rrt-iter", type=int, default=8000)
    # llm (never loaded here — llm_server/ holds the model)
    ap.add_argument("--llm-url", required=True)
    ap.add_argument("--llm-model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--llm-api-key", default=None)
    ap.add_argument("--llm-timeout", type=float, default=60.0)
    # unity link — localhost now that the pipeline runs beside the simulator
    ap.add_argument("--unity-host", default="127.0.0.1")
    ap.add_argument("--unity-port", type=int, default=9000)
    ap.add_argument("--unity-local-port", type=int, default=9001)
    ap.add_argument("--unity-state-port", type=int, default=9002)
    ap.add_argument("--sim-speed", type=float, default=2.0)
    ap.add_argument("--sim-rc-limit", type=int, default=30)
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
