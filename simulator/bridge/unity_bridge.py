"""
UDP bridge for the Unity-based Tello simulator.

Two channels, both UDP:

    server -> Unity, port 9000   verbs (command/takeoff/land/rc/setpos/msg/
                                 light/scan), each acked with "ok"
    Unity -> server, port 9002   20 Hz state JSON, plus events — a packet with
                                 an "event" key is an event, not a state:

        {"event": "scan_done", "degrees": 360.0}
        {"event": "detect", "label": "person", "conf": 0.87,
         "box": {"l": 21.0, "t": 33.0, "w": 16.0, "h": 40.0},
         "image_path": "/abs/detection_frames/x.jpg"}

`box` is in **percent of the camera frame**, which is what the web console's
`window.__patrolDetect` wants, so nothing on the way has to rescale it. Unity
knows its own capture resolution; the pixel→percent division belongs there.

Unknown verbs are logged and still acked "ok" by both Unity and
`fake_unity_sim.py`, so adding one is backward compatible — but that also means
an ack does NOT prove the verb was implemented. Where that matters (`scan`) the
caller waits for the answering event and falls back if it never comes.
"""

from __future__ import annotations

import collections
import json
import socket
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class DroneState:
    x: float
    y: float
    z: float
    yaw: float
    flying: bool
    had_collision: bool
    collision_count: int
    time: float


class UnityTelloBridge:
    def __init__(
        self,
        unity_host: str = "127.0.0.1",
        command_port: int = 9000,
        local_command_port: int = 9001,
        local_state_port: int = 9002,
    ) -> None:
        self.unity_host = unity_host
        self.command_port = command_port
        self.local_command_port = local_command_port
        self.local_state_port = local_state_port

        self.command_socket: Optional[socket.socket] = None
        self.state_socket: Optional[socket.socket] = None
        self.state_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._state_lock = threading.Lock()
        self._latest_state: Optional[DroneState] = None
        self._event_lock = threading.Lock()
        self._events: collections.deque[dict] = collections.deque(maxlen=64)

    def connect(self) -> None:
        self.command_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.command_socket.bind(("0.0.0.0", self.local_command_port))
        self.command_socket.settimeout(0.2)

        self.state_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.state_socket.bind(("0.0.0.0", self.local_state_port))
        self.state_socket.settimeout(0.2)

        self.state_thread = threading.Thread(target=self._state_loop, daemon=True)
        self.state_thread.start()

    def close(self) -> None:
        self._stop_event.set()
        if self.state_thread and self.state_thread.is_alive():
            self.state_thread.join(timeout=1.0)
        if self.command_socket:
            self.command_socket.close()
            self.command_socket = None
        if self.state_socket:
            self.state_socket.close()
            self.state_socket = None

    def send_command(self, command: str, expect_reply: bool = True) -> str:
        if not self.command_socket:
            raise RuntimeError("Bridge is not connected.")

        self.command_socket.sendto(
            command.encode("utf-8"),
            (self.unity_host, self.command_port),
        )
        if not expect_reply:
            return ""

        try:
            response, _ = self.command_socket.recvfrom(1024)
            return response.decode("utf-8", errors="replace").strip()
        except (socket.timeout, ConnectionResetError, OSError):
            return "timeout"

    def send_rc(self, lr: int, fb: int, ud: int, yaw: int) -> str:
        return self.send_command(f"rc {lr} {fb} {ud} {yaw}")

    def set_position(self, x: float, y: float, z: float, yaw: float | None = None) -> str:
        """Teleport the simulator drone to a world-space start position (sim only)."""
        if yaw is None:
            return self.send_command(f"setpos {x} {y} {z}")
        return self.send_command(f"setpos {x} {y} {z} {yaw}")

    def takeoff(self) -> str:
        return self.send_command("takeoff")

    def land(self) -> str:
        return self.send_command("land")

    def initialize_sdk(self) -> str:
        return self.send_command("command")

    def send_status(self, text: str) -> None:
        """On-screen status message shown by the simulator. Unity acks every
        packet with "ok", so consume the reply to keep the socket drained."""
        self.send_command(f"msg {text}")

    def request_state(self) -> str:
        return self.send_command("state")

    def set_light(self, on: bool) -> str:
        """Drone light on/off — used when a patrol detects a person in the dark.

        Unity does not implement `light` yet (HorrorAtmosphere's flashlight is
        keyboard-only); unknown verbs are logged and still acked "ok"
        (TelloSimulator.cs ProcessCommand), so sending this is harmless today and
        starts working the moment the Unity side adds its branch. See
        docs/patrol-agent.md for the patch.
        """
        return self.send_command(f"light {'on' if on else 'off'}")

    # --------------------------------------------------------------- scan ---
    def start_scan(self, deg_per_sec: float, turns: float = 1.0) -> str:
        """Ask Unity to spin in place and run its own person detection.

        The sweep, the camera capture and the YOLO round trip all live in
        Unity (`PatrolPersonDetection.cs`); it answers with `detect` events as
        it finds people and one `scan_done` when the sweep finishes.

        The "ok" reply means the packet arrived, NOT that the verb exists — an
        older build acks and ignores it. Wait for `scan_done` and fall back
        (see `patrol_mission.scan_room`).
        """
        return self.send_command(f"scan {deg_per_sec:g} {turns:g}")

    def stop_scan(self) -> str:
        return self.send_command("scan_stop")

    # ------------------------------------------------------------- events ---
    def drain_events(self) -> None:
        """Discard queued events (call before starting something you'll wait on)."""
        with self._event_lock:
            self._events.clear()

    def pop_events(self, name: Optional[str] = None) -> List[dict]:
        """Take every queued event, or only those with `event == name`.

        Non-blocking. Events not matching `name` stay queued in order, so a
        scan loop can drain `detect` without losing the `scan_done` behind it.
        """
        with self._event_lock:
            if name is None:
                out = list(self._events)
                self._events.clear()
                return out
            out = [e for e in self._events if e.get("event") == name]
            keep = [e for e in self._events if e.get("event") != name]
            self._events.clear()
            self._events.extend(keep)
            return out

    def wait_for_event(self, timeout: float,
                       name: Optional[str] = None) -> Optional[dict]:
        """Next event (optionally only `name`), or None on timeout."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._event_lock:
                for i, ev in enumerate(self._events):
                    if name is None or ev.get("event") == name:
                        del self._events[i]
                        return ev
            time.sleep(0.05)
        return None

    def get_latest_state(self) -> Optional[DroneState]:
        with self._state_lock:
            return self._latest_state

    def wait_for_state(self, timeout: float = 3.0) -> Optional[DroneState]:
        deadline = time.time() + timeout
        while time.time() < deadline:
            state = self.get_latest_state()
            if state is not None:
                return state
            self.request_state()
            time.sleep(0.05)
        return None

    def _state_loop(self) -> None:
        assert self.state_socket is not None
        while not self._stop_event.is_set():
            try:
                payload, _ = self.state_socket.recvfrom(4096)
                parsed = json.loads(payload.decode("utf-8"))
                if "event" in parsed:  # an event, not a state packet
                    with self._event_lock:
                        self._events.append(parsed)
                    continue
                state = DroneState(
                    x=float(parsed["x"]),
                    y=float(parsed["y"]),
                    z=float(parsed["z"]),
                    yaw=float(parsed.get("yaw", 0.0)),
                    flying=bool(parsed.get("flying", False)),
                    had_collision=bool(parsed.get("had_collision", False)),
                    collision_count=int(parsed.get("collision_count", 0)),
                    time=float(parsed.get("time", 0.0)),
                )
                with self._state_lock:
                    self._latest_state = state
            except socket.timeout:
                continue
            except OSError:
                break
            except Exception:
                continue
