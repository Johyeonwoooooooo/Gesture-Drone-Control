"""
Protocol-faithful stdlib stub of the Unity Tello simulator, for testing the
bridge/server without Unity: listens on UDP 9000, replies "ok" to every packet,
integrates `rc` at TelloSimulator.cs's moveSpeed (15 u/s at rc=100), and streams
the JSON state packet to <sender_ip>:9002 at 20 Hz. Obstacle-free.

Also fakes the preview/confirm user flow: on `preview x y z [label]` it can
auto-answer with `{"event":"next"}` K times then `{"event":"confirm"}` after a
delay, so the full confirm-loop pipeline is testable headless.

Usage: python fake_unity_sim.py [--port 9000] [--state-port 9002] [--verbose]
                                [--auto-confirm-sec 3] [--auto-next 1]
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import threading
import time

MOVE_SPEED = 15.0       # u/s at rc=100, matches TelloSimulator.cs moveSpeed
ROTATION_SPEED = 100.0  # deg/s at rc=100, matches TelloSimulator.cs rotationSpeed
MIN_HEIGHT = 0.5
TAKEOFF_LIFT = 1.0
STATE_HZ = 20.0
PHYSICS_HZ = 50.0


class FakeUnitySim:
    def __init__(self, port: int = 9000, state_port: int = 9002, verbose: bool = False,
                 auto_confirm_sec: float = 0.0, auto_next: int = 0):
        self.port = port
        self.state_port = state_port
        self.verbose = verbose
        self.auto_confirm_sec = auto_confirm_sec
        self.auto_next_total = auto_next
        self.auto_next_left = auto_next
        self.preview_timer: threading.Timer | None = None
        self.lock = threading.Lock()
        self.pos = [0.0, MIN_HEIGHT, 0.0]
        self.yaw = 0.0
        self.rc = [0, 0, 0, 0]          # lr, fb, ud, yaw
        self.flying = False
        self.light_on = False
        self.collision_count = 0
        self.last_remote_ip: str | None = None
        self.start_time = time.time()
        self._stop = threading.Event()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", self.port))
        self.sock.settimeout(0.2)
        self.state_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def run(self) -> None:
        threads = [
            threading.Thread(target=self._physics_loop, daemon=True),
            threading.Thread(target=self._state_loop, daemon=True),
        ]
        for t in threads:
            t.start()
        print(f"[fake-sim] UDP server listening on {self.port}, state -> :{self.state_port}")
        try:
            while not self._stop.is_set():
                try:
                    data, remote = self.sock.recvfrom(2048)
                except socket.timeout:
                    continue
                msg = data.decode("utf-8", errors="replace").strip()
                self.sock.sendto(b"ok", remote)
                self.last_remote_ip = remote[0]
                self._handle(msg)
        except KeyboardInterrupt:
            pass
        finally:
            self._stop.set()

    def _handle(self, msg: str) -> None:
        lower = msg.lower()
        if self.verbose and not lower.startswith("rc"):
            print(f"[fake-sim] cmd: {msg}")
        with self.lock:
            if lower == "takeoff":
                self.flying = True
                self.pos[1] = max(self.pos[1], MIN_HEIGHT) + TAKEOFF_LIFT
                self.rc = [0, 0, 0, 0]
            elif lower == "land":
                self.flying = False
                self.rc = [0, 0, 0, 0]
                self.pos[1] = MIN_HEIGHT
            elif lower.startswith("rc "):
                parts = lower.split()
                if len(parts) == 5:
                    try:
                        self.rc = [max(-100, min(100, int(float(p)))) for p in parts[1:]]
                    except ValueError:
                        pass
            elif lower.startswith("setpos "):
                parts = lower.split()
                if len(parts) >= 4:
                    try:
                        self.pos = [float(parts[1]), float(parts[2]), float(parts[3])]
                        if len(parts) >= 5:
                            self.yaw = float(parts[4])
                    except ValueError:
                        pass
            elif lower.startswith("msg "):
                print(f"[fake-sim] STATUS: {msg[4:]}")
            elif lower.startswith("preview "):
                print(f"[fake-sim] PREVIEW: {msg[8:]}")
                self._cancel_preview_timer()
                if self.auto_confirm_sec > 0:
                    ev = "next" if self.auto_next_left > 0 else "confirm"
                    if ev == "next":
                        self.auto_next_left -= 1
                    self.preview_timer = threading.Timer(
                        self.auto_confirm_sec, self._send_event, args=(ev,))
                    self.preview_timer.daemon = True
                    self.preview_timer.start()
            elif lower == "preview_off":
                print("[fake-sim] PREVIEW off")
                self._cancel_preview_timer()
                self.auto_next_left = self.auto_next_total
            elif lower.startswith("light"):
                # Patrol reaction: hover -> LIGHT ON -> photo. Unity itself does
                # not implement this verb yet (see docs/patrol-agent.md); the
                # stub prints it so the reaction order is verifiable headless.
                self.light_on = lower.endswith("on")
                print(f"[fake-sim] LIGHT {'ON' if self.light_on else 'OFF'}")
            # "command" / "state" need no simulation-side effect.

    def _cancel_preview_timer(self) -> None:
        if self.preview_timer is not None:
            self.preview_timer.cancel()
            self.preview_timer = None

    def _send_event(self, name: str) -> None:
        ip = self.last_remote_ip
        if ip is None:
            return
        print(f"[fake-sim] auto EVENT: {name}")
        try:
            self.state_sock.sendto(
                json.dumps({"event": name}).encode("utf-8"), (ip, self.state_port))
        except OSError:
            pass

    def _physics_loop(self) -> None:
        dt = 1.0 / PHYSICS_HZ
        while not self._stop.is_set():
            with self.lock:
                if self.flying:
                    lr, fb, ud, yaw_rc = self.rc
                    lx = lr / 100.0 * MOVE_SPEED
                    ly = ud / 100.0 * MOVE_SPEED
                    lz = fb / 100.0 * MOVE_SPEED
                    yaw = math.radians(self.yaw)
                    # inverse of world_velocity_to_rc's world->local rotation
                    wx = math.cos(yaw) * lx + math.sin(yaw) * lz
                    wz = -math.sin(yaw) * lx + math.cos(yaw) * lz
                    self.pos[0] += wx * dt
                    self.pos[1] = max(MIN_HEIGHT, self.pos[1] + ly * dt)
                    self.pos[2] += wz * dt
                    self.yaw = (self.yaw + yaw_rc / 100.0 * ROTATION_SPEED * dt) % 360.0
            time.sleep(dt)

    def _state_loop(self) -> None:
        dt = 1.0 / STATE_HZ
        while not self._stop.is_set():
            ip = self.last_remote_ip
            if ip is not None:
                with self.lock:
                    payload = {
                        "x": self.pos[0],
                        "y": self.pos[1],
                        "z": self.pos[2],
                        "yaw": self.yaw,
                        "flying": self.flying,
                        "had_collision": False,
                        "collision_count": self.collision_count,
                        "time": time.time() - self.start_time,
                    }
                try:
                    self.state_sock.sendto(
                        json.dumps(payload).encode("utf-8"), (ip, self.state_port)
                    )
                except OSError:
                    pass
            time.sleep(dt)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=9000)
    ap.add_argument("--state-port", type=int, default=9002)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--auto-confirm-sec", type=float, default=0.0,
                    help="Seconds after a preview to auto-answer (0 = never).")
    ap.add_argument("--auto-next", type=int, default=0,
                    help="Answer 'next' this many times before 'confirm'.")
    args = ap.parse_args()
    FakeUnitySim(args.port, args.state_port, args.verbose,
                 args.auto_confirm_sec, args.auto_next).run()


if __name__ == "__main__":
    main()
