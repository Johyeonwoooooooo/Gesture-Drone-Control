"""
Protocol-faithful stdlib stub of the Unity Tello simulator, for testing the
bridge/server without Unity: listens on UDP 9000, replies "ok" to every packet,
integrates `rc` at TelloSimulator.cs's moveSpeed (15 u/s at rc=100), and streams
the JSON state packet to <sender_ip>:9002 at 20 Hz. Obstacle-free.

Also stands in for Unity's side of the patrol scan: `scan <deg/s> <turns>`
spins the stub, optionally emits fake `detect` events, and answers `scan_done`.
That makes the whole patrol loop testable without Unity or YOLO. Pass
`--no-scan` to imitate an OLD build that has no `scan` verb (acks and ignores
it), which is what exercises the rc-rotation fallback.

Usage: python fake_unity_sim.py [--port 9000] [--state-port 9002] [--verbose]
                                [--no-scan] [--detect-per-scan 1]
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
    def __init__(self, port: int = 9000, state_port: int = 9002,
                 verbose: bool = False, support_scan: bool = True,
                 detect_per_scan: int = 0):
        self.port = port
        self.state_port = state_port
        self.verbose = verbose
        self.support_scan = support_scan
        self.detect_per_scan = detect_per_scan
        self.scan_thread: threading.Thread | None = None
        self.scan_stop = threading.Event()
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
            elif lower.startswith("scan_stop"):
                self.scan_stop.set()
                print("[fake-sim] SCAN stop")
            elif lower.startswith("scan"):
                if not self.support_scan:
                    # An old build: acked "ok" by the caller's socket, silently
                    # dropped here. This is exactly what the fallback is for.
                    print("[fake-sim] SCAN ignored (--no-scan)")
                else:
                    parts = lower.split()
                    dps = float(parts[1]) if len(parts) > 1 else 50.0
                    turns = float(parts[2]) if len(parts) > 2 else 1.0
                    self._start_scan(dps, turns)
            elif lower.startswith("light"):
                # Patrol reaction: hover -> LIGHT ON -> photo. Unity itself does
                # not implement this verb yet (see docs/patrol-agent.md); the
                # stub prints it so the reaction order is verifiable headless.
                self.light_on = lower.endswith("on")
                print(f"[fake-sim] LIGHT {'ON' if self.light_on else 'OFF'}")
            # "command" / "state" need no simulation-side effect.

    def _start_scan(self, deg_per_sec: float, turns: float) -> None:
        """Spin like Unity would, then answer scan_done."""
        self.scan_stop.set()
        if self.scan_thread is not None and self.scan_thread.is_alive():
            self.scan_thread.join(timeout=1.0)
        self.scan_stop = threading.Event()

        target = 360.0 * max(0.1, turns)
        dps = max(1.0, deg_per_sec)
        print(f"[fake-sim] SCAN {dps:g} deg/s x {turns:g} ({target / dps:.1f}s)")

        self._send_event({"event": "scan_started", "target": round(target, 1)})

        def _run(stop: threading.Event) -> None:
            turned = 0.0
            dt = 1.0 / PHYSICS_HZ
            fired = 0
            # Space the fake detections through the sweep instead of all at 0°.
            fire_at = [target * (i + 1) / (self.detect_per_scan + 1)
                       for i in range(self.detect_per_scan)]
            while turned < target and not stop.is_set():
                time.sleep(dt)
                with self.lock:
                    self.yaw = (self.yaw + dps * dt) % 360.0
                turned += dps * dt
                while fired < len(fire_at) and turned >= fire_at[fired]:
                    fired += 1
                    self._send_event({
                        "event": "detect", "label": "person",
                        "conf": 0.87,
                        "box": {"l": 21.0, "t": 33.0, "w": 16.0, "h": 40.0},
                        "source": "fake-sim",
                    })
            if not stop.is_set():
                self._send_event({"event": "scan_done", "degrees": round(turned, 1)})

        self.scan_thread = threading.Thread(target=_run, args=(self.scan_stop,),
                                            daemon=True)
        self.scan_thread.start()

    def _send_event(self, payload: dict) -> None:
        ip = self.last_remote_ip
        if ip is None:
            return
        print(f"[fake-sim] EVENT: {payload}")
        try:
            self.state_sock.sendto(
                json.dumps(payload).encode("utf-8"), (ip, self.state_port))
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
    ap.add_argument("--no-scan", action="store_true",
                    help="Imitate a build without the `scan` verb (tests the "
                         "rc-rotation fallback).")
    ap.add_argument("--detect-per-scan", type=int, default=0,
                    help="Fake person detections to emit during each scan.")
    args = ap.parse_args()
    FakeUnitySim(args.port, args.state_port, args.verbose,
                 support_scan=not args.no_scan,
                 detect_per_scan=args.detect_per_scan).run()


if __name__ == "__main__":
    main()
