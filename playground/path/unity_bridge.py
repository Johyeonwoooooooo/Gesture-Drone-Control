"""
UDP bridge for the Unity-based Tello simulator.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from dataclasses import dataclass
from typing import Optional


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
            command.encode("ascii"),
            (self.unity_host, self.command_port),
        )
        if not expect_reply:
            return ""

        try:
            response, _ = self.command_socket.recvfrom(1024)
            return response.decode("ascii").strip()
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

    def request_state(self) -> str:
        return self.send_command("state")

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
                parsed = json.loads(payload.decode("ascii"))
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
