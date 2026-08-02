"""Turn a planned world-meter waypoint path into a DJI Tello SDK command program.

This branch flies in the simulator only (simulator/bridge), so the emitted
program is an artifact, not an execution path: a record of the mission in the
real-SDK vocabulary, kept so a real-drone layer can replay it later.

The shape is the Ollama-style tool-call dict used by the real-drone code on the
`gyucheol` branch (`playground/` there) —
`{"function": {"name": <tool>, "arguments": {...}}}` — dispatchable with
`getattr(drone, name)(**args)` onto a djitellopy `Tello`. 3D path segments use
`go_xyz_speed(x, y, z, speed)`.

Tello SDK constraints honored here:
  - go_xyz_speed: x, y, z in **cm**, each in [-500, 500]; speed in cm/s [10, 100].
    The 'go' command rejects a move where every axis is within +/-20 cm, so we
    merge near-zero segments and split any axis exceeding 500 cm.
  - takeoff / land: no args.

World->body frame: fixed-heading assumption — the drone faces +world-x, so
body x = world x (forward), body y = world y (left), body z = world z (up),
meters -> cm via x100. Real yaw tracking is future work (documented in meta).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

# Tello 'go' limits (cm)
_GO_MAX = 500        # per-axis magnitude cap
_GO_MIN_MOVE = 20    # a 'go' with all axes within +/-this is rejected by the SDK


def _tool(name: str, sdk: str, **args) -> dict:
    """One gyucheol-style tool-call entry."""
    return {"function": {"name": name, "arguments": args}, "sdk": sdk}


def _go_command(dx_cm: int, dy_cm: int, dz_cm: int, speed: int) -> dict:
    return _tool("go_xyz_speed", f"go {dx_cm} {dy_cm} {dz_cm} {speed}",
                 x=dx_cm, y=dy_cm, z=dz_cm, speed=speed)


def _segments_to_go_commands(
    waypoints_cm: Sequence[np.ndarray], speed: int
) -> List[dict]:
    """Convert a body-frame cm waypoint list into go_xyz_speed commands.

    Merges consecutive deltas until the accumulated move clears the +/-20cm
    minimum, then splits it across axes into <=500cm chunks.
    """
    cmds: List[dict] = []
    acc = np.zeros(3, dtype=float)
    for i in range(1, len(waypoints_cm)):
        acc += np.asarray(waypoints_cm[i], dtype=float) - np.asarray(waypoints_cm[i - 1], dtype=float)
        # Not yet a legal move? keep accumulating (Tello rejects all-axes<20).
        if np.all(np.abs(acc) < _GO_MIN_MOVE):
            continue
        cmds.extend(_emit_go(acc, speed))
        acc = np.zeros(3, dtype=float)
    # Flush any remaining motion that is itself a legal move.
    if np.any(np.abs(acc) >= _GO_MIN_MOVE):
        cmds.extend(_emit_go(acc, speed))
    return cmds


def _emit_go(delta_cm: np.ndarray, speed: int) -> List[dict]:
    """Split one accumulated delta into >=1 go commands each within +/-500cm,
    keeping every emitted command a legal (not all-axes<20) move."""
    d = np.rint(delta_cm).astype(int)
    n_split = max(1, int(np.ceil(np.max(np.abs(d)) / _GO_MAX)))
    step = d // n_split
    out: List[dict] = []
    emitted = np.zeros(3, dtype=int)
    for k in range(n_split):
        chunk = step if k < n_split - 1 else (d - emitted)
        emitted += chunk
        cx, cy, cz = int(chunk[0]), int(chunk[1]), int(chunk[2])
        # A chunk could fall under the min-move floor after splitting; nudge the
        # dominant axis so the SDK still accepts it.
        if abs(cx) < _GO_MIN_MOVE and abs(cy) < _GO_MIN_MOVE and abs(cz) < _GO_MIN_MOVE:
            j = int(np.argmax(np.abs([cx, cy, cz])))
            vals = [cx, cy, cz]
            vals[j] = _GO_MIN_MOVE if vals[j] >= 0 else -_GO_MIN_MOVE
            cx, cy, cz = vals
        out.append(_go_command(cx, cy, cz, speed))
    return out


def build_tello_program(
    waypoints_world_m: Sequence[np.ndarray],
    *,
    action: str,
    return_home: bool,
    home_world,
    start_world,
    goal_world,
    target_object: str,
    clip_prompt: str,
    query: str,
    algo: str,
    building: str,
    speed: int = 40,
    timestamp: str = "",
) -> dict:
    """Build a Tello command program (dict) from a world-meter waypoint path.

    `action` is the parsed intent action ("take_photo" | "inspect" | "goto" |
    "other"); `return_home` appends a reversed leg back to `home_world`.
    """
    speed = int(max(10, min(100, speed)))
    wps = [np.asarray(p, dtype=float) for p in (waypoints_world_m or [])]

    # World meters -> body cm (fixed-heading: body == world axes, x100).
    def to_cm(seq):
        return [np.asarray(p, dtype=float) * 100.0 for p in seq]

    commands: List[dict] = [_tool("takeoff", "takeoff")]

    if len(wps) >= 2:
        commands.extend(_segments_to_go_commands(to_cm(wps), speed))

    if action == "take_photo":
        commands.append(_tool("streamon", "streamon"))
        # gyucheol has no still-capture tool; real capture is
        # cv2.imwrite(get_frame_read().frame). Represented as a marker command.
        commands.append(_tool("take_photo", "# capture frame -> cv2.imwrite"))

    if return_home and wps:
        # Reversed leg from the goal back to home.
        home = np.asarray(home_world, dtype=float)
        back = list(reversed(wps)) + [home]
        commands.extend(_segments_to_go_commands(to_cm(back), speed))

    commands.append(_tool("land", "land"))

    return {
        "meta": {
            "query": query,
            "target_object": target_object,
            "clip_prompt": clip_prompt,
            "action": action,
            "return_home": bool(return_home),
            "algo": algo,
            "building": building,
            "home_world": [float(v) for v in np.asarray(home_world, dtype=float)],
            "start_world": [float(v) for v in np.asarray(start_world, dtype=float)],
            "goal_world": [float(v) for v in np.asarray(goal_world, dtype=float)],
            "path_length_m": _length(wps),
            "n_waypoints": len(wps),
            "speed": speed,
            "world_to_body": "fixed-heading: body x=world x (fwd), y=world y (left), "
                             "z=world z (up); meters*100 -> cm",
            "timestamp": timestamp,
        },
        "waypoints_world_m": [[float(v) for v in p] for p in wps],
        "commands": commands,
    }


def _length(wps: List[np.ndarray]) -> float:
    if len(wps) < 2:
        return 0.0
    return float(sum(float(np.linalg.norm(wps[i] - wps[i - 1])) for i in range(1, len(wps))))


def _slug(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", (text or "target").strip().lower()).strip("-")
    return s[:40] or "target"


def save_program(program: dict, out_dir, *, filename: Optional[str] = None) -> Path:
    """Write `program` as JSON under `out_dir`, return the path."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if filename is None:
        ts = program.get("meta", {}).get("timestamp", "") or "0"
        ts = re.sub(r"[^0-9A-Za-z]+", "", ts) or "0"
        slug = _slug(program.get("meta", {}).get("target_object", ""))
        filename = f"{ts}_{slug}.json"
    path = out / filename
    path.write_text(json.dumps(program, ensure_ascii=False, indent=2))
    return path
