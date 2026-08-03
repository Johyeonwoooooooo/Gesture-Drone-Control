"""Detection-event ingest — an out-of-process 2D detector talking to the agent.

NOTE: this is no longer the primary path. Unity does the 360° scan and the
person detection itself (`PatrolPersonDetection.cs` → YOLO over TCP 9100) and
reports `detect` events on the normal state channel, which is what
`patrol_mission.scan_room` consumes. This module stays for a detector that runs
as its own process and pushes to UDP 9004 — it is not wired up by default.

Wire format — one JSON object per datagram::

    {
      "label":      "person",              # REQUIRED
      "conf":       0.87,                  # optional, 0..1
      "bbox":       [x, y, w, h],          # optional, image pixels
      "box":        {"l":21,"t":33,"w":16,"h":40},   # optional, frame percent
      "image_path": "/abs/path/evt.jpg",   # optional but strongly preferred
      "ts":         1754035212.4,          # optional, detector's unix time
      "source":     "yolo26n"              # optional, free text
    }

`box` (percent) is what the web console draws with; `bbox` (pixels) is kept for
detectors that only know pixels. Send whichever you have.

Only `label` is required. Without `image_path` the event is still recorded and
reported, just without a photo — the mission never fails over a missing file.

Gating: detections are only accepted while the agent has ARMED the listener,
i.e. while the drone is scanning inside a patrol room. Everything that arrives
during transit is counted (`ignored_count`) and dropped, which is the flow's
"이동 중에는 탐지하지 않고 순찰 구역에서만 탐지" rule.

Test without the detector::

    python -m patrol.detect_events --emit --label person --conf 0.9 \\
        --image /path/to/any.jpg
"""
from __future__ import annotations

import argparse
import collections
import json
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Sequence

DEFAULT_PORT = 9004


@dataclass
class DetectionEvent:
    label: str
    conf: float = 0.0
    bbox: Optional[List[float]] = None          # image pixels [x, y, w, h]
    box_pct: Optional[Dict[str, float]] = None  # frame percent {l, t, w, h}
    image_path: Optional[str] = None
    ts: float = 0.0                  # detector-supplied unix time (0 if absent)
    recv_ts: float = 0.0             # when WE received it — authoritative
    source: str = ""
    raw: dict = field(default_factory=dict)

    # Filled in by the agent at reaction time (see patrol_mission.on_detection).
    room_name: str = ""
    room_display: str = ""
    drone_world: Optional[List[float]] = None
    drone_unity: Optional[List[float]] = None
    saved_image: Optional[str] = None

    @property
    def iso_time(self) -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S",
                             time.localtime(self.recv_ts or self.ts or time.time()))

    def to_json(self) -> dict:
        return {
            "label": self.label, "conf": self.conf, "bbox": self.bbox,
            "box_pct": self.box_pct,
            "image_path": self.image_path, "ts": self.ts, "recv_ts": self.recv_ts,
            "iso_time": self.iso_time, "source": self.source,
            "room_name": self.room_name, "room_display": self.room_display,
            "drone_world": self.drone_world, "drone_unity": self.drone_unity,
            "saved_image": self.saved_image,
        }

    @staticmethod
    def from_payload(d: dict) -> Optional["DetectionEvent"]:
        label = str(d.get("label") or d.get("class") or d.get("name") or "").strip()
        if not label:
            return None
        try:
            conf = float(d.get("conf", d.get("confidence", d.get("score", 0.0))))
        except (TypeError, ValueError):
            conf = 0.0
        raw_box = d.get("bbox") if d.get("bbox") is not None else d.get("box")
        bbox = None
        box_pct = None
        if isinstance(raw_box, dict):
            # Unity sends {"l","t","w","h"} already as percent of the camera
            # frame — the unit the web console draws in. Pass it through
            # untouched; rescaling it anywhere in between only loses accuracy.
            try:
                box_pct = {k: float(raw_box[k]) for k in ("l", "t", "w", "h")}
            except (KeyError, TypeError, ValueError):
                box_pct = None
        elif isinstance(raw_box, (list, tuple)) and len(raw_box) >= 4:
            try:
                bbox = [float(v) for v in raw_box[:4]]
            except (TypeError, ValueError):
                bbox = None
        image = d.get("image_path") or d.get("image") or d.get("photo")
        try:
            ts = float(d.get("ts", d.get("timestamp", 0.0)))
        except (TypeError, ValueError):
            ts = 0.0
        return DetectionEvent(
            label=label.lower(), conf=conf, bbox=bbox, box_pct=box_pct,
            image_path=str(image) if image else None, ts=ts,
            recv_ts=time.time(), source=str(d.get("source", "")), raw=d,
        )


class DetectionListener:
    """UDP JSON ingest with arm/disarm gating and per-label cooldown."""

    def __init__(self, port: int = DEFAULT_PORT, host: str = "0.0.0.0",
                 cooldown: float = 3.0,
                 labels: Sequence[str] = ("person",),
                 min_conf: float = 0.0,
                 maxlen: int = 64) -> None:
        self.port = port
        self.host = host
        self.cooldown = cooldown
        self.labels = tuple(l.lower() for l in labels) if labels else ()
        self.min_conf = min_conf

        self._sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._queue: Deque[DetectionEvent] = collections.deque(maxlen=maxlen)
        self._armed = False
        self._room_name = ""
        self._last_seen: Dict[str, float] = {}

        self.received_count = 0     # every well-formed datagram
        self.ignored_count = 0      # arrived while disarmed (in transit)
        self.filtered_count = 0     # wrong label / below min_conf / cooldown
        self.accepted_count = 0

    # ------------------------------------------------------------- lifecycle
    def start(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self.host, self.port))
        self._sock.settimeout(0.2)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self._sock:
            self._sock.close()
            self._sock = None

    # ------------------------------------------------------------- gating
    def arm(self, room_name: str = "") -> None:
        """Start accepting detections (the drone entered a patrol room)."""
        with self._lock:
            self._armed = True
            self._room_name = room_name
            self._queue.clear()
            self._last_seen.clear()

    def disarm(self) -> None:
        with self._lock:
            self._armed = False
            self._room_name = ""

    @property
    def armed(self) -> bool:
        with self._lock:
            return self._armed

    # ------------------------------------------------------------- consumption
    def pop_all(self) -> List[DetectionEvent]:
        with self._lock:
            out = list(self._queue)
            self._queue.clear()
        return out

    def wait_for(self, timeout: float) -> Optional[DetectionEvent]:
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                if self._queue:
                    return self._queue.popleft()
            time.sleep(0.02)
        return None

    def stats(self) -> Dict[str, int]:
        return {"received": self.received_count, "accepted": self.accepted_count,
                "ignored_in_transit": self.ignored_count,
                "filtered": self.filtered_count}

    # ------------------------------------------------------------- rx thread
    def _loop(self) -> None:
        assert self._sock is not None
        while not self._stop.is_set():
            try:
                payload, _ = self._sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                data = json.loads(payload.decode("utf-8", errors="replace"))
            except json.JSONDecodeError:
                continue
            if not isinstance(data, dict):
                continue
            ev = DetectionEvent.from_payload(data)
            if ev is None:
                continue
            self.received_count += 1

            with self._lock:
                if not self._armed:
                    self.ignored_count += 1
                    continue
                if self.labels and ev.label not in self.labels:
                    self.filtered_count += 1
                    continue
                if ev.conf and ev.conf < self.min_conf:
                    self.filtered_count += 1
                    continue
                now = ev.recv_ts
                if now - self._last_seen.get(ev.label, -1e9) < self.cooldown:
                    self.filtered_count += 1
                    continue
                self._last_seen[ev.label] = now
                ev.room_name = self._room_name
                self._queue.append(ev)
                self.accepted_count += 1


# --------------------------------------------------------------- test emitter
def emit(host: str, port: int, label: str, conf: float,
         image: Optional[str], bbox: Optional[Sequence[float]],
         source: str = "manual") -> dict:
    """Send one detection event — the exact datagram the detector must send."""
    msg = {"label": label, "conf": conf, "ts": time.time(), "source": source}
    if image:
        msg["image_path"] = image
    if bbox:
        msg["bbox"] = list(bbox)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.sendto(json.dumps(msg).encode("utf-8"), (host, port))
    finally:
        sock.close()
    return msg


def _main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--emit", action="store_true",
                    help="Send one test detection instead of listening.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--label", default="person")
    ap.add_argument("--conf", type=float, default=0.9)
    ap.add_argument("--image", default=None)
    ap.add_argument("--bbox", type=float, nargs=4, default=None)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--interval", type=float, default=1.0)
    args = ap.parse_args()

    if args.emit:
        for i in range(args.repeat):
            msg = emit(args.host, args.port, args.label, args.conf,
                       args.image, args.bbox)
            print(f"-> {args.host}:{args.port}  {json.dumps(msg, ensure_ascii=False)}")
            if i + 1 < args.repeat:
                time.sleep(args.interval)
        return

    listener = DetectionListener(port=args.port)
    listener.start()
    listener.arm("test")
    print(f"listening on udp/{args.port} (armed) — Ctrl-C to stop")
    try:
        while True:
            for ev in listener.pop_all():
                print(f"[{ev.iso_time}] {ev.label} conf={ev.conf:.2f} "
                      f"image={ev.image_path}")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print(f"\n{listener.stats()}")
    finally:
        listener.close()


if __name__ == "__main__":
    _main()
