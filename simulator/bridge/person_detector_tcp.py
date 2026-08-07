#!/usr/bin/env python3
"""
TCP person detector for Unity patrol scans.

Unity's `PatrolPersonDetection.cs` captures the drone camera during a `scan`
and asks this process about each frame; what comes back decides whether it
reports a `detect` event to the pipeline. Nothing else talks to it.

Protocol (one connection per frame):
  request  = 4-byte big-endian length + JPEG bytes
  response = 4-byte big-endian length + UTF-8 JSON

A request whose payload starts with `__PATROL_SAVED_LOG__\n` is not an image —
it is Unity mirroring its saved-frame log into this terminal, which is where
whoever runs the detector is looking.

Run (own terminal, next to Unity):
  python simulator/bridge/person_detector_tcp.py --host 127.0.0.1 --port 9100

**This is a separate process on purpose.** ultralytics drags in torch, and
`patrol/` is kept torch-free (see CLAUDE.md) — the same split as `llm_server/`.
Nothing imports this file, so installing the model deps here does not put torch
in the pipeline's import graph:
  python -m pip install ultralytics

Without it running, every Unity request times out, the sweep still completes,
and the room simply reports no detections. `--mock-person` returns a detection
for every frame, for testing the wiring without a model.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import socket
import struct
import threading
import time
from dataclasses import dataclass
from typing import Any


MAX_IMAGE_BYTES = 8 * 1024 * 1024
PERSON_CLASS_ID = 0
SAVED_LOG_PREFIX = b"__PATROL_SAVED_LOG__\n"


def recv_exact(conn: socket.socket, count: int) -> bytes | None:
    chunks: list[bytes] = []
    remaining = count
    while remaining > 0:
        chunk = conn.recv(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_frame(conn: socket.socket) -> bytes | None:
    header = recv_exact(conn, 4)
    if header is None:
        return None
    (size,) = struct.unpack("!I", header)
    if size <= 0 or size > MAX_IMAGE_BYTES:
        raise ValueError(f"invalid image size: {size}")
    return recv_exact(conn, size)


def send_frame(conn: socket.socket, payload: bytes) -> None:
    conn.sendall(struct.pack("!I", len(payload)) + payload)


def handle_control_frame(conn: socket.socket, payload: bytes) -> bool:
    if not payload.startswith(SAVED_LOG_PREFIX):
        return False

    text = payload[len(SAVED_LOG_PREFIX):].decode("utf-8", errors="replace")
    for line in text.splitlines():
        if line.strip():
            print(line)
    send_frame(conn, json.dumps({"ok": True}).encode("utf-8"))
    return True


@dataclass
class Detection:
    label: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    cx: float
    cy: float


class PersonDetector:
    def __init__(
        self,
        model_name: str,
        confidence: float,
        imgsz: int,
        mock_person: bool = False,
    ) -> None:
        self.model_name = model_name
        self.confidence = confidence
        self.imgsz = imgsz
        self.mock_person = mock_person
        self.model: Any | None = None
        self.load_error = ""
        if mock_person:
            return
        try:
            from ultralytics import YOLO

            self.model = YOLO(model_name)
            self._warmup()
        except Exception as exc:
            self.load_error = (
                f"ultralytics YOLO unavailable ({exc}). "
                "Install with: python -m pip install ultralytics"
            )

    # The first predict() pays for lazy init — measured 1.85 s here against 32 ms
    # for every one after it. Unity gives up on a frame after responseTimeoutMs
    # (1500 ms), so without this the very first frame of the first room is always
    # lost. Spend it at startup instead, before anyone is listening.
    def _warmup(self) -> None:
        try:
            from PIL import Image

            self.model.predict(
                source=Image.new("RGB", (self.imgsz, self.imgsz), (0, 0, 0)),
                conf=self.confidence,
                classes=[PERSON_CLASS_ID],
                imgsz=self.imgsz,
                verbose=False,
            )
        except Exception as exc:
            # Warmup is an optimisation; a failure here must not stop the server.
            print(f"[person-detector] warmup skipped ({exc})")

    def detect(self, image_bytes: bytes) -> dict[str, Any]:
        if self.mock_person:
            width, height = 640, 360
            try:
                from PIL import Image

                image = Image.open(io.BytesIO(image_bytes))
                width, height = image.size
            except Exception:
                pass
            return {
                "ok": True,
                "person_detected": True,
                "best_confidence": 0.99,
                "detections": [
                    {
                        "label": "person",
                        "confidence": 0.99,
                        "x1": 0.0,
                        "y1": 0.0,
                        "x2": float(width),
                        "y2": float(height),
                        "cx": float(width) / 2.0,
                        "cy": float(height) / 2.0,
                    }
                ],
                "error": "",
            }

        if self.model is None:
            return {
                "ok": False,
                "person_detected": False,
                "best_confidence": 0.0,
                "detections": [],
                "error": self.load_error,
            }

        from PIL import Image

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        results = self.model.predict(
            source=image,
            conf=self.confidence,
            classes=[PERSON_CLASS_ID],
            imgsz=self.imgsz,
            verbose=False,
        )

        detections: list[Detection] = []
        best = 0.0
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                conf = float(box.conf[0])
                xyxy = [float(v) for v in box.xyxy[0]]
                cx = (xyxy[0] + xyxy[2]) / 2.0
                cy = (xyxy[1] + xyxy[3]) / 2.0
                best = max(best, conf)
                detections.append(
                    Detection(
                        label="person",
                        confidence=conf,
                        x1=xyxy[0],
                        y1=xyxy[1],
                        x2=xyxy[2],
                        y2=xyxy[3],
                        cx=cx,
                        cy=cy,
                    )
                )

        return {
            "ok": True,
            "person_detected": bool(detections),
            "best_confidence": best,
            "detections": [d.__dict__ for d in detections],
            "error": "",
            }


class SaveLimiter:
    def __init__(self, min_interval_sec: float) -> None:
        self.min_interval_sec = min_interval_sec
        self.last_saved_at = 0.0
        self.lock = threading.Lock()

    def should_save(self) -> bool:
        if self.min_interval_sec <= 0:
            return True
        now = time.time()
        with self.lock:
            if now - self.last_saved_at < self.min_interval_sec:
                return False
            self.last_saved_at = now
            return True


def save_image(save_dir: str, image: bytes, detections: list[dict[str, Any]]) -> tuple[str, str]:
    os.makedirs(save_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    stem = f"frame_{stamp}_{time.time_ns() % 1_000_000_000:09d}"
    path = os.path.join(save_dir, f"{stem}.jpg")
    with open(path, "wb") as f:
        f.write(image)

    annotated_path = ""
    if detections:
        annotated_path = os.path.join(save_dir, f"{stem}_annotated.jpg")
        save_annotated_image(annotated_path, image, detections)
    return path, annotated_path


def save_annotated_image(path: str, image: bytes, detections: list[dict[str, Any]]) -> None:
    from PIL import Image, ImageDraw

    annotated = Image.open(io.BytesIO(image)).convert("RGB")
    draw = ImageDraw.Draw(annotated)
    for detection in detections:
        x1 = float(detection["x1"])
        y1 = float(detection["y1"])
        x2 = float(detection["x2"])
        y2 = float(detection["y2"])
        cx = float(detection.get("cx", (x1 + x2) / 2.0))
        cy = float(detection.get("cy", (y1 + y2) / 2.0))
        conf = float(detection["confidence"])

        draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 0), width=4)
        draw.line((cx - 12, cy, cx + 12, cy), fill=(255, 0, 0), width=4)
        draw.line((cx, cy - 12, cx, cy + 12), fill=(255, 0, 0), width=4)
        draw.text((x1, max(0.0, y1 - 14.0)), f"person {conf:.2f} center=({cx:.1f},{cy:.1f})", fill=(0, 255, 0))

    annotated.save(path, quality=90)


def log_saved_detections(addr: tuple[str, int], response: dict[str, Any]) -> None:
    print(
        f"[person-detector] saved person frame from {addr[0]} "
        f"confidence={response.get('best_confidence', 0):.2f}"
    )
    for detection in response.get("detections", []):
        print(
            "[person-detector] bbox="
            f"({detection['x1']:.1f}, {detection['y1']:.1f}, {detection['x2']:.1f}, {detection['y2']:.1f}) "
            f"center=({detection['cx']:.1f}, {detection['cy']:.1f}) "
            f"confidence={detection['confidence']:.2f}"
        )


def handle_client(
    conn: socket.socket,
    addr: tuple[str, int],
    detector: PersonDetector,
    save_dir: str,
    save_limiter: SaveLimiter,
    save_all: bool,
) -> None:
    with conn:
        try:
            image = recv_frame(conn)
            if image is None:
                return
            if handle_control_frame(conn, image):
                return
            started = time.time()
            response = detector.detect(image)
            response["elapsed_ms"] = round((time.time() - started) * 1000.0, 1)
            response["saved_path"] = ""
            response["saved_annotated_path"] = ""
            if save_dir and (save_all or response.get("person_detected")) and save_limiter.should_save():
                response["saved_path"], response["saved_annotated_path"] = save_image(
                    save_dir,
                    image,
                    response.get("detections", []),
                )
                if response.get("person_detected"):
                    log_saved_detections(addr, response)
            send_frame(conn, json.dumps(response).encode("utf-8"))
        except Exception as exc:
            payload = {
                "ok": False,
                "person_detected": False,
                "best_confidence": 0.0,
                "detections": [],
                "error": str(exc),
            }
            try:
                send_frame(conn, json.dumps(payload).encode("utf-8"))
            except OSError:
                pass


def serve(
    host: str,
    port: int,
    model: str,
    confidence: float,
    imgsz: int,
    mock_person: bool,
    save_dir: str,
    save_min_interval: float,
    save_all: bool,
) -> None:
    detector = PersonDetector(model, confidence, imgsz, mock_person=mock_person)
    save_limiter = SaveLimiter(save_min_interval)
    if detector.mock_person:
        print("[person-detector] mock mode enabled - every frame returns person_detected=true")
    elif detector.load_error:
        print(f"[person-detector] {detector.load_error}")
    else:
        print(f"[person-detector] loaded {model}")

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    sock.listen(8)
    print(f"[person-detector] listening on {host}:{port}")
    if save_dir:
        print(f"[person-detector] saving frames to {save_dir}")
        print(f"[person-detector] save all frames = {save_all}")
        print(f"[person-detector] save min interval = {save_min_interval:.1f}s")
    try:
        while True:
            conn, addr = sock.accept()
            threading.Thread(
                target=handle_client,
                args=(conn, addr, detector, save_dir, save_limiter, save_all),
                daemon=True,
            ).start()
    finally:
        sock.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9100)
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--confidence", type=float, default=0.1)
    parser.add_argument("--imgsz", type=int, default=768,
                        help="YOLO inference image size; larger helps small/dark Unity people")
    parser.add_argument("--mock-person", action="store_true",
                        help="integration test mode: every frame returns a person detection")
    parser.add_argument("--save-dir", default="",
                        help="optional directory for saving JPEG frames received from Unity")
    parser.add_argument("--save-min-interval", type=float, default=3.0,
                        help="minimum seconds between saved person-detection frames")
    parser.add_argument("--save-all", action="store_true",
                        help="debug mode: save received frames even when no person is detected")
    args = parser.parse_args()
    serve(
        args.host,
        args.port,
        args.model,
        args.confidence,
        args.imgsz,
        args.mock_person,
        args.save_dir,
        args.save_min_interval,
        args.save_all,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
