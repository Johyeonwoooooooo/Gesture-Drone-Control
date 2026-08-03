"""OpenAI-compatible LLM server — the GPU-box half of the split deployment.

Loads the intent model once and serves it over HTTP so that
`patrol/server.py --llm-url ...` can run somewhere else (typically the laptop
that runs Unity, next to the simulator and the 2D detector).

    python patrol/llm_serve.py --port 8000 --llm-device cuda:1

Routes (the subset `patrol/remote_llm.py` uses):

    GET  /v1/models             -> {"object":"list","data":[{"id": ...}]}
    POST /v1/chat/completions   -> {"choices":[{"message":{"content": ...}}]}
    GET  /health                -> {"status":"ok"}

Stdlib `http.server` on purpose: this runs in the existing `patrol` conda env
with nothing extra to install, and it only ever serves one REPL. If you
outgrow it, any OpenAI-compatible runtime (vLLM, Ollama, llama.cpp) speaks the
same wire format and `--llm-url` does not change — but install those in their
OWN env, since they pin their own torch and would disturb this one.

Generation is serialized behind a lock: one model on one GPU, and
`transformers.generate` is not safe to call concurrently on it.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

from patrol.llm_parser import LocalLLMParser  # noqa: E402

_LLM: Optional[LocalLLMParser] = None
_LOCK = threading.Lock()
_API_KEY: Optional[str] = None


def _split_messages(messages: List[dict]) -> Tuple[str, str]:
    """Flatten an OpenAI message list into the (system, user) pair the local
    parser wants. All system turns are joined; the last user turn wins — the
    patrol client only ever sends one of each, so this is just tolerance for
    other clients."""
    systems = [str(m.get("content") or "") for m in messages
               if m.get("role") == "system"]
    users = [str(m.get("content") or "") for m in messages
             if m.get("role") == "user"]
    return "\n".join(s for s in systems if s), users[-1] if users else ""


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "patrol-llm-serve/1"

    # ------------------------------------------------------------ plumbing --
    def _send(self, obj: dict, code: int = 200) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self) -> bool:
        if not _API_KEY:
            return True
        got = self.headers.get("Authorization", "")
        return got == f"Bearer {_API_KEY}"

    def log_message(self, fmt: str, *a) -> None:
        print(f"[llm-serve] {self.address_string()} - {fmt % a}", flush=True)

    # -------------------------------------------------------------- routes --
    def do_GET(self) -> None:
        path = self.path.rstrip("/") or "/"
        if path == "/health":
            self._send({"status": "ok", "model": _LLM.model_id if _LLM else ""})
        elif path in ("/v1/models", "/models"):
            if not self._authorized():
                self._send({"error": "unauthorized"}, 401)
                return
            self._send({"object": "list",
                        "data": [{"id": _LLM.model_id if _LLM else "",
                                  "object": "model"}]})
        else:
            self._send({"error": "not found", "path": self.path}, 404)

    def do_POST(self) -> None:
        path = self.path.rstrip("/")
        if path not in ("/v1/chat/completions", "/chat/completions"):
            self._send({"error": "not found", "path": self.path}, 404)
            return
        if not self._authorized():
            self._send({"error": "unauthorized"}, 401)
            return

        n = int(self.headers.get("Content-Length") or 0)
        try:
            req = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError as e:
            self._send({"error": f"invalid json: {e}"}, 400)
            return

        messages = req.get("messages")
        if not isinstance(messages, list) or not messages:
            self._send({"error": "messages must be a non-empty list"}, 400)
            return
        system, user = _split_messages(messages)
        if not user:
            self._send({"error": "no user message"}, 400)
            return

        # temperature/top_p are accepted and ignored: the local parser runs
        # greedy (do_sample=False) so intent parsing stays reproducible.
        max_tokens = int(req.get("max_tokens") or req.get("max_completion_tokens") or 256)

        t0 = time.time()
        try:
            with _LOCK:
                text = _LLM.generate(system, user, max_new_tokens=max_tokens)
        except Exception as e:  # a bad prompt must not kill the server
            print(f"[llm-serve] generate failed: {e!r}", flush=True)
            self._send({"error": f"generation failed: {e}"}, 500)
            return
        dt = time.time() - t0
        print(f"[llm-serve] {len(user)} chars -> {len(text)} chars in {dt:.2f}s",
              flush=True)

        self._send({
            "id": f"patrol-{int(t0 * 1000)}",
            "object": "chat.completion",
            "created": int(t0),
            "model": _LLM.model_id,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0,
                      "total_tokens": 0},
        })


def main() -> None:
    global _LLM, _API_KEY
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default="0.0.0.0",
                    help="0.0.0.0 so the laptop can reach it; 127.0.0.1 to "
                         "force everything through an ssh tunnel.")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--api-key", default=None,
                    help="If set, require `Authorization: Bearer <key>`.")
    ap.add_argument("--llm-model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--llm-device", default="cuda:1")
    ap.add_argument("--llm-dtype", default="float16",
                    choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--llm-device-map", default=None)
    args = ap.parse_args()

    _API_KEY = args.api_key
    device = args.llm_device if args.llm_device_map is None else "cuda:0"
    _LLM = LocalLLMParser(model_id=args.llm_model, device=device,
                          dtype=args.llm_dtype, device_map=args.llm_device_map)

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[llm-serve] {args.llm_model} on {_LLM.device}, "
          f"listening on {args.host}:{args.port}", flush=True)
    print(f"[llm-serve] client: python patrol/server.py --sim "
          f"--unity-host 127.0.0.1 --llm-url http://<this-host>:{args.port}/v1",
          flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[llm-serve] bye", flush=True)
        srv.shutdown()


if __name__ == "__main__":
    main()
