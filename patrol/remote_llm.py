"""LLM client — talks to an OpenAI-compatible server instead of loading a model.

The whole LLM surface of this side is `generate(system, user) -> str`. Both
callers (`patrol_intent` for the patrol area, `patrol_report` for the report
prose) sit on top of it, so the model can live on another machine and nothing
else notices.

    llm = RemoteLLMParser("http://166.104.223.32:8000/v1")
    llm.generate("Answer in JSON.", "2층 전부 순찰해줘")

Deliberately **stdlib only** (urllib) — the PC running Unity should need
nothing beyond numpy, so don't reach for `requests` or the `openai` SDK here.
The server can be `llm_server/serve.py` or any OpenAI-compatible runtime
(vLLM, Ollama, llama.cpp); the wire format is the same.

Self-test (needs data/ for the room directory):
    python patrol/remote_llm.py --llm-url http://<host>:8000/v1 "2층 전부 순찰해줘"
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

DEFAULT_TIMEOUT = 60.0
DEFAULT_RETRIES = 2


class RemoteLLMError(RuntimeError):
    pass


class RemoteLLMParser:
    def __init__(
        self,
        base_url: str,
        model_id: str = "Qwen/Qwen2.5-3B-Instruct",
        api_key: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
        retries: int = DEFAULT_RETRIES,
    ):
        self.base_url = base_url.rstrip("/")
        self.model_id = model_id
        self.api_key = api_key or None
        self.timeout = timeout
        self.retries = max(0, retries)
        self.endpoint = _chat_endpoint(self.base_url)
        print(f"[llm] remote {self.endpoint} (model={model_id}, "
              f"timeout={timeout:g}s)")

    # ---------------------------------------------------------------- wire --
    def _post(self, url: str, payload: dict) -> dict:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        last: Optional[Exception] = None
        for attempt in range(self.retries + 1):
            req = urllib.request.Request(url, data=body, headers=headers,
                                         method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "replace")[:400]
                # 4xx is our fault (bad model name, bad payload) — retrying
                # just repeats it. 5xx can be a transient server hiccup.
                if e.code < 500:
                    raise RemoteLLMError(
                        f"LLM server returned HTTP {e.code} for {url}: {detail}"
                    ) from e
                last = RemoteLLMError(f"HTTP {e.code}: {detail}")
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                last = RemoteLLMError(f"cannot reach {url}: {e}")
            if attempt < self.retries:
                time.sleep(0.5 * (attempt + 1))
        raise RemoteLLMError(
            f"{last}\nIs the LLM server up on that host? Start it with\n"
            f"    python llm_server/serve.py --port <port> --llm-device cuda:1\n"
            f"and check --llm-url."
        )

    # -------------------------------------------------------------- public --
    def generate(self, system: str, user: str, max_new_tokens: int = 256) -> str:
        """One system+user turn — the same contract `llm_server` implements.

        temperature 0 mirrors the server's `do_sample=False`, so the answer does
        not depend on which side of the split you look from.
        """
        data = self._post(self.endpoint, {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user.strip()},
            ],
            "max_tokens": max_new_tokens,
            "temperature": 0.0,
        })
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as e:
            raise RemoteLLMError(
                f"unexpected response shape from {self.endpoint}: "
                f"{json.dumps(data)[:400]}"
            ) from e

    def ping(self) -> List[str]:
        """Fail fast at startup instead of on the user's first query.

        Returns the model ids the server advertises (empty list if it has no
        /models route — that is fine, plenty of servers skip it).
        """
        url = f"{_v1_root(self.base_url)}/models"
        req = urllib.request.Request(url, method="GET")
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")
        try:
            with urllib.request.urlopen(req, timeout=min(self.timeout, 10.0)) as r:
                data = json.loads(r.read().decode("utf-8"))
            return [m.get("id", "") for m in data.get("data", [])]
        except urllib.error.HTTPError:
            return []
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise RemoteLLMError(
                f"cannot reach the LLM server at {url}: {e}\n"
                f"Check that it is running and that the port is open from here "
                f"(curl {url})."
            ) from e


def _v1_root(base_url: str) -> str:
    u = base_url.rstrip("/")
    if u.endswith("/chat/completions"):
        u = u[: -len("/chat/completions")]
    return u if u.endswith("/v1") else f"{u}/v1"


def _chat_endpoint(base_url: str) -> str:
    """Accept `http://h:8000`, `http://h:8000/v1`, or the full path."""
    u = base_url.rstrip("/")
    if u.endswith("/chat/completions"):
        return u
    return f"{_v1_root(u)}/chat/completions"


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("query", nargs="+")
    ap.add_argument("--llm-url", required=True)
    ap.add_argument("--llm-model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--llm-api-key", default=None)
    ap.add_argument("--llm-timeout", type=float, default=DEFAULT_TIMEOUT)
    ap.add_argument("--data-dir",
                    default=str(Path(__file__).resolve().parents[1]
                                / "data" / "final_npy"))
    args = ap.parse_args()

    llm = RemoteLLMParser(args.llm_url, args.llm_model,
                          api_key=args.llm_api_key, timeout=args.llm_timeout)
    served = llm.ping()
    print(f"[llm] server ok, models={served or '(no /models route)'}")

    from patrol import patrol_intent, room_index
    from patrol.litept_backend import LitePTBackend
    rooms = room_index.build_room_index(LitePTBackend(args.data_dir), None)

    t0 = time.time()
    intent = patrol_intent.parse_patrol(llm, " ".join(args.query), rooms)
    picked, why = patrol_intent.resolve_rooms(intent, rooms, " ".join(args.query))
    print(f"[llm] parsed in {time.time() - t0:.2f}s")
    print(json.dumps(intent.__dict__, ensure_ascii=False, indent=2))
    print(f"[rooms] {why}: " + ", ".join(r.display for r in picked))


if __name__ == "__main__":
    main()
