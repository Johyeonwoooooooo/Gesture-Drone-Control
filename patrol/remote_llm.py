"""LLM client — talks to an OpenAI-compatible server instead of loading a model.

Drop-in for `LocalLLMParser`: same `generate` / `parse` surface, so
`server.py`, `patrol_intent.py` and `patrol_report.py` cannot tell the
difference. Lets the laptop that runs Unity also run the whole patrol
pipeline, with only the LLM left on the GPU box.

    llm = RemoteLLMParser("http://166.104.223.32:8000/v1", "Qwen/Qwen2.5-3B-Instruct")
    llm.parse("거실 소파 찾아줘")

Deliberately **stdlib only** (urllib) — the laptop side of the split should
need nothing beyond numpy, so don't reach for `requests` or the `openai` SDK
here. The server can be `patrol/llm_serve.py` or any OpenAI-compatible
runtime (vLLM, Ollama, llama.cpp); the wire format is the same.

Self-test:
    python patrol/remote_llm.py --llm-url http://<host>:8000/v1 "거실 소파 찾아줘"
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

from patrol.llm_base import BaseLLMParser  # noqa: E402

DEFAULT_TIMEOUT = 60.0
DEFAULT_RETRIES = 2


class RemoteLLMError(RuntimeError):
    pass


class RemoteLLMParser(BaseLLMParser):
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
            f"    python patrol/llm_serve.py --port <port> --llm-device cuda:1\n"
            f"and check --llm-url."
        )

    # -------------------------------------------------------------- public --
    def generate(self, system: str, user: str, max_new_tokens: int = 256) -> str:
        """Same contract as LocalLLMParser.generate — one system+user turn.

        temperature 0 mirrors the local `do_sample=False`, so a query gives the
        same answer whichever side of the split it runs on.
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
    args = ap.parse_args()

    llm = RemoteLLMParser(args.llm_url, args.llm_model,
                          api_key=args.llm_api_key, timeout=args.llm_timeout)
    served = llm.ping()
    print(f"[llm] server ok, models={served or '(no /models route)'}")

    t0 = time.time()
    intent = llm.parse(" ".join(args.query))
    print(f"[llm] parsed in {time.time() - t0:.2f}s")
    print(json.dumps(intent.__dict__, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
