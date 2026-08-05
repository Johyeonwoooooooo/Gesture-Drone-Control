"""LLM client — talks to an OpenAI-compatible server instead of loading a model.

The whole LLM surface of this side is `generate(system, user) -> str`. Both
callers (`patrol_intent` for the patrol area, `patrol_report` for the report
prose) sit on top of it, so the model can live on another machine and nothing
else notices.

    llm = RemoteLLMParser("http://127.0.0.1:11434/v1", "qwen2.5:3b-instruct")
    llm.generate("Answer in JSON.", "2층 전부 순찰해줘")

Deliberately **stdlib only** (urllib) — the PC running Unity should need
nothing beyond numpy, so don't reach for `requests` or the `openai` SDK here.
The server can be Ollama (the default), `llm_server/serve.py`, vLLM, or
llama.cpp; the wire format is the same and nothing else in the repo notices.

Self-test (needs data/ for the room directory):
    python patrol/remote_llm.py "2층 전부 순찰해줘"                    # 로컬 Ollama
    python patrol/remote_llm.py --llm-url http://<host>:8000/v1 ...   # GPU 서버
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

DEFAULT_TIMEOUT = 60.0
DEFAULT_RETRIES = 2

# 기본은 **이 PC의 Ollama** 다. 학교 GPU 서버는 교내망 밖에서 막혀 있어
# (166.104.223.32:8000/22 둘 다 filtered) VPN 없이는 못 붙는데, 이 파이프라인의
# LLM 호출은 한 순찰에 2회뿐이고 3B면 노트북에서 충분히 돈다. 서버를 쓰려면
# --llm-url http://<서버>:8000/v1 --llm-model Qwen/Qwen2.5-3B-Instruct 로 덮으면
# 된다 — 프로토콜이 같아서 그게 전부다.
DEFAULT_LLM_URL = "http://127.0.0.1:11434/v1"
DEFAULT_LLM_MODEL = "qwen2.5:3b-instruct"

# 학교 GPU 서버(`llm_server/serve.py`). 교내망 밖에서는 8000·22 둘 다 막혀 있어
# **한양대 SSL VPN 을 켜야** 닿는다. 팀원이 주소를 외울 필요 없게 별칭을 뒀다:
#     python api_server.py --llm-url gpu          (+ 토큰은 아래 환경변수)
# 모델을 따로 안 주면 서버 쪽 모델 이름으로 같이 바뀐다 — Ollama 태그와 다르다.
GPU_LLM_URL = "http://166.104.223.32:8000/v1"
GPU_LLM_MODEL = "Qwen/Qwen2.5-3B-Instruct"
API_KEY_ENV = "PATROL_LLM_API_KEY"     # 토큰을 명령줄에 안 남기려는 용도


def resolve_llm(base_url: Optional[str], model_id: str,
                api_key: Optional[str]) -> tuple[Optional[str], str, Optional[str]]:
    """`--llm-url gpu` 별칭과 토큰 환경변수를 실제 값으로 편다.

    두 진입점(api_server.py / patrol/server.py)이 같은 규칙을 쓰도록 여기 둔다.
    """
    if base_url == "gpu":
        base_url = GPU_LLM_URL
        if model_id == DEFAULT_LLM_MODEL:   # 사용자가 따로 안 준 경우만
            model_id = GPU_LLM_MODEL
    if not api_key:
        api_key = os.environ.get(API_KEY_ENV) or None
    return base_url, model_id, api_key


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
            f"{last}\nLLM 서버가 안 떠 있습니다. 기본값은 이 PC의 Ollama 입니다:\n"
            f"    ollama serve  (+ ollama pull {DEFAULT_LLM_MODEL})\n"
            f"GPU 서버를 쓰려면 --llm-url http://<서버>:8000/v1 로 덮으세요.\n"
            f"LLM 없이 나머지만 돌리려면 --llm-url \"\" (오프라인 모드)."
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
            # 학교 서버는 교내망 밖에서 아예 필터링돼 있어 증상이 '느림'이 아니라
            # '무응답'이다. 팀원이 제일 자주 밟는 곳이라 여기서 VPN 을 짚어준다.
            hint = ("\n한양대 SSL VPN 이 켜져 있는지 확인하세요 — 교내망 밖에서는 "
                    "이 서버의 8000·22 가 둘 다 막혀 있습니다."
                    if url.startswith(GPU_LLM_URL) else "")
            raise RemoteLLMError(
                f"cannot reach the LLM server at {url}: {e}\n"
                f"Check that it is running and that the port is open from here "
                f"(curl {url}).{hint}"
            ) from e


class OfflineLLMParser:
    """No model at all — the same `generate()` contract, answering nothing.

    For running the rest of the pipeline without the GPU box (web console,
    planner, mission loop, report). Every caller already survives an unusable
    answer, which is what makes this cheap: `patrol_intent._extract_json("")`
    degrades to `{}` and the five text-based resolution tiers carry the query on
    aliases/층/방 종류 alone, and `patrol_report._llm_summary` falls back to its
    templated sentence. So the cost of going offline is exactly tier 1 of intent
    resolution (the room ids the model itself proposes) and the report prose —
    a query like "2층 전부 순찰해줘" or "현우방 확인해줘" still resolves.
    """

    def __init__(self, model_id: str = "(offline)", reason: str = "") -> None:
        self.model_id = model_id
        self.reason = reason
        print(f"[llm] offline{f' ({reason})' if reason else ''} — "
              f"별칭/층/방 종류 키워드로만 구역을 찾습니다")

    def generate(self, system: str, user: str, max_new_tokens: int = 256) -> str:
        return ""

    def ping(self) -> List[str]:
        return []


def make_llm(base_url: Optional[str], model_id: str = DEFAULT_LLM_MODEL,
             api_key: Optional[str] = None, timeout: float = DEFAULT_TIMEOUT,
             reason: str = '--llm-url ""'):
    """`RemoteLLMParser` if there is a URL, `OfflineLLMParser` if there isn't.

    Empty string counts as "no URL" — that is how `--llm-url ""` selects
    offline mode now that the flag defaults to the local Ollama endpoint.
    """
    if base_url:
        return RemoteLLMParser(base_url, model_id, api_key=api_key,
                               timeout=timeout)
    return OfflineLLMParser(reason=reason)


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
    ap.add_argument("--llm-url", default=DEFAULT_LLM_URL)
    ap.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
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
