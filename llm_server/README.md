# llm_server — GPU 서버에 남는 유일한 조각

순찰 파이프라인에서 GPU가 필요한 건 **의도 파서 하나뿐**이다. 3D 인식은 사전계산
결과를 읽기만 하고, 경로 계획은 numpy다. 그래서 그 하나만 여기 남기고 나머지는
전부 Unity가 도는 PC로 내렸다. 이 폴더가 서버에서 도는 전부다.

```bash
conda activate patrol
pip install -r llm_server/requirements.txt
python llm_server/serve.py --port 8000 --llm-device cuda:1 --api-key <토큰>
```

클라이언트(로컬 PC)는 `--llm-url http://<이 서버>:8000/v1` 로 붙는다.

## 규칙: 이 폴더는 `patrol/` 을 import하지 않는다

프롬프트도, 방 목록도, JSON 스키마도 전부 클라이언트(`patrol/llm_base.py`)에 있고
여기로는 완성된 system/user 텍스트가 넘어온다. 서버는 모델만 돌린다.

덕분에 **이 폴더만 다른 GPU 박스에 복사해도 그대로 뜬다.** 편의를 위해 여기서
`patrol` 을 import하고 싶어질 때가 오는데, 그 순간 이 성질이 깨진다. 하지 말 것.

| 파일 | 역할 |
|---|---|
| `serve.py` | OpenAI 호환 HTTP 서버 (stdlib `http.server`). 생성은 락으로 직렬화 |
| `local_llm.py` | 모델 로드 + `generate(system, user)`. **저장소에서 torch를 import하는 유일한 파일** |
| `requirements.txt` | torch / transformers / accelerate |

## 라우트

```
GET  /health              -> {"status":"ok","model":"..."}
GET  /v1/models           -> {"object":"list","data":[{"id":"..."}]}
POST /v1/chat/completions -> {"choices":[{"message":{"content":"..."}}]}
```

`temperature` / `top_p` 는 받되 무시한다 — 의도 파싱이 재현 가능해야 해서 항상
greedy(`do_sample=False`)로 돈다. 클라이언트도 temperature 0을 보낸다.

## 상주시키기

터미널에 묶일 이유가 없다.

```bash
nohup python llm_server/serve.py --port 8000 --llm-device cuda:1 \
    > ~/llm_serve.log 2>&1 &
# 또는
tmux new -d -s llm 'conda activate patrol && python llm_server/serve.py --port 8000 --llm-device cuda:1'

curl http://<서버>:8000/health          # 살아있나
pkill -f llm_server/serve.py            # 내리기
```

모델을 GPU에 올린 채 대기한다(3B fp16 기준 6 GB 남짓). 안 쓸 땐 내려도 되고,
다시 띄우는 데 30초쯤 걸린다.

## 보안

`--host 0.0.0.0` 이 기본이라 그대로 두면 포트가 열린 사람 누구나 GPU를 쓸 수 있다.

- **토큰**: `--api-key <문자열>` → 클라이언트는 `--llm-api-key <같은 문자열>`
- **방화벽**: `sudo ufw allow 8000/tcp`
- **포트를 안 열고 싶으면**: `--host 127.0.0.1` 로 띄우고 로컬에서 터널을 판다
  ```bash
  ssh -N -L 8000:localhost:8000 <계정>@<서버>
  # 그러면 클라이언트는 --llm-url http://127.0.0.1:8000/v1
  ```

## 더 빨라져야 하면 — vLLM

프로토콜이 OpenAI 호환이라 **백엔드만 갈아끼우면 클라이언트는 한 줄도 안 바뀐다.**

```bash
conda create -n vllm python=3.10 -y && conda activate vllm && pip install vllm
vllm serve Qwen/Qwen2.5-3B-Instruct --port 8000
```

반드시 **별도 env** 에. vLLM은 자기 torch 버전을 고정하므로 `patrol` env에 얹으면
망가진다. 이 워크로드는 쿼리당 호출이 2~3회뿐이라 vLLM의 연속 배칭은 거의 쓸 일이
없고, 체감되는 이득은 단일 요청 지연이다.

Apple Silicon 맥이면 서버 없이 로컬에서 Ollama로 대신할 수도 있다 —
`ollama serve` 후 `--llm-url http://127.0.0.1:11434/v1 --llm-model qwen2.5:3b-instruct`.
