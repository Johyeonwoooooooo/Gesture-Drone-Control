# 순찰 드론 시뮬레이터: 구역 지정 → 비행 → 360° 스캔 → 보고서

평면도에서 순찰할 구역을 고르거나 자연어로 말하면 → LLM이 의도를 분석하고 →
사전계산된 3D 디텍션에서 방을 해석한 뒤 → 드론이 Unity 시뮬레이터에서 경로를
날며 방마다 360° 스캔하고 → 사람을 찾으면 알리고 → 순찰 보고서가 나온다.

명령은 한 종류다 — **어디를 순찰할지** 고르는 것. `현우방만 탐색해줘`,
`2층 전부 순찰해줘`, `집 전체 돌면서 사람 있는지 확인해줘` 처럼.

구성 요소:

- **탐색 백엔드**: **LitePT** — `data/final_npy/detections.json` (ScanNet-20 인스턴스,
  건물 00809). GPU 추론·CLIP 캐시 불필요, 읽기만 한다.
- **시뮬레이터**: Unity 6 Tello 시뮬레이터. 3인칭(이동방향 뒤) / 1인칭 카메라, C키 전환.
  **360° 스캔과 사람 탐지(YOLO)를 Unity가 직접 한다.**
- **웹 콘솔**: HAUNTED OPS (`web/`). 평면도에서 구역 선택 → 브리핑 → 작전 화면 →
  보고서. 붙는 규격은 **`API.md`**. 원본은 팀원의 `origin/hyeonwoo` 이고 여기엔
  API 배선을 얹어 들어와 있다 (§10).

## 어디서 무엇이 도는가

**GPU 서버에는 LLM만 남는다.** 파이프라인에서 GPU가 필요한 건 의도 파서
하나뿐이고(3D 인식은 사전계산 결과를 읽기만 한다), 저장소 전체에서 torch를
import하는 파일은 `llm_server/local_llm.py` 하나다. 나머지는 numpy만으로 돈다.

```
[GPU 서버 (Linux)]                    [로컬 PC (Windows/macOS)]
llm_server/serve.py                    브라우저 — 웹 콘솔 (web/)
 Qwen2.5-3B                              ↕ HTTP :8123
 /v1/chat/completions  ◀── TCP 8000 ── api_server.py  (patrol/ 두뇌)
 (torch는 여기만)                          ↕ UDP localhost 9000 / 9002
                                        Unity 시뮬레이터 (Play 중)
```

Unity와 파이프라인이 같은 PC라 **UDP가 전부 localhost가 되고 relay가 필요
없다.** 망을 타는 건 로컬→서버 TCP 하나뿐이고, 그것도 순찰 한 번에 **LLM 호출
2회**(구역 해석 · 보고서 문장)가 전부다. NAT는 나가는 연결을 막지 않는다
(실측: 노트북 `10.100.130.17` → 서버 `166.104.223.32` TCP 8000/8080/8443 도달).

| 방향 | 포트 | 내용 |
|---|---|---|
| 로컬 → 서버 | TCP **8000** | LLM `/v1/chat/completions` (OpenAI 호환) |
| 브라우저 → API | HTTP **8123** | 웹 콘솔 ↔ `api_server.py` (`API.md`). `--port` 로 정한다 |
| API → Unity | UDP **9000** | `command`/`takeoff`/`land`/`rc`/`setpos`/`msg`/`light`/`scan`/`scan_stop` |
| Unity → API | (9000 응답) | 각 명령에 `"ok"` (API는 9001 바인딩) |
| Unity → API | UDP **9002** | 상태 JSON 20 Hz + 이벤트 `scan_started`/`scan_done`/`detect` |
| (선택) 외부 디텍터 → API | UDP **9004** | 별도 프로세스를 쓸 때만 (`docs/patrol-agent.md`) |

> **API 서버 포트는 8000이 기본이지만 그대로 쓰지 않기를 권한다.** 8000·8080은
> 로컬에서 이미 쓰이고 있을 확률이 높다 — 특히 **VS Code Remote-SSH**는 서버에서
> 열린 포트를 감지해 같은 번호로 맥/윈도우에 자동 포워딩한다. 그러면
> `localhost:8000` 이 로컬 API가 아니라 **GPU 서버로 가는데도 겉보기엔 잘 도는
> 것처럼 보인다.** 이 문서는 그래서 `--port 8123` 을 쓴다. 겹치면
> `lsof -nP -iTCP:8123 -sTCP:LISTEN` 으로 확인하고 다른 번호를 고르면 된다.

---

## 0. 실행 체크리스트 (⭐ 여기부터)

### A. 최초 1회만 — 설치 & 데이터 & 씬

| 위치 | 할 일 | 참고 |
|---|---|---|
| 서버 | `git clone` → conda 환경 + `llm_server/requirements.txt` | §1 |
| 로컬 | `git clone` → 파이썬 환경 + `requirements.txt` (**torch 불필요**) | §2 |
| 로컬 | `data/final_npy/` 필요한 부분만 내려받기 (약 90 MB) | §2 |
| 로컬 | Unity Hub + 에디터 `6000.3.12f1`, `simulator/tello_simulator` 열기 | §3 |
| 로컬 | **test.unity** 에 00809(Qpor) 배치 확인 → 콜라이더 → 저장 | §8 |
| (선택) | 씬을 새로 배치했으면 좌표 캘리브레이션 재실행 | §8 |

웹 콘솔(`web/`)과 집 모델(`Qpor2mEya8F.glb`, 67 MB)은 **git에 들어 있다** —
클론하면 같이 온다. 따로 받아야 하는 건 `data/final_npy/` 뿐이다.

### B. 매번 다시 실행

```
[서버]  ① python llm_server/serve.py --port 8000 --llm-device cuda:1 --api-key <토큰>
        (한 번 띄우면 계속 켜둔다. nohup/tmux 로 떼어놔도 된다 — §1)
        → curl http://127.0.0.1:8000/health 가 답할 때까지 30초쯤 기다린다

[로컬]  ② Unity ▶ Play → Console "[Tello] UDP server listening on 9000"(초록)
        ③ python simulator/bridge/smoke.py --unity-host 127.0.0.1      # 'ok' 게이트
        ④ python api_server.py --port 8123 \
              --llm-url http://<서버IP>:8000/v1 --llm-api-key <토큰>
        ⑤ 브라우저 http://localhost:8123
```

**게이트가 둘이다. 하나씩 통과시키고 넘어가야 원인이 어디인지 안다.**

- **③ `-> 'ok'`** 가 안 나오면 무조건 ②(Unity 9000) 문제다. 뒤로 가라.
- **④의 `[llm] server ok`** 가 안 나오면 ①(서버) 또는 방화벽 문제다.
  로컬에서 `curl http://<서버IP>:8000/health` 로 갈라볼 것.

④가 뜨면 이 세 줄이 순서대로 나온다:

```
[llm] server ok, models=[...]                 ← 서버까지 뚫림
[sim] Unity 127.0.0.1:9000 -> 'ok'            ← Unity까지 뚫림
[api] http://127.0.0.1:8123   (규격: /docs)
```

> 웹 콘솔 없이 터미널로 쓰려면 ④ 대신
> `python patrol/server.py --sim --unity-host 127.0.0.1 --llm-url ...` (§5).
> **둘을 동시에 띄우면 안 된다** — UDP 브리지를 서로 뺏는다.

> **지금은 탐지가 안 뜬다.** Unity 쪽 `PatrolPersonDetection.cs` 가 이 브랜치에
> 없어서 `scan` verb를 모르고, 파이프라인이 rc 회전 폴백을 탄다. 비행과 360°
> 회전은 정상이지만 `detect` 이벤트가 하나도 안 온다 → 콘솔의 탐지 목록·경보·
> 보고서 사진이 전부 빈다. 배선만 확인하려면 §4 「Unity 없이 스텁으로」가
> 탐지까지 흉내낸다.

---

## 1. GPU 서버 준비

서버에서 도는 건 `llm_server/` 하나다. 자세한 건 **`llm_server/README.md`**.

```bash
git clone <repo> && cd Gesture-Drone-Control
conda create -n patrol python=3.10 -y && conda activate patrol
pip install -r llm_server/requirements.txt      # torch / transformers / accelerate
python llm_server/serve.py --port 8000 --llm-device cuda:1 --api-key <토큰>
```

✅ `[llm-serve] Qwen/Qwen2.5-3B-Instruct on cuda:1, listening on 0.0.0.0:8000`

**모델 로딩에 30초쯤 걸린다.** 바로 다음 단계로 넘어가면 연결 실패로 보인다.
두 곳에서 확인하고 넘어갈 것:

```bash
curl http://127.0.0.1:8000/health         # [서버에서] {"status":"ok","model":"..."}
curl http://<서버IP>:8000/health          # [로컬에서] 여기가 뚫려야 파이프라인이 붙는다
```

로컬에서만 실패하면 방화벽이다 → 서버에서 `sudo ufw allow 8000/tcp`.
`--llm-device` 는 비어 있는 GPU면 아무거나 (`nvidia-smi` 로 확인).

**터미널을 붙들고 있을 필요 없다.**

```bash
nohup python llm_server/serve.py --port 8000 --llm-device cuda:1 \
    --api-key <토큰> > ~/llm_serve.log 2>&1 &
# 또는
tmux new -d -s llm 'conda activate patrol && python llm_server/serve.py --port 8000 --llm-device cuda:1'
pkill -f llm_server/serve.py               # 내리기
```

모델을 GPU에 올린 채 대기한다(3B fp16 기준 6 GB 남짓). 안 쓸 땐 내려도 되고,
다시 띄우는 데 30초쯤 걸린다.

**보안** — `--host 0.0.0.0` 이 기본이라 그대로 두면 포트가 열린 사람 누구나 GPU를
쓴다. `--api-key <토큰>` 을 걸고 방화벽은 `sudo ufw allow 8000/tcp`. 포트를 아예
안 열고 싶으면 `--host 127.0.0.1` 로 띄우고 로컬에서 터널을 판다:

```bash
ssh -N -L 8000:localhost:8000 <계정>@<서버IP>   # 그러면 --llm-url http://127.0.0.1:8000/v1
```

**서버를 아예 안 쓰는 방법도 있다.** `remote_llm.py` 는 OpenAI 호환 프로토콜만
알지 상대가 무엇인지는 모른다. Apple Silicon 맥이면 로컬에서 Ollama로 대신할 수
있다: `ollama pull qwen2.5:3b-instruct && ollama serve` 후
`--llm-url http://127.0.0.1:11434/v1 --llm-model qwen2.5:3b-instruct`.

## 2. 로컬 PC 준비 (파이썬 + 데이터)

**torch도 transformers도 필요 없다.**

```bash
git clone <repo> && cd Gesture-Drone-Control
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt                        # numpy / fastapi / uvicorn (+scipy/pillow)
```

**데이터** — git에 없다(용량). `data/final_npy/` 305 MB 중 파이썬이 실제로 읽는
건 셋뿐이라 그것만 받으면 된다 (`color.npy`/`normal.npy` 는 어느 코드도 안 읽는다).

```bash
# macOS / Linux
rsync -av --include='*/' \
  --include='detections.json' --include='coord.npy' --include='centers.pkl' \
  --exclude='*' \
  <계정>@<서버IP>:/data1/workspaces/jgshin22/Gesture-Drone-Control/data/final_npy/ \
  ./data/final_npy/
```

```powershell
# Windows (WSL 없이) — scp 로 폴더째 받은 뒤 안 쓰는 것만 지운다
scp -r <계정>@<서버IP>:/data1/.../data/final_npy ./data/
Get-ChildItem -Recurse .\data\final_npy -Include color.npy,normal.npy | Remove-Item
```

없으면 `minyeong-3d` 브랜치 `litept_indoor/` (`infer_centers.py` → `export_json.py`)로
생성한다. 이 브랜치에는 생성 파이프라인이 없다 — 결과만 읽는다.

**나머지는 클론에 다 들어 있다.** Unity 씬의 집 모델 `Qpor2mEya8F.glb`(67 MB)와
웹 콘솔 `web/` 은 git에 커밋돼 있어서 따로 받을 필요가 없다.

## 3. Unity — 설치·씬은 (최초 1회), Play는 (매번)

1. [Unity Hub](https://unity.com/download) 설치 → Installs → **`6000.3.12f1`** (Unity 6).
   Apple Silicon 맥이면 Silicon 에디터.
2. Projects → Open → 레포의 `simulator/tello_simulator`.
   최초 임포트는 수 분 + 인터넷 필요(git URL 패키지).
3. **씬 확인** (§8) — 최초 1회.
4. `Assets/test.unity` → ▶ Play → Console `[Tello] UDP server listening on 9000` 확인.

C# 파일들은 리눅스에서 텍스트로 편집되는 일이 있다. **에디터에서 처음 열 때는
Play 전에 Console에 빨간 컴파일 에러가 없는지부터 본다** — 있으면 Play 자체가
안 된다.

방화벽은 **필요 없다.** 파이프라인이 같은 PC에서 돌아 전부 localhost다.
(맥에서 첫 Play 때 "수신 연결 허용" 팝업이 뜨면 허용.)

## 4. 실행 (매번)

**⓪ 정리 (시작 전)**
```bash
# 이전 실행이 남아 있으면
pkill -f api_server.py ; pkill -f 'patrol/server.py'
# Unity Play 껐다 켜기
```

**① 서버 — LLM 엔드포인트** `[서버, 계속 켜둠]` (§1)

**② 로컬 — Unity Play**
`Assets/test.unity` → ▶ Play.
✅ Console 초록 `[Tello] UDP server listening on 9000`.
❌ 빨강 `address already in use` → `lsof -i :9000` → `kill -9 <PID>` → 재Play.
**여기 통과 못 하면 뒤가 전부 안 됨.**

**③ 로컬 — 경로 게이트**
```bash
python simulator/bridge/smoke.py --unity-host 127.0.0.1
```
✅ `command -> 'ok'` + `state pos=...`.
❌ `timeout` → ②의 Unity 9000 문제. **'ok' 나와야 ④ 진행.**

건너뛰지 말 것. 여길 안 보면 나중에 "드론이 안 움직인다"로 나타나는데 원인이 두
단계 앞이라 찾기 어렵다.

**④ 로컬 — API 서버**
```bash
python api_server.py --port 8123 \
    --llm-url http://<서버IP>:8000/v1 --llm-api-key <토큰>
```
✅ `[llm] server ok, models=[...]` → `[sim] Unity 127.0.0.1:9000 -> 'ok'` →
`[api] http://127.0.0.1:8123   (규격: /docs)` → `Uvicorn running on ...`

❌ `[Errno 48] address already in use` → 그 포트를 누가 쓰고 있다.
`lsof -nP -iTCP:<포트> -sTCP:LISTEN` 로 확인:

- **`api_server.py` 가 또 떠 있으면** 포트만 바꿔 도망가면 안 된다. 그 프로세스가
  UDP 브리지(9001)도 쥐고 있어 둘이 Unity를 두고 다툰다. `pkill -f api_server.py`
  후 다시.
- **다른 앱**(VS Code 등)이면 `--port` 를 다른 번호로. 콘솔은 API를 **상대
  경로**로 부르므로 프론트는 손댈 게 없다 — 브라우저 주소만 바뀐다.

**⑤ 브라우저** — `http://localhost:8123`. 규격서는 `/docs` 와 `API.md`.

평면도에서 방을 고르거나 검색창에 자연어를 넣고 → 「순찰 시작」 → 브리핑 →
「작전 시작」. 여기서 실제 비행이 시작된다.

### Unity 없이 스텁으로

물리도 화면도 없지만 좌표와 이벤트를 정직하게 흉내낸다. **탐지까지 흉내내므로
지금은 이쪽이 웹 배선을 끝까지 확인할 수 있는 유일한 경로다.**

```bash
python simulator/bridge/fake_unity_sim.py --detect-per-scan 1 &
python api_server.py --port 8123 --llm-url http://<서버IP>:8000/v1
#   --no-scan 을 주면 scan verb 없는 구버전을 흉내내 rc 회전 폴백을 탄다
```

### 부록 — relay (레거시 방식)

로컬에 데이터를 내릴 수 없어 **파이프라인을 서버에서 돌려야 할 때만** 필요하다.
서버→노트북 UDP가 NAT에 막히므로 UDP를 TCP로 감싸 우회한다.

```bash
① [서버]   python simulator/bridge/relay.py server        # waiting for laptop on TCP :9010
② [노트북] Unity ▶ Play
③ [노트북] python3 .../simulator/bridge/relay.py client --server-host <서버IP>
④ [서버]   python simulator/bridge/smoke.py --unity-host 127.0.0.1
⑤ [서버]   python patrol/server.py --sim --unity-host 127.0.0.1 --llm-url http://127.0.0.1:8000/v1
```

- relay client는 반드시 **Unity가 도는 노트북에서** 실행 (localhost:9000으로 명령을 꽂는다).
- 끊겨도 3초마다 자동 재접속. `--token <문자열>` 로 접속 제한. 서버 방화벽 시
  `sudo ufw allow 9010/tcp`.
- 이 경로에서는 `--viz-dir` 경로 시각화와 탐지 사진 `image_path` 가 공유 마운트
  없이는 동작하지 않는다. 그게 이 방식을 접은 이유이기도 하다.

---
## 5. 사용법

평소에는 **웹 콘솔**에서 쓴다 (`http://localhost:8123`) — 평면도에서 순찰할
구역을 클릭으로 고르고 「순찰 시작」. 콘솔이 어느 API를 부르는지는 `API.md`.

터미널에서 직접 쓰려면 REPL을 띄운다 (API 서버와 **동시에는 안 된다**):

```bash
python patrol/server.py --sim --unity-host 127.0.0.1 \
    --llm-url http://<서버IP>:8000/v1 --llm-api-key <토큰>
```

```
query> 현우방만 탐색해줘        # 별칭 → 002_012
query> 2층 전부 순찰해줘        # 층 전체
query> 화장실 전부 확인해줘      # 방 종류
query> 집 전체 돌면서 사람 있는지 확인해줘
```

**흐름**: 방 해석 → **이륙 1회** → 방마다 A* 이동 + 제자리 360° 스캔 →
사람 탐지 시 **정지 → 라이트 온 → 사진 기록 → 알림** → 복귀·착륙 →
`patrol/out/reports/<ts>_patrol/` 에 `report.md` / `report.html` /
`report.json` + `events/*.jpg` 생성.

다음 쿼리는 드론이 선 자리에서 이어진다.

> **확인 단계는 없다.** 예전에는 Unity 화면의 [이동]/[다음 후보] 버튼을
> 기다렸는데, 순찰 구역을 웹 평면도에서 이륙 전에 고르게 되면서 날고 있는 드론
> 앞에서 다시 물어볼 게 없어졌다.

구역을 못 집으면 아무 방도 안 고르고 이유를 돌려준다. LLM이 그럴듯한 방 코드를
지어내는 일이 있어서, **사용자가 실제로 쓴 말이 뒷받침하지 않는 방은 버린다**
(방 번호·별칭·방 종류·층 중 하나가 문장에 있어야 한다).

| REPL 명령 | 동작 |  | Unity 키 | 동작 |
|---|---|---|---|---|
| `home` | 드론 홈으로 텔레포트 + 미션 리셋 |  | **C** | 1인칭 ↔ 3인칭 |
| `rooms` | 순찰 가능한 방 목록 |  | **L** | 호러 연출 on/off (§9) |
| `report` | 마지막 순찰 보고서 재생성 |  | **F** | 손전등 on/off |
| `quit`/`exit` | 종료 |  | **[** / **]** | 밝기 −/+ (어두우면 `]`) |
|  |  |  | **N** | 나이트샷(IR 초록 화면) |
|  |  |  | **H** | 캠코더 HUD 숨김/표시 |
|  |  |  | **Tab** | 설정 패널 (비행 속도·경로 표시·사운드) |

3인칭 카메라는 드론의 **이동방향 뒤**에서 따라간다.

### 360° 스캔과 사람 탐지

**Unity가 직접 한다.** 서버가 `scan <deg/s> <turns>` 를 보내면 Unity가 제자리로
돌면서 자기 카메라 프레임을 YOLO에 넘기고, 찾으면 상태 채널로 알린다:

```json
{"event":"detect","label":"person","conf":0.87,
 "box":{"l":21.0,"t":33.0,"w":16.0,"h":40.0},"image_path":"/abs/evt.jpg"}
{"event":"scan_done","degrees":360.0}
```

`box` 는 **프레임 대비 %** 다 — 웹 콘솔이 박스를 그리는 단위와 같아서 중간에서
아무도 환산하지 않는다. 픽셀→% 나눗셈은 자기 캡처 해상도를 아는 Unity가 한다.

> **⚠ 탐지 쪽 Unity 컴포넌트가 이 브랜치에 아직 없다.**
> `PatrolPersonDetection.cs` 와 `person_detector_tcp.py` 는
> `origin/feature/drone-camera-person-detection` 에만 있다. 합류시키려면 그
> 컴포넌트가 `TelloSimulator.ReportDetection(label, conf, l, t, w, h, imagePath)`
> 를 부르게 하면 된다. 그 전까지는 아래 폴백을 탄다.

> **구버전 Unity 폴백.** 모르는 verb도 `ok` 로 acked 되므로 ack만으로는 그
> 빌드가 `scan` 을 구현했는지 알 수 없다. 그래서 Unity가 `scan_started` 로 즉시
> 답하게 했고, 그게 3초 안에 안 오면 파이프라인이 구버전으로 판정해 **rc 회전**
> 으로 내려간다(`--scan-mode auto|unity|rc`). 그 경로에서는 탐지가 없다 —
> 외부 디텍터를 UDP 9004로 띄우면 있다 (`docs/patrol-agent.md`).
>
> 조용한 방은 `scan_done` 까지 아무것도 안 보내므로 "아무 이벤트나 기다리기"로는
> 판별이 안 된다. `scan_started` 를 따로 두는 이유가 이것이다.

주요 인자: `--scan-mode auto`, `--hover-height 1.2`, `--scan-deg-per-sec 50`,
`--max-rooms 12`, `--room-aliases`, `--no-light`,
`--viz-dir simulator/tello_simulator/Assets/Resources`(경로·탐지 지점을 씬에 렌더).

방 별칭("현우방")은 `patrol/room_aliases.json` 에서 편집한다 — LitePT
데이터에는 방 코드와 타입만 있어서 이 파일이 없으면 사람 이름 방을 못 찾는다.
웹 콘솔은 자기 이름표를 따로 들고 있어서 어긋나 있다 — `GET /api/rooms` 로
이쪽 이름을 받아갈 수 있다 (`API.md`).

### 홈에서 시작 (Play 즉시)

`TelloSimulator` 의 `spawnAtHome` (기본 켜짐) 이 Play 시 드론을 집 안 홈으로
텔레포트한다. 파이프라인 없이 Unity만 켜도 드론이 집 안에 있다. REPL도 시작할 때
같은 지점으로 한 번 더 텔레포트하므로(`server.py` 의 `teleport_home()`) `home` 을
칠 필요는 없다 — `home` 은 비행 중간에 되돌릴 때 쓴다.

`spawnPosition` 기본값 **(−22.51, 5.20, 5.22)** 는 00809 전용이다. 출처:
`litept_backend.default_home()` = 디텍션 프레임 `(4.50, −1.04, −2.06)` → §6 변환.
**건물이나 glb 배치를 바꾸면 이 값도 바꿔야 한다.** 새 값 구하는 법:

```bash
python patrol/litept_backend.py   # 마지막 줄 home = ... (디텍션 프레임)
```
그 값을 §6 식에 넣거나, 서버를 한 번 띄워 드론이 멈춘 Unity 좌표를 Inspector에서
읽어 `spawnPosition` 에 박으면 된다.

## 6. 좌표계 (00809)

집 glb를 **Position (0, 15.5, 0), Rotation (−90, 0, 0), Scale (5, 5, 5)** 로 배치.
디텍션 프레임(Z-up, m) ↔ Unity(Y-up) 변환:

```
unity = ( -5·x,  5·z + 15.5,  -5·y )      # 디텍션 (x,y,z)
```

- y 오프셋 15.5 = 가장 낮은 바닥(z≈−3.09)을 Unity y=0 위로 올림. TelloSimulator
  `minHeight=0.5` 클램프가 y<0을 막아서 필수.
- `calibrate_transform.py` 가 Unity 복셀맵과 대조해 확정 (candidate `x-y+z-`,
  **score 1.0 / hit_rate 1.0**, 2위 0.65 대비 1.53배). 결과:
  `simulator/bridge/transforms/00809_Qpor2mEya8F.json`.

## 7. 구성 요소

```
llm_server/                  # GPU 서버에서 도는 전부 (이 폴더만 복사해도 뜬다)
├── serve.py                 # OpenAI 호환 HTTP 서버 (stdlib http.server)
├── local_llm.py             # 모델 로드 + generate() — 저장소에서 유일하게 torch를 쓴다
└── requirements.txt         # torch / transformers / accelerate

api_server.py                # 웹 콘솔의 백엔드 (FastAPI). web/ 정적 + 순찰 API
API.md                       # 그 계약 — 콘솔의 어느 함수에서 뭘 부르면 되는지

patrol/                      # 두뇌 (파이썬, torch 없음)
├── server.py                # 디버그 REPL (LLM→LitePT→plan→fly)
├── remote_llm.py            # 유일한 generate() 구현 — HTTP (urllib만)
├── patrol_intent.py         # 순찰 구역 해석 (LLM + 별칭·타입·층 5단 폴백)
├── patrol_mission.py        # 순찰 실행 (구간 비행·스캔 지휘·탐지 반응, on_progress)
├── patrol_report.py         # 보고서 md/html/json
├── detect_events.py         # UDP 9004 탐지 수신 (외부 디텍터용, 기본 경로 아님)
├── litept_backend.py        # detections.json 로드, 포인트 병합, 홈
├── room_index.py            # 방 인덱스·별칭·스캔 포즈·웹 id 변환
├── planner.py               # A* / RRT* (복셀 그리드)
└── room_aliases.json        # 방 별칭 ("현우방" → 002_012)

simulator/                   # 시뮬 (Unity + 브리지)
├── tello_simulator/Assets/
│   ├── TelloSimulator.cs    # UDP 수신, 비행, 360° 스캔, 이벤트 송신
│   ├── CameraFollow.cs      # 3/1인칭 카메라 (이동방향 기준), C키 토글
│   ├── HorrorAtmosphere.cs  # 호러 조명·포그·포스트FX·손전등 (L/F/[/] 키)
│   ├── CamcorderHUD.cs      # 캠코더 UI: REC·배터리·시계·글리치 (N/H 키)
│   ├── SettingsPanel.cs     # 설정 패널: 비행 속도·경로 표시·사운드 (Tab, PlayerPrefs 저장)
│   ├── HorrorAudio.cs       # 앰비언트/스팅어/심박 (클립 없으면 무음)
│   ├── PlannedPathRenderer.cs / FlightReportRenderer.cs / VoxelMapRenderer.cs
│   │                        # 시각화 — 씬에 수동 부착 (§8)
│   └── Resources/Audio/     # 사운드 클립 놓는 곳 (Audio/README.md 참고)
└── bridge/
    ├── unity_bridge.py      # UDP 브리지 (명령 + 상태/이벤트 수신)
    ├── coord_transform.py   # 좌표 변환 (JSON)
    ├── calibrate_transform.py  # 좌표 캘리브레이션
    ├── follow_path.py       # PID rc 추종, fly_mission
    ├── fake_unity_sim.py    # Unity 없는 테스트 스텁 (--detect-per-scan/--no-scan)
    ├── relay.py             # NAT 우회 UDP-over-TCP 릴레이 (레거시, §4 부록)
    ├── smoke.py             # 연결 점검
    └── transforms/*.json    # 건물별 좌표 변환
```

비행: `takeoff` → 20 Hz PID `rc` (드론 현 위치 출발) → `land`.
안전: 상태 5초 끊기면 정지·착륙, 경로 길이 타임아웃, 충돌 카운트.

문서: 웹 콘솔 계약 **`API.md`**, 파이썬 모듈 상세 `patrol/README.md`, GPU 서버
`llm_server/README.md`, 외부 디텍터 계약 `docs/patrol-agent.md`(기본 경로 아님).

## 8. 씬 준비 (최초 1회)

커밋된 `test.unity` 에는 **00809(Qpor2mEya8F)** 가 이미 배치돼 있다. 열어서
Hierarchy에 `tello`(드론) + `Main Camera` + `Qpor2mEya8F` 가 보이면 그대로 Play하면
된다. Scene 뷰에서 드론은 집에 비해 작고 멀어 안 보일 수 있는데 정상 — **Play를
누르면** 드론이 집 안 홈으로 순간이동하고 카메라가 따라간다 (§5의 `spawnAtHome`).

**다른 건물로 바꾸거나 배치를 다시 할 때만** 아래 절차:

1. **glb 임포트**: `data/final_npy/<건물>.glb` 를 Unity **Project 창** `Assets` 로 드래그.
2. **옛 집 제거**: Hierarchy에서 기존 집 루트 우클릭 → **Delete**.
   (두 집을 같이 두면 콜라이더·좌표가 겹쳐 꼬인다.)
3. **배치**: Project의 프리팹 → Hierarchy로 드래그 → 선택 → Inspector **Transform**:

   | Transform | X | Y | Z |
   |---|---|---|---|
   | Position | `0` | `15.5` | `0` |
   | Rotation | `-90` | `0` | `0` |
   | Scale | `5` | `5` | `5` |

4. 루트 선택 → 메뉴 `Tools → Add Mesh Colliders to Selected`.
5. **⌘S (Ctrl+S) 저장** ← 빠지면 다시 열 때 사라진다.

**복셀맵 재익스포트 + 캘리브레이션** (위 값과 다르게 배치했을 때만):
집 루트 선택 → `Tools → Export 3D Voxel Map (Selected Root)` → JSON을
`simulator/bridge/` 로 복사 → 서버에서:
```bash
python simulator/bridge/calibrate_transform.py --building 00809_Qpor2mEya8F \
    --voxel-map simulator/bridge/<복셀맵>.json --scale 5 --translation 0 15.5 0
```
성공 시 `transforms/<건물>.json` 갱신 → 커밋.

### 경로 시각화 컴포넌트 (선택, 최초 1회)

계획 경로·실제 궤적·장애물 복셀을 씬에 그리려면:

1. Hierarchy 우클릭 → Create Empty → 이름 `Visualizers`
2. Inspector에서 컴포넌트 추가:
   - `PlannedPathRenderer` — 계획 경로 (빨간 라인, `Resources/planned_path_3d.json`)
   - `FlightReportRenderer` — 실제 궤적 (하늘색) + 침범 지점 (빨간 구)
   - `VoxelMapRenderer` — 장애물 복셀 (와이어 큐브). `Voxel Map Path` 에
     `simulator/bridge/Qpor2mEya8F_voxel_map_3d.json` 절대경로 입력
3. 씬 저장 (Ctrl+S)

서버는 `--viz-dir simulator/tello_simulator/Assets/Resources` 로 띄운다.
드론의 비행 트레일과 충돌 마커는 `TelloSimulator` 에 내장이라 별도 셋업 불필요.
컴포넌트를 안 붙여도 에러는 안 나고, 설정 패널(Tab)의 해당 토글이 무동작일 뿐이다.

## 9. 호러 연출 (조명·포그·사운드)

**씬 편집 불필요.** `HorrorAtmosphere.cs` 가 `[RuntimeInitializeOnLoadMethod]` 로
Play 시 스스로 뜬다. Play만 누르면 어두워진다.

| 켜지는 것 | 값 |
|---|---|
| 포그 | Exp2, density `0.006`, 암청색 `(0.03, 0.035, 0.05)` |
| 앰비언트 | Skybox → Flat `(0.11, 0.12, 0.16)`, 스카이박스 제거 (카메라 clear = 검정) |
| Directional Light | intensity 1.0 → `0.25`, 따뜻한 백색 → 달빛 청색 |
| 손전등 | Main Camera에 Spot (intensity 28, range 70, 각 68°, soft shadow) — 3인칭·1인칭 둘 다 자연스럽게 앞을 비춤 |
| 드론 필 라이트 | 드론에 약한 Point (intensity 1.6, range 30). 스팟 콘 밖이 완전 암흑이 되는 걸 막아 드론 위치·주변 형태가 읽힌다 |
| 포스트FX | URP Volume 런타임 생성: Vignette / FilmGrain / ColorAdjustments(노출 −0.1, 채도 −35) / Bloom / ChromaticAberration / 그림자 청색 틸트. 카메라 post-processing + FXAA 자동 on |

### 캠코더 HUD (`CamcorderHUD.cs`)

파라노말 액티비티식 캠코더 오버레이. 이것도 자가 부팅이라 씬 편집 불필요.

- 좌상단: 깜빡이는 빨간 ● **REC** + 테이프 카운터 `SP 00:03:41`
- 우상단: **배터리** 게이지 + % (기본 92%에서 시작, 25분에 소진, 비행 중 2배 소모.
  20% 아래면 빨갛게 깜빡임). **순수 연출** — 드론을 착륙시키거나 하지 않는다
- 좌하단: **날짜 + 현재 시각** (실제 시스템 시계)
- 우하단: 카메라 ID, 나이트샷 켜면 `◉ NIGHT SHOT`
- 화면 네 귀퉁이 뷰파인더 브래킷 + 중앙 오토포커스 박스
- 9~26초마다 **VHS 트래킹 글리치** — 찢어진 스캔라인 띠가 화면을 훑고 지나감

`TelloSimulator` 의 옛 좌상단 텔레메트리 라벨(state/rc/position/collision)은
`showDebugHud` 로 내려갔다. 기본 꺼짐, 링크나 충돌 디버깅할 때만 Inspector에서 켠다.

### 키

| 키 | 동작 |
|---|---|
| `L` | 호러 연출 전체 on/off — 원래 밝기로 즉시 복원 (원본 값을 Start에서 캐시). 좌표 캘리브레이션·복셀맵 확인처럼 잘 보여야 하는 작업은 끄고 한다 |
| `F` | 손전등만 토글 |
| `N` | **나이트샷** — 초록 IR 화면 + 광량 1.9배. 캠코더 야간 모드가 육안보다 멀리 보는 걸 흉내낸 것 |
| `H` | 캠코더 HUD 숨김/표시 (스크린샷 찍을 때) |
| `Tab` | **설정 패널** — 아래 참고 |
| `[` / `]` | **밝기 배율 −/+** (0.25 ~ 6.0, 0.25 스텝). 앰비언트·달빛·손전등·필 라이트에 한꺼번에 곱한다 |

키 목록은 화면 최하단에 흐리게 상시 표시된다.

### 값 튜닝

**너무 어두우면 `]` 를 눌러 배율을 올린다.** 콘솔에 `[Horror] brightness = 2.25`
처럼 찍히므로, 마음에 드는 값을 `brightness` 기본값에 그대로 넣으면 고정된다.

Play 중 Hierarchy에서 `HorrorAtmosphere` 오브젝트를 골라 Inspector로 조정해도 된다.
값을 **영구 고정**하려면 씬의 아무 오브젝트에 `HorrorAtmosphere` 컴포넌트를 직접
붙이고 저장 — 그러면 부트스트랩이 건너뛰고 저장된 값이 쓰인다.

실내는 Directional Light가 거의 닿지 않으므로 **어두울 때 올릴 값 순서**는
`brightness` → `ambientColor` → `fillIntensity` → `flashlightIntensity` 다.
`postExposure` 는 음수면 조명과 무관하게 화면을 더 깎으므로 0 근처로 둔다.

`fogDensity` 는 집 glb가 **scale 5배**라 거리 스케일이 커서 `0.003 ~ 0.02` 사이에서
"복도 끝은 안 보이는데 방 안은 보이는" 지점을 찾는 게 좋다. 포그는 밝기 배율과
무관하게 따로 걸리므로, 시야가 답답하면 이 값을 따로 낮춘다.

### 설정 패널 (`SettingsPanel.cs`, Tab)

드래그 가능한 창. 바꾸면 즉시 적용되고 **PlayerPrefs에 저장**돼 다음 Play에도 남는다.

**비행** — `이동 속도` 슬라이더가 `TelloSimulator.moveSpeed`(rc=100일 때의 u/s)를
5~60 사이에서 조절한다. 기본 15는 Inspector 값을 그대로 읽어오고, 옆의 **[기본값]**
버튼이 거기로 되돌린다. 비행 중에 바꿔도 안전하다 (이·착륙은 순간이동이라 무관).

> **눈금 주의.** 서버는 목표 속도를 rc로 바꿀 때 15 u/s를 하드코딩한다
> (`follow_path.UNITY_MOVE_SPEED`). 그래서 배율이 ×r 이면 드론은 `--sim-speed` 로
> 지정한 값의 **r배**로 난다. 두 제어 루프 모두 시뮬레이터가 보고한 위치·yaw로
> 닫혀 있어서 미션 자체는 정상 완료되지만, u/s 눈금이 명목값이 되고 코너에서
> 오버슈트가 커진다. 배율이 1이 아니면 패널이 경고 문구를 띄운다. 비행 시간을
> 재거나 `arrival_threshold` 를 튜닝할 땐 [기본값] 으로 두고 한다.

**경로 표시** — 전부 켜짐이 기본. 어두운 집 안에서 unlit 라인이 밝게 떠서 분위기를
깨므로, 연출 확인할 땐 끄고 디버깅할 땐 켜는 식으로 쓴다.

| 항목 | 대상 |
|---|---|
| 비행 트레일 | `TelloSimulator` 의 `TrailRenderer` (지나온 경로). **숨겨도 기록은 계속**되므로 다시 켜면 전체 궤적이 나온다 |
| 계획 경로 | `PlannedPathRenderer` — A* 결과 (`Resources/planned_path_3d.json`) |
| 비행 리포트 | `FlightReportRenderer` — 실제 궤적 + 침범 마커 (`flight_trajectory_3d.json`) |
| 충돌 마커 | 충돌 지점 빨간 구 |

씬에 `PlannedPathRenderer`/`FlightReportRenderer` 가 없으면 그 토글은 아무 일도
하지 않는다(에러 없음). 붙이는 방법은 §8.

**사운드** — 마스터(=`AudioListener.volume`), 전체 음소거, 그리고 레이어별 볼륨:
앰비언트 / 로터(호버링·전속) / 스팅어 / 심박.

### 사운드 파일

`Assets/Resources/Audio/` 에 클립을 넣으면 자동 로드 —
`ambient.wav`(룸톤 루프), `heartbeat.wav`(360° 스캔 중 볼륨·피치 상승),
`Stingers/*.wav`(12~35초 랜덤 간격, 드론 주변 3D 랜덤 위치에서 재생),
`drone.wav`(로터 루프, 속도에 따라 피치·볼륨 상승) + `drone_takeoff.wav` /
`drone_land.wav`(이·착륙 1회).
자세한 규격·무료 출처는 `Assets/Resources/Audio/README.md`.
**클립이 하나도 없어도 정상 동작한다** (해당 레이어만 조용히 꺼짐).

### 참고

- 비행 트레일·경로선·충돌 마커는 `Sprites/Default` unlit 셰이더라
  어두워져도 그대로 보인다. 트레일이 분위기를 깨면 `TelloSimulator` 의
  `showFlightTrail` 을 끄면 된다.
- 집 glb는 머티리얼 41개가 파일 안에 임베드돼 있어 개별 편집이 안 된다.
  어둡게 만드는 수단은 조명/앰비언트/포그뿐이라 전부 이 스크립트에 모아뒀다.
- UDP 프로토콜은 건드리지 않았다. 서버에서 원격으로 연출을 트리거하려면
  (`fx blackout` 같은 것) `unity_bridge.py` 의 `send_command` 래퍼 1개 +
  `TelloSimulator.ProcessCommand` 분기 1개면 된다. 모르는 verb는 Unity·스텁 양쪽에서
  `ok` 후 무시되므로 하위 호환은 안전.

## 10. Future work

- **Unity `PatrolPersonDetection.cs` 합류** — `origin/feature/drone-camera-person-detection`
  의 컴포넌트가 `TelloSimulator.ReportDetection(...)` 를 부르게 하면 실제 YOLO
  탐지가 파이프라인까지 올라온다. **지금 제일 큰 구멍** — 실제 Unity로 돌리면
  스캔만 돌고 탐지가 비어 있다.
- **`web/` 병합 방향 정리** — 콘솔은 이 브랜치로 들어왔지만(`web-api`) 원본은
  `origin/hyeonwoo` 에도 그대로 있다. 같은 파일이 두 브랜치에 있어 다음 병합에서
  충돌한다. 특히 `startMission()`/`runSearch()` 의 API 배선은 이쪽에만 있어서,
  저쪽 버전이 덮이면 조용히 사라진다.
- **경로 플래너 통일** — 웹 브리핑은 SAC 정책(`rl_planner`, `origin/hyeonwoo`)을
  기대하고 여기 `/plan` 은 A\* 다. 교체 지점은 `patrol_mission.fly_leg` 한 곳.
- **탐지 사진 서빙** — Unity가 저장한 프레임은 로컬 절대경로라 브라우저가 못
  읽는다. 지금은 웹이 화면 공유에서 직접 캡처한다(공유가 꺼져 있으면 `NO FRAME`).
- 다층(cross-floor) 경로 개선 (A* max_iters 한계).
- `return_home` 시뮬레이터 왕복 비행 (현재 SDK JSON에만).

## 11. 트러블슈팅

| 증상 | 원인 / 해결 |
|---|---|
| `[Errno 48] address already in use` (API 서버) | 그 포트를 누가 쓴다. `lsof -nP -iTCP:<포트> -sTCP:LISTEN`. 범인이 `api_server.py` 면 포트를 바꾸지 말고 죽여라 — UDP 브리지까지 쥐고 있다. 다른 앱이면 `--port` 를 옮긴다 (§4 ④) |
| `localhost:<포트>` 가 로컬이 아니라 서버로 감 | **VS Code Remote-SSH 자동 포트 포워딩.** 서버에서 열린 포트를 같은 번호로 로컬에 포워딩해서, 겉보기엔 잘 도는데 딴 기계로 간다. PORTS 패널 → *Stop Forwarding Port*, 또는 안 겹치는 `--port` 사용 |
| 시작 시 `cannot reach the LLM server at ...` | 서버 `llm_server/serve.py`(§1)가 안 떠 있거나 포트가 막힘. `curl http://<서버IP>:8000/health` 로 갈라볼 것 — 응답 오면 `--llm-url` 오타, 안 오면 서버/방화벽. **띄운 직후면 30초쯤 기다릴 것** (모델 로딩) |
| `LLM server returned HTTP 401` | `--api-key` 를 걸어놓고 `--llm-api-key` 를 안 줬거나 값이 다름 |
| `LLM server returned HTTP 404` | `--llm-url` 이 `/v1` 까지여야 한다. `http://<IP>:8000` 도 받아주지만 그 외 경로면 404 |
| `ModuleNotFoundError: No module named 'torch'` (로컬) | 로컬엔 torch가 없는 게 정상이다. `patrol/` 이나 `api_server.py` 가 torch를 끌어온다면 그건 버그 — `llm_server/` 를 import하는 코드가 새로 생긴 것이다 |
| 첫 쿼리가 느림 / 타임아웃 | LLM 서버가 요청을 하나씩 처리한다. 다른 사람이 쓰는 중이면 대기. 늘리려면 `--llm-timeout` |
| `smoke`/시작 시 `-> 'timeout'` | Unity가 9000을 안 듣는 중 (§4 ②). Console에 `listening on 9000` 초록 확인 |
| Unity Console 빨강 `address already in use` | 9000 점유 — `lsof -i :9000` → `kill -9 <PID>` → 재Play. **API 서버와 REPL을 같이 띄우면 이렇게 된다** |
| 명령 보내도 드론 안 움직임 / 배너 안 뜸 | Unity 경로 끊김. §4 ④ smoke로 'ok' 확인부터 |
| 스캔은 도는데 탐지가 0건 | **실제 Unity면 지금은 정상이다** — `PatrolPersonDetection.cs` 가 이 브랜치에 없어 rc 폴백을 탄다 (§5). 로그의 `Unity 쪽 scan 응답 없음` 으로 확인. 배선만 볼 거면 `fake_unity_sim.py --detect-per-scan 1` |
| 순찰 시작이 `409` | 이미 순찰이 돌고 있다. `POST /api/patrol/abort` 로 멈추고 다시 |
| `/plan` 목표가 엉뚱한 층 | 보낸 `goal` 의 z가 그 방 범위 밖 — 응답의 `goal` 이 실제로 쓰인 스냅 좌표다 (`API.md`) |
| Play해도 드론이 씬 기본위치 그대로 | `TelloSimulator` 의 `spawnAtHome` 이 꺼졌거나 `spawnPosition` 이 다른 건물 값 (§5) |
| `home` 쳐도 드론이 안 움직임 | setpos 미도달 = 위와 동일. 순서 ①→⑤ 다시 |
| 3인칭 카메라가 고정 각도 / `InvalidOperationException ... Input System` | 구버전 `CameraFollow.cs` — 최신 pull 후 Unity 재컴파일. 급하면 Player Settings → Active Input Handling → Both |
| Scene 뷰에서 드론 안 보임 | 정상 (집 대비 작고 멂). Play하면 집 안 홈으로 스폰 |
| 드론이 벽으로 돌진 / 바닥·천장에 붙음 | glb 배치값 오류 — §8 (0,15.5,0)/(−90,0,0)/(5,5,5) 재확인 |
| 비행 중 `state lost` | 링크 끊김 — 자동 정지·착륙. 복구 후 `home` |
| Unity 배너 한글 깨짐 | `--sim-no-status` 로 배너를 끌 수 있다 (REPL) |
| Unity 임포트 실패 | 에디터 `6000.3.12f1` 확인, 인터넷 확인, `Library/` 삭제 후 재열기 |
| `No detections.json` | 데이터 미동기화 — §2 |
| (레거시) `relay client: Connection refused` | 서버 relay server(§4 부록 ①)가 안 떠 있음 |
