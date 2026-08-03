# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is (web-api branch)

A **patrol drone simulator**: pick patrol areas (on the web console's floor plan,
or by typing Korean/English) → an LLM parses the intent → rooms are resolved from
precomputed 3D detections → the drone flies the planned route in Unity, scanning
each room 360° and reacting to person detections → a report comes out.

Three pieces, on **two machines**:

1. **`llm_server/`** — the only thing that runs on the GPU box. Loads the intent
   model and serves it over an OpenAI-compatible HTTP endpoint. **The only place
   in the repo that imports torch**, and it does not import `patrol/`.
2. **`patrol/`** — the brain (Python, numpy only). LLM intent parsing, detection
   matching, room index, A*/RRT* planning, patrol mission loop, report writer.
   Runs on the PC next to Unity. `api_server.py` (repo root) puts the web console
   in front of it; `patrol/server.py` is the debug REPL.
3. **`simulator/`** — the sim. `simulator/bridge/` is the Python↔Unity UDP link
   (+ coordinate transform, PID path following, legacy NAT relay, test stub);
   `simulator/tello_simulator/` is the Unity 6 project, which owns the 360° scan
   and the person detection.

This branch was cut from `sim-integration` to hold **only** what the patrol
simulator needs. The gesture/voice control layer, the Mosaic3D/UniDet3D research
code, the viser web apps, and the DJI Tello real-drone path were removed — they
live on `main`, `gyucheol*`, `minyeong*`, `jiyun-simul`.

The web console itself (`web/`, HAUNTED OPS) lives on `origin/hyeonwoo` and is
owned by a teammate; **which branch absorbs which is still undecided.** `API.md`
is the contract between it and `api_server.py`, and is written to hold either way.

`README.md` is the run guide. `patrol/README.md` documents the Python modules.
`llm_server/README.md` covers the GPU-box half. `docs/patrol-agent.md` describes
the out-of-process detector, which is no longer the default path.

## Environment

**Two requirement sets, because two machines.** No compiled 3D stack either way —
no spconv/MinkowskiEngine/mmdet3d/CLIP.

| | file | contents |
|---|---|---|
| GPU 서버 | `llm_server/requirements.txt` | torch, transformers, accelerate |
| 로컬 PC | `requirements.txt` | numpy, fastapi, uvicorn (+ scipy/pillow, optional) |

The local side has **no torch**. That is a property to protect, not an accident:
`patrol/` reaches the model only through `remote_llm.py` (stdlib urllib), so a
laptop with numpy runs the whole pipeline. Check it the way the split was
verified — block torch/transformers in `sys.meta_path` and import everything.

On this box the **`patrol` conda env** carries both sets (numpy 2.2.6 /
torch 2.13 / transformers 5.14 / fastapi 0.141), so either half runs here.
`conda activate patrol`.

> **Only when reusing an older env:** `unidet3d` (torch 2.1.2) ships wheels built
> against numpy 1.x, so numpy 2 there makes every `torch.from_numpy` raise
> `RuntimeError: Numpy is not available` — keep numpy < 2 in that env. The code
> itself is numpy-2 clean (verified on 2.2.6 and 2.4.4). The old `mosaic3d` env
> was broken this way and has been deleted; rebuild from another branch's
> `3D-segmentation/setup_env/setup_env.sh` if the Mosaic3D/viser stack is ever
> needed again.

Detections are **read**, never computed here: `data/final_npy/` (gitignored,
~305 MB) holds `detections.json` + per-room `coord/color/normal.npy` +
`centers.pkl`, produced by the `minyeong-3d` branch's `litept_indoor/` pipeline
(ScanNet-20 closed set).

## Common commands

```bash
# [GPU 서버] 계속 켜둠. 이것 하나가 서버에서 도는 전부다
python llm_server/serve.py --port 8000 --llm-device cuda:1 --api-key <토큰>

# [로컬 PC] Unity Play → 탐지기 → API 서버 (웹 콘솔이 이 앞에 붙는다)
python simulator/bridge/smoke.py --unity-host 127.0.0.1   # 연결 게이트: 'ok' 필수
python api_server.py --llm-url http://166.104.223.32:8000/v1 --llm-api-key <토큰>

# 웹 없이 터미널로 (디버그. API 서버와 동시에 띄우지 말 것 — 브리지를 뺏는다)
python patrol/server.py --sim --unity-host 127.0.0.1 \
    --llm-url http://166.104.223.32:8000/v1

# Unity 없이 프로토콜 스텁으로 (--no-scan 이면 rc 회전 폴백 경로를 탄다)
python simulator/bridge/fake_unity_sim.py --detect-per-scan 1 &
python patrol/server.py --sim --unity-host 127.0.0.1 --llm-url http://127.0.0.1:8000/v1

# 모듈 자가 테스트
python patrol/litept_backend.py                  # 방별 인스턴스 수 + home 좌표
python patrol/remote_llm.py --llm-url http://<host>:8000/v1 "2층 전부 순찰해줘"
python -m patrol.room_index --list               # 방 목록/별칭 (cwd = repo root)
python -m patrol.detect_events --emit --label person --conf 0.9 --image /abs/x.jpg

# 좌표 재보정 (Unity에서 복셀맵 export 후)
python simulator/bridge/calibrate_transform.py --building 00809_Qpor2mEya8F \
    --voxel-map simulator/bridge/Qpor2mEya8F_voxel_map_3d.json \
    --scale 5 --translation 0 15.5 0
```

테스트 스위트나 린터는 없다 — 연구 코드.

## Architecture notes that span files

- **좌표계가 두 개다.** 디텍션/플래너 프레임은 Z-up 미터 (`data/final_npy` 의
  `coord.npy` 원본 월드 좌표). Unity는 Y-up. 변환은 `simulator/bridge/
  transforms/<building>.json` 의 아핀 변환이고 `coord_transform.py` 가 읽는다.
  00809는 `unity = (-5x, 5z + 15.5, -5y)` — glb를 scale 5, X축 −90°,
  position (0, 15.5, 0) 으로 배치한 결과이며 `calibrate_transform.py` 가
  Unity 복셀맵과 대조해 score 1.0으로 확정한 값이다. y 오프셋 15.5는 가장 낮은
  바닥을 `minHeight` 클램프 위로 올리기 위한 것 — 건물을 바꾸면 반드시 재보정.

- **UDP 프로토콜** (`unity_bridge.py` ↔ `TelloSimulator.cs`):
  서버 → Unity **9000** `command`/`takeoff`/`land`/`rc`/`setpos`/`msg`/
  `light`/`scan`/`scan_stop`; Unity → 서버 **9002** 상태 JSON 20 Hz + 이벤트
  `scan_started`/`scan_done`/`detect`. 모르는 verb는 Unity와 스텁 양쪽에서
  `ok` 후 무시되므로 verb를 더하는 건 안전하다 — 그런데 **그래서 ack만으로는
  상대가 그 verb를 구현했는지 알 수 없다.** `scan` 이 `scan_started` 로
  즉시 답하는 이유가 이것이고, 그 답이 없으면 `patrol_mission` 이 구버전으로
  판정해 rc 회전으로 내려간다. 조용한 방은 `scan_done` 까지 아무것도 안
  보내므로 "아무 이벤트나 기다리기"로는 판별이 안 된다.

- **`detect` 의 `box` 는 프레임 대비 %** 다. 웹 콘솔의 `__patrolDetect` 가
  그대로 받는 단위라 서버는 중계만 한다. 픽셀→% 나눗셈은 자기 캡처 해상도를
  아는 Unity 에서 한다. 중간에서 환산하려 들지 말 것.

- **진입점이 둘이고 상태는 각자 들고 있다.** `api_server.py`(웹 콘솔용,
  `Scene` + `MissionRunner` + `EventLog`)와 `patrol/server.py`(디버그 REPL).
  둘 다 백엔드/방 인덱스/복셀 그리드를 로드하고 UDP 브리지를 잡는다 — **동시에
  띄우면 서로 뺏는다.** 각 미션은 드론의 **현재 시뮬 위치**에서 계획한다
  (상태 수신 실패 시 직전 목표 → 홈 순으로 폴백).

- **웹은 폴링만 한다.** 콘솔에 WebSocket도 EventSource도 없고 `fetch` 뿐이라
  진행 상황은 `GET /api/patrol/events?since=<seq>` 커서로 나간다. 그 이벤트의
  원천은 `patrol_mission.run_patrol(on_progress=...)` 이다 — 한국어 상태 문장
  (`on_status`)과 **별개의 통로**이고, 웹이 문장을 파싱하는 일이 없도록 일부러
  갈라놨다. 새 진행 단계를 추가하면 `progress()` 한 줄과 `API.md` 표 한 줄.

- **LLM 인스턴스는 하나다.** server가 파서를 하나 만들어
  `patrol_intent`(순찰 구역 해석)와 `patrol_report`(보고서 문장)에 주입한다.
  이 둘이 파이프라인의 LLM 호출 전부다.

- **LLM 접점은 `generate(system, user) -> str` 하나뿐이고, 거기서 기계가
  갈린다.** 클라이언트는 `remote_llm.RemoteLLMParser` 하나(HTTP, urllib만)이고
  프롬프트는 각 호출부가 들고 있다. 모델 자체는 `llm_server/local_llm.py` 에
  있고 **저장소에서 torch를 import하는 파일은 그거 하나**다.

  `llm_server/` 는 `patrol/` 을 import하지 않는다 — 서버는 완성된 system/user
  텍스트를 받아 모델만 돌리므로 프롬프트도 방 목록도 필요 없다. 그래서 그
  폴더만 복사해도 다른 GPU 박스에서 뜬다. 편의상 `patrol` 을 부르고 싶어지는
  순간이 오는데, 그 순간 이 성질이 깨진다. LLM 관련 코드를 추가할 때는
  `generate()` **위에** 얹을 것.

- **relay 는 이제 레거시다.** Unity 와 파이프라인이 같은 PC 라 UDP 가 전부
  localhost 다. `relay.py` 는 로컬에 데이터를 못 내리는 상황용으로만 남겨뒀다.
  로컬↔서버는 나가는 TCP 하나(LLM 엔드포인트)뿐이고 NAT 는 그걸 막지 않는다
  (실측: 노트북 10.100.130.17 → 166.104.223.32 TCP 8000/8080/8443 도달).
  로컬엔 `requirements.txt` + `data/final_npy` 가 필요한데, 파이썬이 실제로
  읽는 건 `detections.json` + 방별 `coord.npy`/`centers.pkl` 뿐이다
  (`color.npy`/`normal.npy` 는 어느 코드도 안 읽는다).

- **방 좌표를 방으로 되짚을 땐 z 를 봐야 한다** (`api_server.Scene.room_at`).
  층이 겹쳐 있어서 xy 만으로는 여러 층의 방이 동시에 맞고, 그중 제일 작은 걸
  고르면 엉뚱한 층으로 간다. 게다가 1층 거실처럼 층고가 높은 방은 2층 높이까지
  bbox 가 걸쳐서 "z 가 박스 안"만으로도 부족하다 — **그 점이 서 있는 바닥**
  (z 이하 중 가장 높은 `floor_z`)을 먼저 고르고 그다음 면적으로 자른다.
  22개 방 전부 자기 방으로 스냅되는지가 판정 기준이다.

- **Unity 스크립트 중 씬에 붙어 있는 건 둘뿐이다** (`TelloSimulator`,
  `CameraFollow`). `HorrorAtmosphere`/`HorrorAudio`/`CamcorderHUD`/
  `SettingsPanel` 은 `TelloSimulator.cs` 가 런타임에 `AddComponent` 하고,
  `PlannedPathRenderer`/`FlightReportRenderer`/`VoxelMapRenderer` 는 필요할 때
  에디터에서 수동으로 붙인다 (README.md §8). 씬 파일에 GUID가 없다고 지우면 안 된다.

- **경로 시각화는 파일 경유다.** 서버가 `--viz-dir` (Unity `Assets/Resources`)
  에 `planned_path_3d.json` / `flight_trajectory_3d.json` 을 쓰고 위 렌더러가
  읽는다. 이 두 파일은 추적 대상이라 실행할 때마다 diff가 생긴다 — 커밋하지 말 것.

- **하드코딩된 절대 경로**가 두 루트로 나뉜다:
  `/data1/workspaces/jgshin22/Gesture-Drone-Control` (README) 와
  `/home/jgshin22/work/Gesture-Drone-Control`. 둘 다 이 저장소의 체크아웃이다.
  스크립트를 고칠 때는 그 파일이 이미 쓰는 스타일을 따를 것.

- 건물은 **00809 (`Qpor2mEya8F`) 하나만** 지원한다. 00800(`TEEsavR23oF`)의 glb·
  복셀맵·transform은 이 브랜치에서 제거했다 — 필요하면
  `git checkout sim-integration -- <path>` 로 되살린다.
