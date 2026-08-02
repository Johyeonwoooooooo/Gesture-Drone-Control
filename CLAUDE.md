# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is (patrol-mvp branch)

A **patrol drone simulator**: type Korean/English at a terminal → a local LLM
parses intent → a target object or a set of rooms is resolved from precomputed
3D detections → the Unity camera previews it and the user confirms with a button
→ the drone flies the planned path in the simulator. Patrol missions add a 360°
scan per room, a reaction to person detections, and a generated report.

Two directories, one pipeline:

1. **`patrol/`** — the brain (Python). REPL, LLM intent parsing, detection
   matching, room index, A*/RRT* planning, patrol mission loop, report writer.
2. **`simulator/`** — the sim. `simulator/bridge/` is the Python↔Unity UDP link
   (+ coordinate transform, PID path following, NAT relay, test stub);
   `simulator/tello_simulator/` is the Unity 6 project.

This branch was cut from `sim-integration` to hold **only** what the patrol
simulator needs. The gesture/voice control layer, the Mosaic3D/UniDet3D research
code, the viser web apps, and the DJI Tello real-drone path were removed — they
live on `main`, `gyucheol*`, `minyeong*`, `jiyun-simul`.

`README.md` is the run guide (설치 → relay 접속 → 실행 → 트러블슈팅).
`patrol/README.md` documents the Python modules. `docs/patrol-agent.md` is the
contract with the separately-owned 2D person detector.

## Environment

One env, no compiled 3D stack. `pip install -r requirements.txt` (numpy, torch,
transformers, accelerate, scipy, pillow) — no spconv/MinkowskiEngine/mmdet3d/CLIP.
GPU is used only by the LLM.

On this box the **`patrol` conda env (5.1G)** is that env: numpy 2.2.6 /
torch 2.13 / transformers 5.14, whole pipeline verified. `conda activate patrol`.

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
# 전체 실행 (릴레이 3단계는 README.md §4)
python simulator/bridge/relay.py server                      # 서버, 계속 켜둠
python simulator/bridge/smoke.py --unity-host 127.0.0.1      # 연결 게이트: 'ok' 필수
python patrol/server.py --sim --unity-host 127.0.0.1 --llm-device cuda:1

# Unity 없이 프로토콜 스텁으로
python simulator/bridge/fake_unity_sim.py --auto-next 1 --auto-confirm-sec 3 &
python patrol/server.py --sim --unity-host 127.0.0.1

# 모듈 자가 테스트
python patrol/litept_backend.py "거실 소파"      # 매칭·랭킹 + home 좌표
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
  `preview`/`preview_off`/`light`; Unity → 서버 **9002** 상태 JSON 20 Hz +
  버튼 이벤트 `{"event":"confirm"|"next"}`; 2D 디텍터 → 서버 **9004** 탐지 JSON.
  모르는 verb는 Unity와 스텁 양쪽에서 `ok` 후 무시되므로 하위 호환이 안전하다.

- **확인 루프가 파이프라인의 중심**이다. 후보를 계산하면 바로 날지 않고
  `preview` 로 Unity 카메라를 보내 사용자의 [이동]/[다음 후보] 를 기다린다
  (`--confirm-timeout`, 기본 120초). 순찰은 계획 전체에 대해 한 번 확인한다.

- **`patrol/server.py` 가 상태를 들고 있다.** 드론 현재 위치(연속 미션의 시작점),
  로드된 백엔드/방 인덱스/복셀 그리드, 마지막 순찰 결과(`report` 재생성용).
  각 쿼리는 드론의 **현재 시뮬 위치**에서 계획한다 — 상태 수신 실패 시 직전
  목표 → 홈 순으로 폴백.

- **LLM 인스턴스는 하나다.** `llm_parser.LocalLLMParser` 를 server가 만들어
  `patrol_intent`(FIND/PATROL 라우팅·방 해석)와 `patrol_report`(보고서 문장)에
  주입한다. 새 LLM 호출을 추가할 때 모델을 또 로드하지 말 것.

- **탐지 이벤트는 ARM 상태에서만 채택된다** (`detect_events.DetectionListener`).
  순찰 중 **방 안에서 스캔할 때만** ARM 하고 이동 중엔 DISARM 이라, 늦게 도착한
  탐지가 엉뚱한 방에 기록되지 않는다. 2D 디텍터 프로세스는 이 저장소 밖이다.

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
