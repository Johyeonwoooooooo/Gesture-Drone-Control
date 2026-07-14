# 통합 파이프라인: 자연어 명령 → 3D 탐색 → Unity 시뮬레이터 비행

`sim-integration` 브랜치는 두 파이프라인을 하나로 연결합니다.

- **gyucheol-webapp-llm-v2**: 터미널에 자연어 입력 → LLM 의도 분석(Qwen) → Mosaic3D
  CLIP 히트맵 + DBSCAN으로 물체 3D 위치 탐색 → A\*/RRT\* 경로 계획 → Tello SDK
  프로그램(JSON) 출력 + viser 3D 시각화
- **jiyun-simul**: Unity 6 기반 Tello 드론 시뮬레이터 (같은 집 `TEEsavR23oF` glb 씬,
  UDP로 Tello 명령 수신)

통합 후 흐름:

```
[GPU 서버 (Linux)]                                  [내 노트북 (Windows/macOS)]
 터미널에 자연어 입력
   └→ LLM 의도 분석 ─→ Mosaic3D 물체 탐색 ─→ A*/RRT* 경로 계산
        └→ Tello SDK JSON 저장 (out/)                Unity 시뮬레이터 (Play 중)
        └→ 좌표 변환(mosaic→Unity) ──── UDP ──────→   드론이 실제로 경로 비행
        └→ 각 단계 상태 메시지 ──────── UDP ──────→   화면 상단 상태 배너 표시
        └→ viser(웹 3D 뷰)에도 드론 위치 실시간 반영
```

| 방향 | 포트 | 내용 |
|---|---|---|
| 서버 → 노트북 | UDP **9000** | Tello 명령 (`command`/`takeoff`/`land`/`rc`/`setpos`/`msg`) |
| 노트북 → 서버 | (9000 응답) | 각 명령에 `"ok"` 응답 (서버는 9001에 바인딩) |
| 노트북 → 서버 | UDP **9002** | 드론 상태 JSON (위치/yaw/비행중/충돌) 20 Hz |

> 프롬프트 입력은 현재 **서버 터미널**에서 합니다. Unity 화면에는 처리 상태
> (의도 분석 중 / 위치 탐색 중 / 경로 계산 중 / 비행 중)가 배너로 표시됩니다.
> Unity 안에서 직접 입력하는 UI는 future work 입니다 (아래 §9).

---

## 1. 서버 (Linux GPU) 준비

한 번만:

```bash
git clone <repo> && cd Gesture-Drone-Control
git checkout sim-integration
source /data1/workspaces/jgshin22/miniconda3/etc/profile.d/conda.sh
conda activate mosaic3d          # 환경 구성은 3D-segmentation/setup_env/ 참고
```

**캐시 확인** — feature 캐시(`*.npy`)는 git에 없습니다. 서버에
`3D-segmentation/cache/00800_TEEsavR23oF/feat/<region>/{feat.npy,coord.npy}` 가
있어야 합니다. 없으면:

```bash
bash 3D-segmentation/scripts/run_precompute.sh TEEsavR23oF cuda:0
```

**좌표 캘리브레이션** — `simulator/bridge/transforms/00800_TEEsavR23oF.json` 이
이미 커밋되어 있으므로 보통 생략합니다. (씬을 바꾼 경우에만 §8 참고.)

**실행:**

```bash
python 3D-segmentation/webapp_llm_v2/server.py \
    --building 00800_TEEsavR23oF \
    --clip-device cuda:0 --llm-device cuda:1 \
    --sim --unity-host <노트북_IP>
```

- `--sim` 없이 실행하면 기존처럼 viser 애니메이션만 동작합니다 (Unity 불필요).
- viser 3D 뷰: 브라우저에서 `http://<서버IP>:8095`
- 주요 옵션: `--sim-speed 2.0` (Unity 유닛/초; 집이 5배 스케일이라 실제 0.4 m/s),
  `--sim-rc-limit 30`, `--sim-no-status` (Unity 배너 끄기), `--algo astar|rrt`

## 2. Windows에서 Unity 시뮬레이터 실행

1. [Unity Hub](https://unity.com/download) 설치.
2. Unity Hub → Installs → **정확히 `6000.3.12f1`** (Unity 6) 설치.
   다른 버전으로 열면 업그레이드 프롬프트가 뜨는데, 프로젝트가 깨질 수 있으니
   버전을 맞추는 것을 권장합니다.
3. Unity Hub → Projects → Open → 레포의 `simulator/tello_simulator` 폴더 선택.
   - **최초 임포트는 수 분** 걸리고 **인터넷이 필요**합니다
     (UnityGLTF, URDF-Importer 패키지가 git URL로 받아짐).
4. Project 창에서 `Assets/test.unity` 씬 더블클릭.
   (씬에 집 모델·콜라이더·드론·카메라가 모두 세팅되어 있어 추가 설정 불필요.)
5. ▶ **Play**. Console에 `[Tello] UDP server listening on 9000` 확인.
6. **방화벽**: 첫 Play 때 Windows Defender가 Unity Editor의 네트워크 허용을
   물으면 **허용**. 팝업을 놓쳤다면 관리자 PowerShell에서:
   ```powershell
   New-NetFirewallRule -DisplayName "Unity Tello Sim" -Direction Inbound -Protocol UDP -LocalPort 9000 -Action Allow
   ```
7. 노트북 IP 확인: `ipconfig` → IPv4 주소. 이 값을 서버의 `--unity-host`에 넣습니다.

## 3. macOS에서 Unity 시뮬레이터 실행

1. Unity Hub 설치 (Apple Silicon이면 Silicon 버전 에디터 선택).
2. 이후 §2와 동일: `6000.3.12f1` 설치 → `simulator/tello_simulator` 열기 →
   `Assets/test.unity` → Play.
3. 첫 Play 때 "수신 네트워크 연결을 허용하시겠습니까?" 팝업 → **허용**.
   (시스템 설정 → 네트워크 → 방화벽에서도 Unity 허용 확인 가능.)
4. 노트북 IP 확인: `ipconfig getifaddr en0` (Wi-Fi 기준).

## 4. 네트워크 요구사항 & 사전 점검

- 서버와 노트북이 **같은 LAN 또는 VPN**에 있어야 합니다 (UDP 양방향).
  - 서버 → 노트북 UDP 9000 도달 (노트북 인바운드 허용)
  - 노트북 → 서버 UDP 9002 도달 (서버 방화벽 사용 시 `sudo ufw allow 9002/udp`)
- ML 파이프라인을 띄우기 **전에** 연결부터 점검하세요:

```bash
# 서버에서 — Unity가 Play 중일 때
python simulator/bridge/smoke.py --unity-host <노트북_IP> --fly
```

`command -> 'ok'`, `state pos=...`, takeoff/land 까지 나오면 통과입니다.
Unity 없이 서버 단독 테스트를 하려면 스텁을 쓰세요:

```bash
python simulator/bridge/fake_unity_sim.py &     # 가짜 시뮬레이터 (localhost)
python simulator/bridge/smoke.py --unity-host 127.0.0.1 --fly
```

### 4-1. 노트북이 Wi-Fi NAT 뒤라 서버→노트북이 안 될 때 (릴레이)

교내 Wi-Fi 등에서는 **노트북 → 서버는 되는데 서버 → 노트북(UDP 9000)은 막히는**
경우가 흔합니다 (smoke.py가 `command timeout`으로 실패, 노트북에서
`ping <서버IP>` 는 성공). 이때는 `relay.py` 로 방향을 뒤집습니다 —
노트북이 서버로 TCP 연결을 걸고, 모든 UDP 트래픽이 그 연결로 중계됩니다.

```bash
# ① 서버: 릴레이 서버 (TCP 9010에서 노트북 접속 대기)
python simulator/bridge/relay.py server

# ② 노트북: relay.py 파일 하나만 복사해서 (Unity Play 상태에서)
python relay.py client --server-host <서버IP>

# ③ 서버: 이후 모든 명령에서 --unity-host 를 127.0.0.1 로
python simulator/bridge/smoke.py --unity-host 127.0.0.1 --fly
python 3D-segmentation/webapp_llm_v2/server.py ... --sim --unity-host 127.0.0.1
```

- 노트북 클라이언트는 끊겨도 3초마다 자동 재접속합니다.
- 원하면 양쪽에 `--token <비밀문자열>` 을 주어 접속을 제한할 수 있습니다.
- 서버 방화벽 사용 시 TCP 9010 인바운드 허용 필요 (`sudo ufw allow 9010/tcp`).

## 5. 사용법

서버 터미널의 `query>` 프롬프트에 자연어로 입력합니다 (한국어/영어):

```
query> 거실에 있는 소파로 가줘
query> 주방에서 냉장고 찾아서 사진 찍고 돌아와
query> go to the tv in the living room
```

한 쿼리의 흐름: 의도 분석 → 물체 탐색 → 경로 계산 → Tello SDK JSON 저장
(`3D-segmentation/webapp_llm_v2/out/`) → **Unity에서 드론이 경로를 따라 비행**
(하늘색 궤적, 충돌 시 빨간 구) → 착륙. 다음 쿼리는 이전 목적지에서 이어집니다.

REPL 명령:

| 명령 | 동작 |
|---|---|
| `home` | 시작점을 홈으로 리셋 (Unity 드론도 홈으로 텔레포트) |
| `building <id>` | 건물 전환 (`00800_TEEsavR23oF` \| `00809_Qpor2mEya8F`) — 00809는 Unity 씬이 없어 viser 애니메이션만 |
| `quit` / `exit` | 종료 |

화면 요소:

- **Unity**: 상단 상태 배너(msg), 좌상단 텔레메트리, 하늘색 비행 궤적, 빨간 충돌 마커
- **viser** (`http://<서버IP>:8095`): 금색 HOME, 청록 DRONE(시뮬레이터 위치 실시간 미러링),
  빨강 타깃 + bbox, 초록 경로, 방 라벨 토글

## 6. 좌표계가 맞는 이유 (요약)

Unity 씬은 집 glb를 **스케일 5, X축 −90° 회전, 위치 (1.26, 0, 0)** 으로 배치했습니다.
mosaic3d 좌표(Z-up, m)와 Unity 좌표(Y-up)는 다음 캘리브레이션된 변환으로 연결됩니다:

```
unity = ( -5·x + 1.26,  5·z,  -5·y )        # mosaic (x,y,z) 기준
```

이 변환은 `simulator/bridge/calibrate_transform.py`가 Unity에서 추출한 복셀맵
(`TEEsavR23oF_voxel_map_3d.json`)과 포인트클라우드를 대조해 8개 부호 후보 중
자동 선택한 것입니다 (일치율 100%, 2위 후보 대비 1.55배 마진). 결과는
`simulator/bridge/transforms/00800_TEEsavR23oF.json` 에 커밋되어 있습니다.

## 7. 구성 요소

```
simulator/
├── tello_simulator/        # Unity 6 프로젝트 (test.unity 씬, TelloSimulator.cs)
├── bridge/
│   ├── unity_bridge.py     # UDP 브리지 (명령 송신 + 상태 수신 스레드)
│   ├── coord_transform.py  # mosaic ↔ Unity 좌표 변환 (JSON 로드/저장)
│   ├── calibrate_transform.py  # 좌표 변환 자동 캘리브레이션
│   ├── follow_path.py      # PID 웨이포인트 추종 (rc 명령), fly_mission
│   ├── fake_unity_sim.py   # Unity 없이 테스트용 프로토콜 스텁
│   ├── relay.py            # NAT 우회 UDP-over-TCP 릴레이 (§4-1; 노트북에 단독 복사 가능)
│   ├── smoke.py            # 네트워크 연결 점검 도구
│   └── transforms/00800_TEEsavR23oF.json   # 캘리브레이션 결과 (커밋됨)
└── docs/AUTOPILOT_3D_GUIDE.md  # (참고) 시각화 컴포넌트·복셀맵 재추출 가이드
```

`3D-segmentation/webapp_llm_v2/server.py` 가 `--sim` 플래그로 이 브리지를 사용합니다.
비행 로직: `setpos(시작점)` → `takeoff` → 20 Hz PID `rc` 추종 → `land`.
안전장치: 상태 수신 5초 끊기면 정지·착륙, 경로 길이 기반 타임아웃, 충돌 카운트 로깅.

## 8. Scene(집) 교체 가이드

새 건물/집으로 바꾸려면:

1. **서버**: 새 건물의 Mosaic3D feature 캐시 생성
   ```bash
   bash 3D-segmentation/scripts/run_precompute.sh <houseID> cuda:0
   ```
   방 라벨 JSON(`labels.json`)도 준비 (기존 건물 형식 참고).
2. **Unity**: 새 glb를 `Assets/`에 임포트 → 씬에 배치.
   **배치할 때 쓴 scale / rotation / position 값을 기록**해 두세요.
   → 메뉴 `Tools → Add Mesh Colliders` (콜라이더 부착)
   → 메뉴 `Tools → Export 3D Voxel Map` (복셀맵 JSON 생성; 상세는
   `simulator/docs/AUTOPILOT_3D_GUIDE.md` 참고)
   → 생성된 JSON을 레포 `simulator/bridge/` 에 복사.
3. **서버**: 캘리브레이션 실행
   ```bash
   python simulator/bridge/calibrate_transform.py \
       --building <building_id> --voxel-map simulator/bridge/<새복셀맵>.json \
       --scale <배치스케일> --translation <x> <y> <z>
   ```
   성공하면 `simulator/bridge/transforms/<building_id>.json` 생성 → **커밋**.
   (점수 마진이 부족하면 실패로 종료하며 후보별 점수표를 출력합니다.)
4. `3D-segmentation/webapp_llm_v2/server.py` 의 `ALLOWED_BUILDINGS` 에 새 건물 ID 추가.
5. 검증: 서버 재시작 → REPL `building <id>` → 짧은 쿼리로 시험 비행,
   드론이 벽을 뚫지 않는지 확인.

## 9. Future work

- **Unity 프론트에서 프롬프트 입력**: Unity에 텍스트 입력 UI를 추가하고 입력을
  UDP로 서버 REPL에 전달하는 방식으로 확장 가능 (현재는 터미널 입력).
- 00809 건물의 Unity 씬 제작.
- `return_home` 시 시뮬레이터에서도 왕복 비행 (현재는 SDK JSON에만 포함).

## 10. 트러블슈팅

| 증상 | 원인 / 해결 |
|---|---|
| `smoke.py`: `command` timeout | Unity가 Play 상태인지, `--unity-host` IP가 맞는지, 노트북 인바운드 UDP 9000 방화벽 확인. 노트북이 Wi-Fi NAT 뒤(서버에서 `ping <노트북IP>` 실패)면 §4-1 릴레이 사용 |
| `command -> 'ok'` 는 되는데 state 없음 | 노트북 → 서버 UDP 9002 경로 차단. 서버 방화벽(`ufw allow 9002/udp`)·VPN 설정 확인 |
| 비행 중 `simulator state lost` | 네트워크 끊김 — 드론은 자동 정지·착륙. 연결 복구 후 `home` 으로 리셋 |
| Unity 상태 배너 한글 깨짐 | 폰트 문제 — 서버에서 `--sim-no-status` 로 끄고 터미널 로그만 사용 가능 |
| Unity 프로젝트 임포트 실패 | 에디터 버전(6000.3.12f1) 확인, 인터넷 연결 확인(git URL 패키지), `Library/` 삭제 후 재열기 |
| 드론이 벽으로 돌진 | 좌표 변환 문제 — §8의 캘리브레이션 재실행 (`transforms/*.json` 확인) |
| `No cache dir` / 건물 없음 | feature 캐시 미생성 — §1의 `run_precompute.sh` 실행 |
| 포트 9000 사용 중 (Unity Console) | 이전 Play 세션/다른 앱이 점유 — Unity 재시작 또는 점유 프로세스 종료 |
| 서버 시작 시 `simulator flight disabled` 경고 | 해당 건물의 transforms JSON 없음 — §8 절차로 생성 |
