# 순찰 드론 시뮬레이터: 자연어 → 탐색/순찰 → Unity에서 확인 → 비행

서버 터미널에 자연어를 입력하면 → LLM이 의도를 분석하고 → LitePT 사전계산
디텍션에서 대상을 찾은 뒤 → Unity 카메라가 후보(또는 순찰 계획)를 비춰주고 →
사용자가 **[이동]** 을 누르면 → 드론이 시뮬레이터에서 실제로 경로를 비행한다.

두 모드가 프롬프트마다 자동으로 갈린다.

- **FIND** (`거실 소파로 가줘`) — 물체 하나를 찾아 후보 확인 후 비행.
- **PATROL** (`현우방만 탐색해줘`) — 구역을 돌며 360° 스캔, 사람 탐지 시 반응,
  마지막에 순찰 보고서 생성.

구성 요소:

- **탐색 백엔드**: **LitePT** — `data/final_npy/detections.json` (ScanNet-20 인스턴스,
  건물 00809). GPU 추론·CLIP 캐시 불필요, 읽기만 한다.
- **시뮬레이터**: Unity 6 Tello 시뮬레이터. 3인칭(이동방향 뒤) / 1인칭 카메라, C키 전환.
- **확인 플로우**: 프리뷰 → [이동] 확정 / [다음 후보] 순환.

```
[GPU 서버 (Linux)]                                  [노트북 (Windows/macOS)]
 터미널에 자연어 입력
   └→ LLM 의도 분석 ─→ LitePT 디텍션 매칭(후보 랭킹)     Unity 시뮬레이터 (Play 중)
        └→ 후보 좌표 전송(preview) ──── UDP ──────→   카메라가 후보 위치로 이동
        ┌───────────────────────────── UDP ←──────   [이동]/[다음 후보] 버튼 클릭
        └→ [이동] 확정 시: A* 경로 계산
             └→ 좌표 변환(§6) ──────── UDP ──────→   드론이 경로 비행
        └→ 각 단계 상태 ──────────────── UDP ──────→   화면 상단 상태 배너
```

| 방향 | 포트 | 내용 |
|---|---|---|
| 서버 → 노트북 | UDP **9000** | Tello 명령 (`command`/`takeoff`/`land`/`rc`/`setpos`/`msg`/`preview`) |
| 노트북 → 서버 | (9000 응답) | 각 명령에 `"ok"` (서버는 9001 바인딩) |
| 노트북 → 서버 | UDP **9002** | 드론 상태 JSON 20 Hz + 버튼 이벤트(`{"event":"confirm"\|"next"}`) |
| 2D 디텍터 → 서버 | UDP **9004** | 사람 탐지 JSON 한 줄 (별도 프로세스, `docs/patrol-agent.md`) |

> 프롬프트는 **서버 터미널**에서 입력. Unity 화면엔 상태 배너 + 확인 버튼이 표시된다.

---

## 0. 실행 체크리스트 (⭐ 여기부터)

### A. 최초 1회만 — 설치 & 씬 준비

| 위치 | 할 일 | 참고 |
|---|---|---|
| 서버 | `git clone` → `checkout patrol-mvp`, conda 환경 구성 | §1 |
| 서버 | `data/final_npy/` 데이터(detections.json + 방별 npy + glb) 확인 | §1 |
| 노트북 | Unity Hub + 에디터 `6000.3.12f1` 설치, `simulator/tello_simulator` 열기 | §2 / §3 |
| 노트북 | **test.unity** 에 00809(Qpor) 배치 확인 → 콜라이더 → 저장 | §8 |
| (선택) 서버 | 씬을 새로 배치했으면 좌표 캘리브레이션 재실행 | §8 |

> 캘리브레이션 결과(`transforms/00809_Qpor2mEya8F.json`, score 1.0)는 이미 커밋돼
> 있다. glb를 §8의 값 그대로 배치했으면 재실행 불필요.

### B. 매번 다시 실행 — 이 순서 그대로 (§4가 상세)

```
① [서버]   python simulator/bridge/relay.py server          # relay 서버, 계속 켜둠
② [노트북] Unity ▶ Play → Console "listening on 9000"(초록) 확인
③ [노트북] python3 .../simulator/bridge/relay.py client --server-host <서버IP>
④ [서버]   python simulator/bridge/smoke.py --unity-host 127.0.0.1   # 'ok' 게이트
⑤ [서버]   python patrol/server.py --llm-device cuda:1 --sim --unity-host 127.0.0.1
⑥ [서버]   query> 자연어 입력. (드론은 Play 시 이미 집 안 홈에 있음 — §5)
```

**순서 핵심**: ①relay서버 → ②Unity(9000 열림) → ③relay클라이언트 → **④에서 'ok'
확인(게이트) → ⑤파이프라인**. ④가 통과 안 되면 무조건 ②(Unity 9000) 문제 → 뒤로.

> 서버·노트북이 같은 LAN이라 직접 UDP가 되면 릴레이(①③④) 없이 §4의 직접 방식 사용.
> 노트북이 Wi-Fi/VPN NAT 뒤면(대부분) 릴레이 필수.

---

## 1. 서버 (Linux GPU) 준비 — 설치는 (최초 1회)

```bash
git clone <repo> && cd Gesture-Drone-Control
git checkout patrol-mvp
```

**파이썬 환경** — 필요한 건 `requirements.txt` 가 전부다 (numpy, scipy, pillow,
torch, transformers). GPU는 LLM 의도 파서에만 쓴다.

```bash
conda create -n patrol python=3.10 -y && conda activate patrol
pip install -r requirements.txt
```

> **numpy는 반드시 2 미만.** 이 저장소 서버의 기존 env 중에서는 `unidet3d`
> (numpy 1.24 / torch 2.1.2) 가 그대로 동작하고, `mosaic3d` 는 numpy가 2.2로
> 올라가 `torch.from_numpy` 가 `RuntimeError: Numpy is not available` 로 깨져 있다.
> 새로 만들지 않고 쓸 거라면 `conda activate unidet3d`.

**데이터 확인** — git에 없다(용량). 서버에 다음이 있어야 한다:

```
data/final_npy/
├── detections.json                      # 전체 인스턴스 (export_json.py 출력)
├── Qpor2mEya8F.glb                      # Unity 씬용 집 모델
└── 00809_Qpor2mEya8F_<층>_<방>/          # 방별 폴더 × 22
    ├── coord.npy  color.npy  normal.npy
    └── centers.pkl
```

없으면 `minyeong-3d` 브랜치 `litept_indoor/` (`infer_centers.py` → `export_json.py`)로
생성한다. 이 브랜치에는 생성 파이프라인이 없다 — 결과만 읽는다.

## 2. Windows Unity — 설치·씬은 (최초 1회), Play는 (매번)

1. [Unity Hub](https://unity.com/download) 설치 → Installs → **`6000.3.12f1`** (Unity 6).
2. Projects → Open → 레포의 `simulator/tello_simulator`.
   최초 임포트는 수 분 + 인터넷 필요(git URL 패키지).
3. **씬 확인** (§8) — 최초 1회.
4. `Assets/test.unity` → ▶ Play → Console `[Tello] UDP server listening on 9000` 확인.
5. 방화벽: 직접 UDP 방식일 때만 인바운드 9000 허용 (릴레이면 불필요):
   ```powershell
   New-NetFirewallRule -DisplayName "Unity Tello Sim" -Direction Inbound -Protocol UDP -LocalPort 9000 -Action Allow
   ```

## 3. macOS Unity — 설치·씬은 (최초 1회), Play는 (매번)

1. Unity Hub 설치 (Apple Silicon이면 Silicon 에디터), `6000.3.12f1` 설치.
2. `simulator/tello_simulator` 열기 → 씬 확인(§8) → `Assets/test.unity` → Play.
3. 직접 UDP 방식이면 첫 Play 때 "수신 연결 허용" 팝업 → 허용 (릴레이면 불필요).

## 4. 접속 (매번) — 상세

노트북이 NAT 뒤라 서버→노트북 직접 UDP가 안 되는 게 일반적이므로 **릴레이**가 기본.
서버IP는 GPU 박스 주소(예: `166.104.223.32`). 터미널: 서버 2개 + 노트북 1개 + Unity.

**⓪ 정리 (시작 전)**
```bash
# 서버
pkill -f 'relay\.py server'; pkill -f 'patrol/server\.py'
# 노트북: Unity Play 끄고, 이전 relay client 터미널 Ctrl+C
```

**① 서버 — relay server** `[서버 터미널 A, 계속 켜둠]`
```bash
cd /data1/workspaces/jgshin22/Gesture-Drone-Control
python simulator/bridge/relay.py server
```
✅ `waiting for laptop on TCP :9010 ...`

**② 노트북 — Unity Play**
`Assets/test.unity` → ▶ Play.
✅ Console 초록 `[Tello] UDP server listening on 9000`.
❌ 빨간 `address already in use` → Play 멈춤 → `lsof -i :9000` → `kill -9 <PID>` → 재Play.
**여기 통과 못 하면 뒤가 전부 안 됨.**

**③ 노트북 — relay client** `[노트북 터미널]` (Unity Play 유지)
```bash
python3 .../simulator/bridge/relay.py client --server-host 166.104.223.32
```
✅ 노트북 `connected to ...:9010`, **서버 A**에 `laptop connected`.
❌ `Connection refused` → ①이 안 떠 있음.

**④ 서버 — 경로 게이트** `[서버 터미널 B]`
```bash
cd /data1/workspaces/jgshin22/Gesture-Drone-Control
conda activate patrol            # 또는 unidet3d (§1)
python simulator/bridge/smoke.py --unity-host 127.0.0.1
```
✅ `command -> 'ok'` + `state pos=...`.
❌ `timeout` → ②의 Unity 9000 문제. **'ok' 나와야 ⑤ 진행.**

**⑤ 서버 — 파이프라인** `[서버 터미널 B]`
```bash
python patrol/server.py --llm-device cuda:1 --sim --unity-host 127.0.0.1
```
✅ 시작 로그 `[sim] Unity 127.0.0.1:9000 -> 'ok'`.

**릴레이 참고**
- relay client는 반드시 **Unity가 도는 노트북에서** 실행 (localhost:9000으로 명령 꽂음).
- 끊겨도 3초마다 자동 재접속. `--token <문자열>` 로 접속 제한 가능.
- 서버 방화벽 시 `sudo ufw allow 9010/tcp`.

**직접 UDP 방식** (같은 LAN일 때만): 릴레이(①③④) 생략, 노트북 IP 확인
(`ipconfig` / `ipconfig getifaddr en0`) 후 서버에서 `--unity-host <노트북IP>`.
사전 점검 `python simulator/bridge/smoke.py --unity-host <노트북IP> --fly`.

**Unity 없이 서버만 테스트**:
```bash
python simulator/bridge/fake_unity_sim.py --auto-next 1 --auto-confirm-sec 3 &
python patrol/server.py --sim --unity-host 127.0.0.1
```

## 5. 사용법

`query>` 프롬프트에 자연어 입력 (한국어/영어):
```
query> home                    # 홈으로 리셋 (Play 시 이미 홈에서 시작함)
query> 거실에 있는 소파로 가줘
query> 침실 침대 찾아줘
query> go to the refrigerator
```

흐름: 의도 분석(LLM) → LitePT 후보 랭킹 → **Unity 카메라가 1순위 후보로 이동, 노란
마커+라벨** → 사용자가 [이동]/[다음 후보] → 확정 시 드론 현 위치에서 A* → 비행 → 착륙.
다음 쿼리는 드론이 선 자리에서 이어진다.

탐색 물체 (ScanNet-20, wall/floor 제외): `cabinet bed chair sofa table door window
bookshelf picture counter desk curtain refrigerator "shower curtain" toilet sink
bathtub otherfurniture` (tv/모니터 등 → otherfurniture).

| REPL 명령 | 동작 |  | Unity 조작 | 동작 |
|---|---|---|---|---|
| `home` | 드론 홈으로 텔레포트 + 미션 리셋 |  | **C** 키 | 1인칭 ↔ 3인칭 |
| `rooms` | 순찰 가능한 방 목록 |  | **[이동]** | 후보 확정·비행 (순찰은 시작) |
| `report` | 마지막 순찰 보고서 재생성 |  | **[다음 후보]** | 다음 후보 / 다음 구역 |
| `quit`/`exit` | 종료 |  |  |  |
|  |  |  | **L** 키 | 호러 연출 on/off (§9) |
|  |  |  | **F** 키 | 손전등 on/off |
|  |  |  | **[** / **]** | 밝기 −/+ (어두우면 `]`) |
|  |  |  | **N** 키 | 나이트샷(IR 초록 화면) |
|  |  |  | **H** 키 | 캠코더 HUD 숨김/표시 |
|  |  |  | **Tab** 키 | 설정 패널 (경로 표시·사운드) |

3인칭 카메라는 드론의 **이동방향 뒤**에서 따라간다. `--confirm-timeout`(기본 120초) 내
버튼 무응답 시 쿼리 취소.

### 순찰 모드 (구역 탐색 + 보고서)

같은 프롬프트에서 **물체 찾기**와 **구역 순찰**이 자동으로 갈린다 (LLM 라우팅,
`patrol/patrol_intent.py`). 순찰이면:

```
query> 현우방만 탐색해줘        # 별칭 → 002_012
query> 2층 전부 순찰해줘        # 층 전체
query> 집 전체 돌면서 사람 있는지 확인해줘
query> rooms                    # 순찰 가능한 방 목록
query> report                   # 마지막 순찰 보고서 재생성
```

흐름: 방 해석 → (프리뷰 [이동] 확인) → **이륙 1회** → 방마다 A* 이동 + 제자리
360° 스캔 → 사람 탐지 시 **정지 → 라이트 온 → 사진 기록 → 알림** → 복귀·착륙 →
`patrol/out/reports/<ts>_patrol/` 에 `report.md` / `report.html` /
`report.json` + `events/*.jpg` 생성.

2D detection은 **별도 프로세스**가 담당하고, 사람을 찾으면 UDP 9004로 JSON
한 줄을 보낸다 (`{"label":"person","conf":0.87,"image_path":"/abs/evt.jpg"}`).
탐지는 **순찰 구역 안에서 스캔 중일 때만** 채택되고 이동 중 도착분은 버려진다.
전체 계약·Unity 쪽 남은 작업(`light` verb, 촬영 카메라)은 **`docs/patrol-agent.md`**.

주요 인자: `--patrol-port 9004`, `--hover-height 1.2`, `--scan-deg-per-sec 50`,
`--max-rooms 12`, `--no-patrol-confirm`, `--room-aliases`, `--no-light`,
`--viz-dir simulator/tello_simulator/Assets/Resources`(경로·탐지 지점을 씬에 렌더).

방 별칭("현우방")은 `patrol/room_aliases.json` 에서 편집한다 — LitePT
데이터에는 방 코드와 타입만 있어서 이 파일이 없으면 사람 이름 방을 못 찾는다.

### 홈에서 시작 (Play 즉시)

`TelloSimulator` 의 `spawnAtHome` (기본 켜짐) 이 Play 시 드론을 집 안 홈으로
텔레포트한다. 서버 없이 Unity만 켜도 드론이 집 안에 있다. 서버도 시작할 때
같은 지점으로 한 번 더 텔레포트하므로(`server.py` 의 `teleport_home()`), REPL에서
`home` 을 칠 필요는 없다 — `home` 은 비행 중간에 되돌릴 때 쓴다.

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
patrol/                      # 두뇌 (파이썬)
├── server.py                # 메인 REPL (LLM→LitePT→confirm→plan→fly)
├── llm_parser.py            # 로컬 HF LLM — 의도 JSON 파싱 (FIND/PATROL 공용)
├── patrol_intent.py         # FIND/PATROL 라우팅 + 방 해석
├── patrol_mission.py        # 순찰 실행 (구간 비행·360° 스캔·탐지 반응)
├── patrol_report.py         # 보고서 md/html/json
├── detect_events.py         # UDP 9004 탐지 수신 (ARM/DISARM)
├── litept_backend.py        # detections.json 로드/랭킹, 포인트 병합, 홈
├── room_index.py            # 방 인덱스·별칭·스캔 포즈
├── planner.py               # A* / RRT* (복셀 그리드)
├── sdk_export.py            # Tello SDK 커맨드 프로그램 기록 (out/)
└── room_aliases.json        # 방 별칭 ("현우방" → 002_012)

simulator/                   # 시뮬 (Unity + 브리지)
├── tello_simulator/Assets/
│   ├── TelloSimulator.cs    # UDP 수신, 비행, preview/버튼 UI, 이벤트 송신
│   ├── CameraFollow.cs      # 3/1인칭(이동방향 기준)/프리뷰 카메라, C키 토글
│   ├── HorrorAtmosphere.cs  # 호러 조명·포그·포스트FX·손전등 (L/F/[/] 키)
│   ├── CamcorderHUD.cs      # 캠코더 UI: REC·배터리·시계·글리치 (N/H 키)
│   ├── SettingsPanel.cs     # 설정 패널: 경로 표시·사운드 (Tab, PlayerPrefs 저장)
│   ├── HorrorAudio.cs       # 앰비언트/스팅어/심박 (클립 없으면 무음)
│   ├── PlannedPathRenderer.cs / FlightReportRenderer.cs / VoxelMapRenderer.cs
│   │                        # 시각화 — 씬에 수동 부착 (§8)
│   └── Resources/Audio/     # 사운드 클립 놓는 곳 (Audio/README.md 참고)
└── bridge/
    ├── unity_bridge.py      # UDP 브리지 (명령 + 상태/이벤트 수신)
    ├── coord_transform.py   # 좌표 변환 (JSON)
    ├── calibrate_transform.py  # 좌표 캘리브레이션
    ├── follow_path.py       # PID rc 추종, fly_mission
    ├── fake_unity_sim.py    # Unity 없는 테스트 스텁 (--auto-confirm-sec/--auto-next)
    ├── relay.py             # NAT 우회 UDP-over-TCP 릴레이
    ├── smoke.py             # 연결 점검
    └── transforms/*.json    # 건물별 좌표 변환
```

비행: `takeoff` → 20 Hz PID `rc` (드론 현 위치 출발) → `land`.
안전: 상태 5초 끊기면 정지·착륙, 경로 길이 타임아웃, 충돌 카운트.

문서: 순찰 에이전트 계약 `docs/patrol-agent.md`, 파이썬 모듈 상세 `patrol/README.md`.

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
| preview 하이라이트 | `preview` 중 후보 위치에 호박색 Point light — 안 그러면 너무 어두워 [이동]/[다음 후보] 판단 불가 |

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
`ambient.wav`(룸톤 루프), `heartbeat.wav`(후보 접근 시 볼륨·피치 상승),
`Stingers/*.wav`(12~35초 랜덤 간격, 드론 주변 3D 랜덤 위치에서 재생),
`drone.wav`(로터 루프, 속도에 따라 피치·볼륨 상승) + `drone_takeoff.wav` /
`drone_land.wav`(이·착륙 1회).
자세한 규격·무료 출처는 `Assets/Resources/Audio/README.md`.
**클립이 하나도 없어도 정상 동작한다** (해당 레이어만 조용히 꺼짐).

### 참고

- 비행 트레일·경로선·충돌 마커·preview 마커는 `Sprites/Default` unlit 셰이더라
  어두워져도 그대로 보인다. 트레일이 분위기를 깨면 `TelloSimulator` 의
  `showFlightTrail` 을 끄면 된다.
- 집 glb는 머티리얼 41개가 파일 안에 임베드돼 있어 개별 편집이 안 된다.
  어둡게 만드는 수단은 조명/앰비언트/포그뿐이라 전부 이 스크립트에 모아뒀다.
- UDP 프로토콜은 건드리지 않았다. 서버에서 원격으로 연출을 트리거하려면
  (`fx blackout` 같은 것) `unity_bridge.py` 의 `send_command` 래퍼 1개 +
  `TelloSimulator.ProcessCommand` 분기 1개면 된다. 모르는 verb는 Unity·스텁 양쪽에서
  `ok` 후 무시되므로 하위 호환은 안전.

## 10. Future work

- Unity 프론트에서 프롬프트 직접 입력 (현재는 서버 터미널).
- `return_home` 시뮬레이터 왕복 비행 (현재 SDK JSON에만).
- 다층(cross-floor) 경로 개선 (A* max_iters 한계).
- Unity 쪽 `light` verb + 촬영 카메라 (`docs/patrol-agent.md` 참고).

## 11. 트러블슈팅

| 증상 | 원인 / 해결 |
|---|---|
| `RuntimeError: Numpy is not available` | numpy 2.x + 구 torch 조합 — §1의 numpy<2 조건 |
| `relay client`: `Connection refused` | 서버 relay server(§4 ①)가 안 떠 있음. ping은 되는데 refused면 리스너 없음 |
| `smoke`/서버 시작: `-> 'timeout'` | Unity가 9000을 안 듣는 중 (§4 ②). Console에 `listening on 9000` 초록 확인 |
| Unity Console 빨강 `address already in use` | 9000 점유 — `lsof -i :9000` → `kill -9 <PID>` → 재Play |
| 명령 보내도 드론 안 움직임 / 배너 안 뜸 | 서버→Unity 경로 끊김. §4 ④ smoke로 'ok' 확인부터 |
| Play해도 드론이 씬 기본위치 그대로 | `TelloSimulator` 의 `spawnAtHome` 이 꺼졌거나 `spawnPosition` 이 다른 건물 값 (§5) |
| `home` 쳐도 드론이 안 움직임 | setpos 미도달 = 위와 동일. 순서 ①→⑤ 다시 |
| 3인칭 카메라가 고정 각도 | 구버전 `CameraFollow.cs` — 최신 pull 후 Unity 재컴파일 |
| `InvalidOperationException ... Input System` | 구버전 `CameraFollow.cs` — 최신 pull. 급하면 Player Settings → Active Input Handling → Both |
| Scene 뷰에서 드론 안 보임 | 정상 (집 대비 작고 멂). Play하면 집 안 홈으로 스폰 |
| 드론이 벽으로 돌진 / 바닥·천장에 붙음 | glb 배치값 오류 — §8 (0,15.5,0)/(−90,0,0)/(5,5,5) 재확인 |
| 후보 확인 중 `확인 시간 초과` | `--confirm-timeout` 내 버튼 무클릭 — 쿼리 재입력 |
| 비행 중 `state lost` | 네트워크 끊김 — 자동 정지·착륙. 복구 후 `home` |
| Unity 배너/버튼 한글 깨짐 | 서버 `--sim-no-status` 로 배너 끄기 가능 |
| Unity 임포트 실패 | 에디터 `6000.3.12f1` 확인, 인터넷 확인, `Library/` 삭제 후 재열기 |
| `No detections.json` | LitePT 데이터 미생성 — §1 |
