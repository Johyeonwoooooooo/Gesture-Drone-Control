# 통합 파이프라인: 자연어 명령 → LitePT 탐색 → Unity에서 확인 → 드론 비행

`sim-integration` 브랜치의 통합 시스템입니다.

- **탐색 백엔드**: **LitePT** (minyeong-3d 브랜치 방식) — 사전계산된 ScanNet-20
  인스턴스 디텍션(`data/final_npy/detections.json`)에서 물체를 찾습니다.
  CLIP/Mosaic3D 캐시가 필요 없어 서버 기동이 빠릅니다.
- **시뮬레이터**: Unity 6 기반 Tello 드론 시뮬레이터 (건물 `00809_Qpor2mEya8F`,
  UDP로 Tello 명령 수신). 3인칭 체이스캠 / 1인칭 드론캠 전환 지원.
- **확인(confirm) 플로우**: 탐색 결과를 Unity 카메라가 먼저 비춰주고, 사용자가
  화면의 **[이동]** 버튼을 눌러야 드론이 비행합니다. **[다음 후보]** 버튼으로
  다른 후보를 순환할 수 있습니다.

통합 후 흐름:

```
[GPU 서버 (Linux)]                                  [내 노트북 (Windows/macOS)]
 터미널에 자연어 입력
   └→ LLM 의도 분석 ─→ LitePT 디텍션 매칭(후보 랭킹)     Unity 시뮬레이터 (Play 중)
        └→ 후보 좌표 전송(preview) ──── UDP ──────→   카메라가 후보 위치로 이동
        ┌───────────────────────────── UDP ←──────   [이동]/[다음 후보] 버튼 클릭
        └→ [이동] 확정 시: A*/RRT* 경로 계산
             └→ Tello SDK JSON 저장 (out/)
             └→ 좌표 변환(mosaic→Unity) ─ UDP ────→   드론이 실제로 경로 비행
        └→ 각 단계 상태 메시지 ──────── UDP ──────→   화면 상단 상태 배너 표시
```

| 방향 | 포트 | 내용 |
|---|---|---|
| 서버 → 노트북 | UDP **9000** | Tello 명령 (`command`/`takeoff`/`land`/`rc`/`setpos`/`msg`/`preview`) |
| 노트북 → 서버 | (9000 응답) | 각 명령에 `"ok"` 응답 (서버는 9001에 바인딩) |
| 노트북 → 서버 | UDP **9002** | 드론 상태 JSON 20 Hz + 버튼 이벤트(`{"event":"confirm"\|"next"}`) |

> 프롬프트 입력은 **서버 터미널**에서 합니다. Unity 화면에는 처리 상태 배너와
> 후보 확인 버튼이 표시됩니다. Unity 안에서 직접 텍스트 입력하는 UI는
> future work 입니다 (§9).

---

## 0. 실행 체크리스트 (⭐ 여기부터 읽기)

무엇을 매번 다시 해야 하고 무엇이 1회성인지 정리했습니다.

### A. 최초 1회만 — 설치 & 씬 준비 (한 번 해두면 끝)

| 위치 | 할 일 | 참고 |
|---|---|---|
| 서버 | `git clone`/`checkout sim-integration`, conda `mosaic3d` 환경 구성 | §1 |
| 서버 | `data/final_npy/` 데이터(detections.json + 방별 npy + glb) 확인 | §1 |
| 노트북 | Unity Hub + 에디터 `6000.3.12f1` 설치, `simulator/tello_simulator` 열기 | §2 / §3 |
| 노트북 | 00809 씬 준비: glb 임포트 → 배치(pos 0,15.5,0) → 콜라이더 → 복셀맵 익스포트 | §8 |
| 서버 | 좌표 캘리브레이션 1회 실행 → `transforms/00809_Qpor2mEya8F.json` 갱신·커밋 | §8-5 |

### B. 매번 다시 실행할 때 — 이 5줄만

**순서대로**. (교내 Wi-Fi/VPN 등으로 서버→노트북 직접 UDP가 안 되면 릴레이 필수 — §4-1.)

```
[노트북]
1. Unity 열기 → Assets/test.unity → ▶ Play
   (Console에 "[Tello] UDP server listening on 9000" 확인)
2. (릴레이 쓸 때만) python relay.py client --server-host <서버IP>

[서버]
3. conda activate mosaic3d
4. (릴레이 쓸 때만, 3번보다 먼저 떠 있어야 함) python simulator/bridge/relay.py server
5. python 3D-segmentation/webapp_llm_v2/server.py --llm-device cuda:1 \
       --sim --unity-host <노트북IP 또는 릴레이 시 127.0.0.1>
```

- 서버가 뜨면 `query>` 프롬프트에 자연어 입력 → Unity에서 후보 확인 → [이동] 클릭 (§5).
- Unity Play를 껐다 켜면 드론이 씬 초기위치로 돌아갑니다 → 서버 REPL에서 `home` 입력해 동기화.
- 씬을 안 바꿨으면 A의 캘리브레이션·씬 준비는 다시 할 필요 없습니다.

---

## 1. 서버 (Linux GPU) 준비 — 설치는 (최초 1회), 실행 커맨드는 (매번)

설치·환경 구성은 한 번만:

```bash
git clone <repo> && cd Gesture-Drone-Control
git checkout sim-integration
source /data1/workspaces/jgshin22/miniconda3/etc/profile.d/conda.sh
conda activate mosaic3d          # 환경 구성은 3D-segmentation/setup_env/ 참고
```

**데이터 확인** — LitePT 디텍션 데이터는 git에 없습니다. 서버의
`data/final_npy/` 에 다음이 있어야 합니다:

```
data/final_npy/
├── detections.json                      # 전체 인스턴스 목록 (export_json.py 출력)
├── Qpor2mEya8F.glb                      # Unity 씬용 집 모델
└── 00809_Qpor2mEya8F_<층>_<방>/          # 방별 폴더 × 22
    ├── coord.npy  color.npy  normal.npy
    └── centers.pkl
```

없으면 minyeong-3d 브랜치의 `litept_indoor/` 파이프라인
(`infer_centers.py` → `export_json.py`)으로 생성하세요.

**실행 (매번):** 아래는 시스템을 켤 때마다 실행하는 명령입니다.

```bash
python 3D-segmentation/webapp_llm_v2/server.py \
    --llm-device cuda:1 \
    --sim --unity-host <노트북_IP>
```

- `--sim` 없이 실행하면 Unity 없이 동작합니다 — 후보 확인은 터미널에서
  `[이동=y / 다음=n / 취소=q]` 입력으로 대신합니다.
- 주요 옵션: `--data-dir` (기본 `<repo>/data/final_npy`),
  `--sim-speed 2.0` (Unity 유닛/초; 집이 5배 스케일이라 실제 0.4 m/s),
  `--confirm-timeout 120` (버튼 대기 초), `--sim-no-status` (배너 끄기),
  `--algo astar|rrt`, `--home-xyz X Y Z` (기본: 첫 방 중심, 바닥 위 1 m)

## 2. Windows에서 Unity 시뮬레이터 실행 — 설치·씬 준비는 (최초 1회), Play는 (매번)

1. [Unity Hub](https://unity.com/download) 설치.
2. Unity Hub → Installs → **정확히 `6000.3.12f1`** (Unity 6) 설치.
   다른 버전으로 열면 업그레이드 프롬프트가 뜨는데, 프로젝트가 깨질 수 있으니
   버전을 맞추는 것을 권장합니다.
3. Unity Hub → Projects → Open → 레포의 `simulator/tello_simulator` 폴더 선택.
   - **최초 임포트는 수 분** 걸리고 **인터넷이 필요**합니다
     (UnityGLTF, URDF-Importer 패키지가 git URL로 받아짐).
4. **씬 준비**: 최초 1회 §8의 00809 씬 교체 절차를 수행합니다
   (glb 임포트 → 배치 → 콜라이더). 이미 되어 있으면 생략.
5. Project 창에서 `Assets/test.unity` 씬 더블클릭 → ▶ **Play**.
   Console에 `[Tello] UDP server listening on 9000` 확인.
6. **방화벽**: 첫 Play 때 Windows Defender가 Unity Editor의 네트워크 허용을
   물으면 **허용**. 팝업을 놓쳤다면 관리자 PowerShell에서:
   ```powershell
   New-NetFirewallRule -DisplayName "Unity Tello Sim" -Direction Inbound -Protocol UDP -LocalPort 9000 -Action Allow
   ```
7. 노트북 IP 확인: `ipconfig` → IPv4 주소. 이 값을 서버의 `--unity-host`에 넣습니다.

## 3. macOS에서 Unity 시뮬레이터 실행 — 설치·씬 준비는 (최초 1회), Play는 (매번)

1. Unity Hub 설치 (Apple Silicon이면 Silicon 버전 에디터 선택).
2. 이후 §2와 동일: `6000.3.12f1` 설치 → `simulator/tello_simulator` 열기 →
   씬 준비(§8) → `Assets/test.unity` → Play.
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
# 가짜 시뮬레이터: 프리뷰에 3초 뒤 자동 confirm ([다음 후보] 1회 후)
python simulator/bridge/fake_unity_sim.py --auto-next 1 --auto-confirm-sec 3 &
python simulator/bridge/smoke.py --unity-host 127.0.0.1 --fly
```

### 4-1. 노트북이 Wi-Fi NAT 뒤라 서버→노트북이 안 될 때 (릴레이)

교내 Wi-Fi 등에서는 **노트북 → 서버는 되는데 서버 → 노트북(UDP 9000)은 막히는**
경우가 흔합니다 (smoke.py가 `command timeout`으로 실패, 노트북에서
`ping <서버IP>` 는 성공). 이때는 `relay.py` 로 방향을 뒤집습니다 —
노트북이 서버로 TCP 연결을 걸고, 모든 UDP 트래픽(버튼 이벤트 포함)이 그
연결로 중계됩니다.

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
- **주의**: 릴레이 클라이언트는 반드시 **Unity가 실행 중인 그 노트북에서**
  실행해야 합니다 (Unity로 localhost UDP를 쏘기 때문).

## 5. 사용법

서버 터미널의 `query>` 프롬프트에 자연어로 입력합니다 (한국어/영어):

```
query> 거실에 있는 소파로 가줘
query> 침실 침대 찾아줘
query> go to the refrigerator
```

한 쿼리의 흐름:

1. **의도 분석** (LLM) → **LitePT 디텍션 매칭**: 물체 클래스 + 방 힌트로 후보를
   랭킹 (요청한 방 우선, 포인트 수 순).
2. **후보 확인**: Unity 카메라가 1순위 후보 위치로 날아가고 노란 마커 + 라벨 표시.
   - **[이동]** 클릭 → 해당 후보로 확정, 비행 시작
   - **[다음 후보]** 클릭 → 다음 순위 후보로 카메라 이동 (순환)
   - `--confirm-timeout` (기본 120초) 내 무응답 → 쿼리 취소
3. **경로 계산** (A\*, 드론 현재 위치 기준) → Tello SDK JSON 저장
   (`3D-segmentation/webapp_llm_v2/out/`) → **드론이 경로를 따라 비행**
   (하늘색 궤적, 충돌 시 빨간 구) → 목표 위에서 착륙. 다음 쿼리는 드론이 선
   자리에서 이어집니다.

탐색 가능한 물체 (ScanNet-20, wall/floor 제외):
`cabinet, bed, chair, sofa, table, door, window, bookshelf, picture, counter,
desk, curtain, refrigerator, shower curtain, toilet, sink, bathtub,
otherfurniture` (tv/모니터 등은 otherfurniture로 매칭)

REPL 명령:

| 명령 | 동작 |
|---|---|
| `home` | 드론을 홈으로 텔레포트, 미션 리셋 |
| `quit` / `exit` | 종료 |

Unity 화면 조작:

| 키/버튼 | 동작 |
|---|---|
| **C** 키 | 1인칭(드론캠) ↔ 3인칭(체이스캠) 전환 |
| **[이동]** | 프리뷰 중인 후보로 비행 확정 |
| **[다음 후보]** | 다음 순위 후보 프리뷰 |

화면 요소: 상단 상태 배너(msg), 하단 후보 라벨+버튼, 좌상단 텔레메트리,
노란 후보 마커, 하늘색 비행 궤적, 빨간 충돌 마커.

## 6. 좌표계 (00809)

Unity 씬은 집 glb를 **스케일 5, X축 −90° 회전, 위치 (0, 15.5, 0)** 으로
배치합니다 (§8). mosaic/LitePT 좌표(Z-up, m)와 Unity 좌표(Y-up)의 변환:

```
unity = ( -5·x,  5·z + 15.5,  -5·y )        # mosaic (x,y,z) 기준
```

- y 오프셋 15.5는 **가장 낮은 바닥(z≈−3.09)이 Unity y=0 바로 위**에 오도록
  하는 값입니다. TelloSimulator의 최저고도 클램프(`minHeight=0.5`)가 y<0
  영역을 막기 때문에 이 오프셋이 필수입니다.
- 부호 후보(`x-y+z-`)는 00800 캘리브레이션에서 검증된 것과 동일한 glTF
  임포트 규약을 가정한 **잠정(provisional)** 값입니다. Unity에서 복셀맵을
  익스포트한 뒤 §8의 캘리브레이션으로 확정하세요 (서버가 provisional 상태면
  기동 시 경고를 출력합니다).
- 결과 파일: `simulator/bridge/transforms/00809_Qpor2mEya8F.json`

## 7. 구성 요소

```
simulator/
├── tello_simulator/        # Unity 6 프로젝트 (test.unity 씬)
│   └── Assets/
│       ├── TelloSimulator.cs   # UDP 수신, 비행, preview/버튼 UI, 이벤트 송신
│       └── CameraFollow.cs     # 3인칭/1인칭/프리뷰 카메라 (C키 토글)
├── bridge/
│   ├── unity_bridge.py     # UDP 브리지 (명령 송신 + 상태/이벤트 수신 스레드)
│   ├── coord_transform.py  # mosaic ↔ Unity 좌표 변환 (JSON 로드/저장)
│   ├── calibrate_transform.py  # 좌표 변환 자동 캘리브레이션 (--coords-dir 지원)
│   ├── follow_path.py      # PID 웨이포인트 추종 (rc 명령), fly_mission
│   ├── fake_unity_sim.py   # Unity 없이 테스트용 스텁 (--auto-confirm-sec/--auto-next)
│   ├── relay.py            # NAT 우회 UDP-over-TCP 릴레이 (§4-1; 노트북에 단독 복사 가능)
│   ├── smoke.py            # 네트워크 연결 점검 도구
│   └── transforms/*.json   # 건물별 좌표 변환 (00800 확정, 00809 잠정)
└── docs/AUTOPILOT_3D_GUIDE.md  # (참고) 시각화 컴포넌트·복셀맵 재추출 가이드

3D-segmentation/webapp_llm_v2/
├── server.py               # 메인 REPL (LLM → LitePT → confirm → plan → fly)
├── litept_backend.py       # detections.json 로드/랭킹, 포인트 병합, 홈 계산
├── planner.py  sdk_export.py
```

비행 로직: `takeoff` → 20 Hz PID `rc` 추종 (드론 현 위치에서 출발) → `land`.
안전장치: 상태 수신 5초 끊기면 정지·착륙, 경로 길이 기반 타임아웃, 충돌 카운트 로깅.

## 8. Scene(집) 준비 / 교체 가이드 (최초 1회)

**00809 최초 세팅** (현재 씬은 00800 기준이므로 1회 필요):

1. **glb 임포트**: 파일 탐색기(파인더)에서 `data/final_npy/Qpor2mEya8F.glb` 파일을
   Unity 하단 **Project 창**의 `Assets` 폴더 안으로 드래그 앤 드롭.
   임포트가 끝나면 Assets 안에 `Qpor2mEya8F` 프리팹 아이콘이 생깁니다.
2. **씬에 배치** (`Assets/test.unity` 씬이 열린 상태에서):
   1. Project 창의 `Qpor2mEya8F` 프리팹을 왼쪽 **Hierarchy 창**으로 드래그 앤 드롭
      → Hierarchy에 `Qpor2mEya8F` 오브젝트가 생기고 Scene 뷰에 집이 나타남
      (처음엔 위치/방향이 이상해도 정상 — 다음 단계에서 맞춤).
   2. Hierarchy에서 `Qpor2mEya8F` 를 클릭해 선택 → 오른쪽 **Inspector 창** 맨 위
      **Transform** 컴포넌트의 숫자 칸에 아래 값을 **직접 타이핑**:

      | Transform | X | Y | Z |
      |---|---|---|---|
      | Position | `0` | `15.5` | `0` |
      | Rotation | `-90` | `0` | `0` |
      | Scale | `5` | `5` | `5` |

      Position Y **15.5는 필수**입니다 (§6의 바닥 오프셋).
   3. 확인: 오브젝트를 선택한 채 Scene 뷰에 마우스를 두고 **F 키** → 카메라가
      집으로 이동. 집 바닥이 대략 y=0 근처(드론 아래)에 깔려 있으면 성공.
   4. 기존 00800 집 오브젝트(TEEsavR23oF 계열 이름)는 Hierarchy에서 우클릭 →
      **Delete** (또는 Inspector 이름 옆 체크박스 해제로 비활성화).
3. 배치한 집 루트 선택 → 메뉴 `Tools → Add Mesh Colliders to Selected`.
4. 같은 루트 선택 → `Tools → Export 3D Voxel Map (Selected Root)` →
   생성된 JSON을 레포 `simulator/bridge/` 에 복사.
5. **서버**: 캘리브레이션으로 잠정 변환 확정:
   ```bash
   python simulator/bridge/calibrate_transform.py \
       --building 00809_Qpor2mEya8F \
       --coords-dir data/final_npy \
       --voxel-map simulator/bridge/<새복셀맵>.json \
       --scale 5 --translation 0 15.5 0
   ```
   성공하면 `simulator/bridge/transforms/00809_Qpor2mEya8F.json` 갱신 → **커밋**.
   (점수 마진이 부족하면 실패로 종료하며 후보별 점수표를 출력합니다.)
6. 검증: Play → 서버 실행 → 짧은 쿼리로 시험 비행, 드론이 벽을 뚫지 않는지 확인.

**다른 건물로 교체**할 때도 절차는 같습니다. 추가로:

- LitePT 데이터를 새 건물로 생성 (minyeong-3d 브랜치 `litept_indoor/`:
  방별 coord/color/normal.npy → `infer_centers.py` → `export_json.py`) 후
  서버 `--data-dir` 로 지정.
- glb 배치 시 **바닥이 Unity y=0 위로 오도록** position y를 조정
  (y ≈ −(최저 z) × 5) 하고, 그 값을 캘리브레이션 `--translation` 에 그대로 사용.

## 9. Future work

- **Unity 프론트에서 프롬프트 입력**: Unity에 텍스트 입력 UI를 추가하고 입력을
  UDP로 서버 REPL에 전달하는 방식으로 확장 가능 (현재는 터미널 입력).
- `return_home` 시 시뮬레이터에서도 왕복 비행 (현재는 SDK JSON에만 포함).
- location_hint 기반 다층(cross-floor) 경로 개선 (A* max_iters 한계).

## 10. 트러블슈팅

| 증상 | 원인 / 해결 |
|---|---|
| `smoke.py`: `command` timeout | Unity가 Play 상태인지, `--unity-host` IP가 맞는지, 노트북 인바운드 UDP 9000 방화벽 확인. 노트북이 Wi-Fi NAT 뒤(서버에서 `ping <노트북IP>` 실패)면 §4-1 릴레이 사용 |
| `command -> 'ok'` 는 되는데 state 없음 | 노트북 → 서버 UDP 9002 경로 차단. 서버 방화벽(`ufw allow 9002/udp`)·VPN 설정 확인 |
| 프리뷰는 되는데 **버튼이 안 보임** | Unity가 구버전 `TelloSimulator.cs` — 브랜치 최신 커밋으로 갱신 후 재실행 |
| 버튼을 눌러도 **반응 없음** (서버가 계속 대기) | 버튼 이벤트는 UDP 9002로 갑니다 — state가 정상 수신되는지 먼저 확인 (위 항목), 릴레이 사용 시 릴레이 클라이언트 연결 상태 확인 |
| 후보 확인 중 `확인 시간 초과` | `--confirm-timeout` 내 클릭 없음 — 쿼리 다시 입력 |
| 비행 중 `simulator state lost` | 네트워크 끊김 — 드론은 자동 정지·착륙. 연결 복구 후 `home` 으로 리셋 |
| Unity 상태 배너/버튼 한글 깨짐 | 폰트 문제 — 서버에서 `--sim-no-status` 로 배너를 끌 수 있음 (버튼 라벨은 영문 클래스명으로도 표시됨) |
| Unity 프로젝트 임포트 실패 | 에디터 버전(6000.3.12f1) 확인, 인터넷 연결 확인(git URL 패키지), `Library/` 삭제 후 재열기 |
| 드론이 벽으로 돌진 | 좌표 변환 문제 — §8의 캘리브레이션 재실행 (`transforms/*.json` 확인). 서버 기동 시 `PROVISIONAL` 경고가 있으면 아직 미캘리브레이션 상태 |
| 드론이 바닥/천장에 붙음 | glb 배치 y 오프셋 오류 — §8의 position (0, 15.5, 0) 확인 |
| `No detections.json` | LitePT 데이터 미생성 — §1의 데이터 확인 절차 참고 |
| 포트 9000 사용 중 (Unity Console) | 이전 Play 세션/다른 앱이 점유 — Unity 재시작 또는 점유 프로세스 종료 |
