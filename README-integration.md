# 통합 파이프라인: 자연어 명령 → LitePT 탐색 → Unity에서 확인 → 드론 비행

`sim-integration` 브랜치. 서버 터미널에 자연어를 입력하면 → LLM이 의도 분석 →
LitePT 사전계산 디텍션에서 물체를 찾고 → Unity 카메라가 후보를 비춰주며 →
사용자가 **[이동]** 버튼을 누르면 → 드론이 시뮬레이터에서 실제로 경로를 비행합니다.

- **탐색 백엔드**: **LitePT** — `data/final_npy/detections.json` (ScanNet-20 인스턴스,
  건물 00809). CLIP/Mosaic3D 캐시 불필요.
- **시뮬레이터**: Unity 6 Tello 시뮬레이터. 3인칭(이동방향 뒤) / 1인칭 카메라, C키 전환.
- **확인 플로우**: 후보 프리뷰 → [이동] 확정 / [다음 후보] 순환.

```
[GPU 서버 (Linux)]                                  [노트북 (Windows/macOS)]
 터미널에 자연어 입력
   └→ LLM 의도 분석 ─→ LitePT 디텍션 매칭(후보 랭킹)     Unity 시뮬레이터 (Play 중)
        └→ 후보 좌표 전송(preview) ──── UDP ──────→   카메라가 후보 위치로 이동
        ┌───────────────────────────── UDP ←──────   [이동]/[다음 후보] 버튼 클릭
        └→ [이동] 확정 시: A* 경로 계산
             └→ 좌표 변환(mosaic→Unity) ─ UDP ────→   드론이 경로 비행
        └→ 각 단계 상태 ──────────────── UDP ──────→   화면 상단 상태 배너
```

| 방향 | 포트 | 내용 |
|---|---|---|
| 서버 → 노트북 | UDP **9000** | Tello 명령 (`command`/`takeoff`/`land`/`rc`/`setpos`/`msg`/`preview`) |
| 노트북 → 서버 | (9000 응답) | 각 명령에 `"ok"` (서버는 9001 바인딩) |
| 노트북 → 서버 | UDP **9002** | 드론 상태 JSON 20 Hz + 버튼 이벤트(`{"event":"confirm"\|"next"}`) |

> 프롬프트는 **서버 터미널**에서 입력. Unity 화면엔 상태 배너 + 확인 버튼이 표시됩니다.

---

## 0. 실행 체크리스트 (⭐ 여기부터)

### A. 최초 1회만 — 설치 & 씬 준비

| 위치 | 할 일 | 참고 |
|---|---|---|
| 서버 | `git clone`/`checkout sim-integration`, conda `mosaic3d` 구성 | §1 |
| 서버 | `data/final_npy/` 데이터(detections.json + 방별 npy + glb) 확인 | §1 |
| 노트북 | Unity Hub + 에디터 `6000.3.12f1` 설치, `simulator/tello_simulator` 열기 | §2 / §3 |
| 노트북 | **test.unity** 에서 옛 집(TEE) 지우고 00809(Qpor) 배치 → 콜라이더 → 저장 | §8 |
| (선택) 서버 | 씬을 새로 배치했으면 좌표 캘리브레이션 재실행 | §8 |

> 캘리브레이션 결과(`transforms/00809_Qpor2mEya8F.json`, score 1.0)는 이미 커밋돼
> 있음. glb를 §8의 값 그대로 배치했으면 재실행 불필요.

### B. 매번 다시 실행 — 이 순서 그대로 (§4가 상세)

```
① [서버]   python simulator/bridge/relay.py server          # relay 서버, 계속 켜둠
② [노트북] Unity ▶ Play → Console "listening on 9000"(초록) 확인
③ [노트북] python3 .../simulator/bridge/relay.py client --server-host <서버IP>
④ [서버]   python simulator/bridge/smoke.py --unity-host 127.0.0.1   # 'ok' 게이트
⑤ [서버]   python 3D-segmentation/webapp_llm_v2/server.py \
               --llm-device cuda:1 --sim --unity-host 127.0.0.1
⑥ [서버]   query> home  → 드론이 집 안으로. 이후 자연어 쿼리.
```

**순서 핵심**: ①relay서버 → ②Unity(9000 열림) → ③relay클라이언트 → **④에서 'ok'
확인(게이트) → ⑤파이프라인**. ④가 통과 안 되면 무조건 ②(Unity 9000) 문제 → 뒤로.

> 서버·노트북이 같은 LAN이라 직접 UDP가 되면 릴레이(①③④) 없이 §4의 직접 방식 사용.
> 노트북이 Wi-Fi/VPN NAT 뒤면(대부분) 릴레이 필수.

---

## 1. 서버 (Linux GPU) 준비 — 설치는 (최초 1회)

```bash
git clone <repo> && cd Gesture-Drone-Control
git checkout sim-integration
source /data1/workspaces/jgshin22/miniconda3/etc/profile.d/conda.sh
conda activate mosaic3d          # 환경 구성은 3D-segmentation/setup_env/ 참고
```

**데이터 확인** — git에 없음. 서버에 다음이 있어야 함:

```
data/final_npy/
├── detections.json                      # 전체 인스턴스 (export_json.py 출력)
├── Qpor2mEya8F.glb                      # Unity 씬용 집 모델
└── 00809_Qpor2mEya8F_<층>_<방>/          # 방별 폴더 × 22
    ├── coord.npy  color.npy  normal.npy
    └── centers.pkl
```

없으면 minyeong-3d 브랜치 `litept_indoor/` (`infer_centers.py` → `export_json.py`)로 생성.

## 2. Windows Unity — 설치·씬은 (최초 1회), Play는 (매번)

1. [Unity Hub](https://unity.com/download) 설치 → Installs → **`6000.3.12f1`** (Unity 6).
2. Projects → Open → 레포의 `simulator/tello_simulator`.
   최초 임포트는 수 분 + 인터넷 필요(git URL 패키지).
3. **씬 준비** (§8) — 최초 1회.
4. `Assets/test.unity` → ▶ Play → Console `[Tello] UDP server listening on 9000` 확인.
5. 방화벽: 직접 UDP 방식일 때만 인바운드 9000 허용 (릴레이면 불필요):
   ```powershell
   New-NetFirewallRule -DisplayName "Unity Tello Sim" -Direction Inbound -Protocol UDP -LocalPort 9000 -Action Allow
   ```

## 3. macOS Unity — 설치·씬은 (최초 1회), Play는 (매번)

1. Unity Hub 설치 (Apple Silicon이면 Silicon 에디터), `6000.3.12f1` 설치.
2. `simulator/tello_simulator` 열기 → 씬 준비(§8) → `Assets/test.unity` → Play.
3. 직접 UDP 방식이면 첫 Play 때 "수신 연결 허용" 팝업 → 허용 (릴레이면 불필요).

## 4. 접속 (매번) — 상세

노트북이 NAT 뒤라 서버→노트북 직접 UDP가 안 되는 게 일반적이므로 **릴레이**가 기본.
서버IP는 GPU 박스 주소(예: `166.104.223.32`). 터미널: 서버 2개 + 노트북 1개 + Unity.

**⓪ 정리 (시작 전)**
```bash
# 서버
pkill -f 'relay\.py server'; pkill -f webapp_llm_v2/server.py
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
source /data1/workspaces/jgshin22/miniconda3/etc/profile.d/conda.sh && conda activate mosaic3d
python simulator/bridge/smoke.py --unity-host 127.0.0.1
```
✅ `command -> 'ok'` + `state pos=...`.
❌ `timeout` → ②의 Unity 9000 문제. **'ok' 나와야 ⑤ 진행.**

**⑤ 서버 — 파이프라인** `[서버 터미널 B]`
```bash
python 3D-segmentation/webapp_llm_v2/server.py --llm-device cuda:1 --sim --unity-host 127.0.0.1
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
python 3D-segmentation/webapp_llm_v2/server.py --sim --unity-host 127.0.0.1
```

## 5. 사용법

`query>` 프롬프트에 자연어 입력 (한국어/영어):
```
query> home                    # 먼저 — 드론을 집 안 홈으로
query> 거실에 있는 소파로 가줘
query> 침실 침대 찾아줘
query> go to the refrigerator
```

흐름: 의도 분석(LLM) → LitePT 후보 랭킹 → **Unity 카메라가 1순위 후보로 이동, 노란
마커+라벨** → 사용자가 [이동]/[다음 후보] → 확정 시 드론 현 위치에서 A* → 비행 → 착륙.
다음 쿼리는 드론이 선 자리에서 이어짐.

탐색 물체 (ScanNet-20, wall/floor 제외): `cabinet bed chair sofa table door window
bookshelf picture counter desk curtain refrigerator "shower curtain" toilet sink
bathtub otherfurniture` (tv/모니터 등 → otherfurniture).

| REPL 명령 | 동작 |  | Unity 조작 | 동작 |
|---|---|---|---|---|
| `home` | 드론 홈으로 텔레포트 + 미션 리셋 |  | **C** 키 | 1인칭 ↔ 3인칭 |
| `quit`/`exit` | 종료 |  | **[이동]** | 후보 확정·비행 |
|  |  |  | **[다음 후보]** | 다음 후보 |
|  |  |  | **L** 키 | 호러 연출 on/off (§9) |
|  |  |  | **F** 키 | 손전등 on/off |

3인칭 카메라는 드론의 **이동방향 뒤**에서 따라감. `--confirm-timeout`(기본 120초) 내
버튼 무응답 시 쿼리 취소.

## 6. 좌표계 (00809)

집 glb를 **Position (0, 15.5, 0), Rotation (−90, 0, 0), Scale (5, 5, 5)** 로 배치.
mosaic/LitePT(Z-up, m) ↔ Unity(Y-up) 변환:

```
unity = ( -5·x,  5·z + 15.5,  -5·y )      # mosaic (x,y,z)
```

- y 오프셋 15.5 = 가장 낮은 바닥(z≈−3.09)을 Unity y=0 위로 올림. TelloSimulator
  `minHeight=0.5` 클램프가 y<0을 막아서 필수.
- `calibrate_transform.py` 가 Unity 복셀맵과 대조해 확정 (candidate `x-y+z-`,
  **score 1.0 / hit_rate 1.0**, 2위 0.65 대비 1.53배). 결과:
  `simulator/bridge/transforms/00809_Qpor2mEya8F.json`.

## 7. 구성 요소

```
simulator/
├── tello_simulator/Assets/
│   ├── TelloSimulator.cs   # UDP 수신, 비행, preview/버튼 UI, 이벤트 송신
│   ├── CameraFollow.cs     # 3/1인칭(이동방향 기준)/프리뷰 카메라, C키 토글
│   ├── HorrorAtmosphere.cs # 호러 조명·포그·포스트FX·손전등 (L/F 키)
│   ├── HorrorAudio.cs      # 앰비언트/스팅어/심박 (클립 없으면 무음)
│   └── Resources/Audio/    # 사운드 클립 놓는 곳 (README.md 참고)
├── bridge/
│   ├── unity_bridge.py     # UDP 브리지 (명령 + 상태/이벤트 수신)
│   ├── coord_transform.py  # 좌표 변환 (JSON)
│   ├── calibrate_transform.py  # 좌표 캘리브레이션 (--coords-dir)
│   ├── follow_path.py      # PID rc 추종, fly_mission
│   ├── fake_unity_sim.py   # Unity 없는 테스트 스텁 (--auto-confirm-sec/--auto-next)
│   ├── relay.py            # NAT 우회 UDP-over-TCP 릴레이
│   ├── smoke.py            # 연결 점검
│   └── transforms/*.json   # 건물별 좌표 변환
3D-segmentation/webapp_llm_v2/
├── server.py               # 메인 REPL (LLM→LitePT→confirm→plan→fly)
├── litept_backend.py       # detections.json 로드/랭킹, 포인트 병합, 홈
└── planner.py  sdk_export.py
```

비행: `takeoff` → 20 Hz PID `rc` (드론 현 위치 출발) → `land`.
안전: 상태 5초 끊기면 정지·착륙, 경로 길이 타임아웃, 충돌 카운트.

## 8. 씬 준비 (test.unity에서 TEE → Qpor 교체, 최초 1회)

커밋된 `test.unity`에는 옛 집 `TEEsavR23oF`(00800)가 들어있음. 이걸 **00809(Qpor)** 로 교체:

1. **glb 임포트**: 파인더에서 `data/final_npy/Qpor2mEya8F.glb` 를 Unity **Project 창**
   `Assets` 로 드래그. (레포에 이미 `Qpor2mEya8F.glb` 가 있으면 생략.)
2. `Assets/test.unity` 열기 (SampleScene 아님).
3. **옛 집 제거**: Hierarchy에서 `TEEsavR23oF` (집 모양 최상위) 우클릭 → **Delete**.
   (두 집을 같이 두면 콜라이더·좌표가 겹쳐 꼬임.)
4. **Qpor 배치**: Project의 `Qpor2mEya8F` 프리팹 → Hierarchy로 드래그 → 선택 →
   Inspector **Transform** 에 직접 입력:

   | Transform | X | Y | Z |
   |---|---|---|---|
   | Position | `0` | `15.5` | `0` |
   | Rotation | `-90` | `0` | `0` |
   | Scale | `5` | `5` | `5` |

5. Qpor 루트 선택 → 메뉴 `Tools → Add Mesh Colliders to Selected`.
6. **⌘S (Ctrl+S) 저장** ← 빠지면 다시 열 때 사라짐.

확인: Hierarchy에 `tello`(드론) + `Main Camera` + `Qpor2mEya8F` 가 있으면 됨. Scene 뷰에서
드론은 집에 비해 작고 멀어서 안 보일 수 있는데 정상 — Play 후 서버 `home` 치면 드론이
집 안으로 순간이동하고 카메라가 따라감.

**복셀맵 재익스포트가 필요한 경우** (glb를 위 값과 다르게 배치했을 때만):
Qpor 루트 선택 → `Tools → Export 3D Voxel Map (Selected Root)` → JSON을
`simulator/bridge/` 로 복사 → 서버에서 캘리브레이션:
```bash
python simulator/bridge/calibrate_transform.py --building 00809_Qpor2mEya8F \
    --coords-dir data/final_npy --voxel-map simulator/bridge/<복셀맵>.json \
    --scale 5 --translation 0 15.5 0
```
성공 시 `transforms/00809_Qpor2mEya8F.json` 갱신 → 커밋.

## 9. 호러 연출 (조명·포그·사운드)

**씬 편집 불필요.** `HorrorAtmosphere.cs` 가 `[RuntimeInitializeOnLoadMethod]` 로
Play 시 스스로 뜬다. Play만 누르면 어두워짐.

| 켜지는 것 | 값 |
|---|---|
| 포그 | Exp2, density `0.012`, 암청색 `(0.02, 0.025, 0.035)` |
| 앰비언트 | Skybox → Flat `(0.03, 0.035, 0.05)`, 스카이박스 제거 (카메라 clear = 검정) |
| Directional Light | intensity 1.0 → `0.06`, 따뜻한 백색 → 달빛 청색 |
| 손전등 | Main Camera에 Spot (range 45, 각 55°, soft shadow) — 3인칭·1인칭 둘 다 자연스럽게 앞을 비춤 |
| 포스트FX | URP Volume 런타임 생성: Vignette / FilmGrain / ColorAdjustments(노출 −0.5, 채도 −40) / Bloom / ChromaticAberration / 그림자 청색 틸트. 카메라 post-processing + FXAA 자동 on |
| preview 하이라이트 | `preview` 중 후보 위치에 호박색 Point light — 안 그러면 너무 어두워 [이동]/[다음 후보] 판단 불가 |

**끄기: `L` 키** → 원래 밝기로 즉시 복원 (원본 값을 Start에서 캐시함).
좌표 캘리브레이션·복셀맵 확인처럼 잘 보여야 하는 작업은 `L` 로 끄고 하면 된다.
`F` 는 손전등만 토글.

### 값 튜닝

Hierarchy에서 Play 중 `HorrorAtmosphere` 오브젝트를 골라 Inspector에서 조정.
값을 **고정**하고 싶으면 씬의 아무 오브젝트에 `HorrorAtmosphere` 컴포넌트를 직접
붙이고 저장 — 그러면 부트스트랩이 건너뛰고 저장된 값이 쓰인다.

가장 먼저 만질 값은 `fogDensity`. 집 glb가 **scale 5배**라 거리 스케일이 커서
`0.005 ~ 0.03` 사이에서 "복도 끝은 안 보이는데 방 안은 보이는" 지점을 찾는 게 좋다.

### 사운드

`Assets/Resources/Audio/` 에 클립을 넣으면 자동 로드 —
`ambient.wav`(룸톤 루프), `heartbeat.wav`(후보 접근 시 볼륨·피치 상승),
`Stingers/*.wav`(12~35초 랜덤 간격, 드론 주변 3D 랜덤 위치에서 재생).
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

## 11. 트러블슈팅

| 증상 | 원인 / 해결 |
|---|---|
| `relay client`: `Connection refused` | 서버 relay server(§4 ①)가 안 떠 있음. ping은 되는데 refused면 리스너 없음 |
| `smoke`/서버 시작: `-> 'timeout'` | Unity가 9000을 안 듣는 중 (§4 ②). Console에 `listening on 9000` 초록 확인 |
| Unity Console 빨강 `address already in use` | 9000 점유 — `lsof -i :9000` → `kill -9 <PID>` → 재Play |
| 명령 보내도 드론 안 움직임 / 배너 안 뜸 | 서버→Unity 경로 끊김. §4 ④ smoke로 'ok' 확인부터 |
| `home` 쳐도 드론이 씬 기본위치 그대로 | setpos 미도달 = 위와 동일. 순서 ①→⑤ 다시 |
| 3인칭 카메라가 고정 각도 | 구버전 `CameraFollow.cs` — 최신 pull 후 Unity 재컴파일 (이동방향 추종은 최신 커밋) |
| `InvalidOperationException ... Input System` | 구버전 `CameraFollow.cs` — 최신 pull. 급하면 Player Settings → Active Input Handling → Both |
| test.unity 열었는데 집 안 보임 | SampleScene 보는 중일 수 있음. `Assets/test.unity` 로 전환 |
| `Missing Prefab ... house_scan_v2` | SampleScene의 옛 프리팹 참조 — 무해. test.unity 사용, SampleScene 무시 |
| Scene 뷰에서 드론 안 보임 | 정상 (집 대비 작고 멂). Play + `home` 이면 집 안으로 |
| 드론이 벽으로 돌진 / 바닥·천장에 붙음 | glb 배치값 오류 — §8 (0,15.5,0)/(−90,0,0)/(5,5,5) 재확인 |
| 후보 확인 중 `확인 시간 초과` | `--confirm-timeout` 내 버튼 무클릭 — 쿼리 재입력 |
| 비행 중 `state lost` | 네트워크 끊김 — 자동 정지·착륙. 복구 후 `home` |
| Unity 배너/버튼 한글 깨짐 | 서버 `--sim-no-status` 로 배너 끄기 가능 |
| Unity 임포트 실패 | 에디터 `6000.3.12f1` 확인, 인터넷 확인, `Library/` 삭제 후 재열기 |
| `No detections.json` | LitePT 데이터 미생성 — §1 |
