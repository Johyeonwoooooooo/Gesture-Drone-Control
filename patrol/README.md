# patrol — 자연어 → 탐색/순찰 → 경로 계획 → 시뮬레이터 비행

실행 방법·접속 절차는 저장소 루트 `README.md`. 이 문서는 **모듈 내부** 설명이다.

```
terminal query
  → 로컬 LLM 의도 파싱                (llm_parser.py)
  → FIND / PATROL 라우팅              (patrol_intent.py)
  ├ FIND:   LitePT 디텍션 매칭·랭킹    (litept_backend.py)
  │           → Unity 프리뷰 → [이동] 확인 → A*/RRT* (planner.py) → 비행
  └ PATROL: 방 해석 (room_index.py) → [이동] 확인
              → 방마다 이동 + 360° 스캔, 탐지 반응   (patrol_mission.py)
              → 복귀·착륙 → 보고서                  (patrol_report.py)
  → 비행 명령 프로그램 기록                          (sdk_export.py → out/)
```

비행 자체는 `simulator/bridge` 가 담당한다 (UDP → Unity). 이 패키지는 좌표를
계산해 넘기고 상태/버튼 이벤트를 기다린다.

## 모듈

| 파일 | 역할 |
|---|---|
| `server.py` | 메인 REPL. 모든 인자·상태(드론 위치, 마지막 순찰)를 들고 있다 |
| `llm_parser.py` | HF 로컬 LLM 래퍼. 의도 JSON 1개 생성. 한 인스턴스를 intent/report가 공유 |
| `patrol_intent.py` | FIND/PATROL 라우팅 + 방 지정 해석 (별칭·타입·층) |
| `litept_backend.py` | `detections.json` 로드, 쿼리 매칭·랭킹, 방별 포인트 병합, 홈 좌표 |
| `room_index.py` | 방 인덱스(코드·타입·중심·바닥높이) + 별칭 + 스캔 포즈. `out/room_index.json` 캐시 |
| `planner.py` | 복셀 그리드 A* / RRT*. `chaewon` 브랜치 `comparison/3D.py` 에서 가져와 순수 numpy로 정리 |
| `patrol_mission.py` | 순찰 실행 루프 — 구간 비행, 스캔, 탐지 반응(정지·라이트·사진), 복귀 |
| `detect_events.py` | UDP 9004 탐지 수신기. ARM/DISARM 로 구역 안 스캔 중에만 채택 |
| `patrol_report.py` | 순찰 보고서 md/html/json + 이벤트 사진 |
| `sdk_export.py` | 웨이포인트 → Tello SDK 커맨드 프로그램 JSON |
| `room_aliases.json` | 방 별칭 ("현우방" → `002_012`), `floor_offset` |

## 주요 인자

**LLM** `--llm-model Qwen/Qwen2.5-3B-Instruct` `--llm-device cuda:1`
`--llm-dtype float16` `--llm-device-map`(멀티 GPU 분산)

**플래너** `--algo astar|rrt` `--resolution 0.15`(복셀 크기 m) `--margin 1`(장애물
팽창 셀) `--point-stride 4`(포인트 병합 스트라이드) `--rrt-iter 8000`

**시뮬** `--sim --unity-host <IP>` `--sim-speed 2.0`(Unity u/s, 집 scale 5라
2.0 u/s ≈ 0.4 m/s) `--sim-rc-limit 30` `--sim-transform`(기본
`simulator/bridge/transforms/<building>.json`) `--confirm-timeout 120`

**순찰** `--patrol-port 9004` `--patrol-labels person` `--patrol-min-conf`
`--hover-height 1.2` `--scan-deg-per-sec 50` `--scan-turns 1` `--max-rooms 12`
`--no-light` `--no-patrol-confirm` `--room-aliases` `--report-dir`

**출력** `--out-dir`(기본 `patrol/out`) `--viz-dir`(Unity `Assets/Resources` 를
가리키면 계획 경로·궤적이 씬에 그려짐)

## 연속 미션

각 쿼리는 **드론의 현재 시뮬레이터 위치**에서 시작한다 (상태를 못 받으면 직전
목표 → 홈 순으로 폴백). `home` 을 치면 홈으로 텔레포트하며 리셋.

## 자가 테스트 (서버 없이)

```bash
python patrol/litept_backend.py "거실 소파"   # 매칭·랭킹 + home 좌표 출력
python -m patrol.room_index --list             # 방 목록/별칭 (cwd = repo root)
python -m patrol.detect_events --emit --label person --conf 0.9 --image /abs/x.jpg
```

## 출력 형식 (`out/<timestamp>_<target>.json`)

Ollama 스타일 tool-call dict — 실기체 레이어가 `getattr(drone, name)(**arguments)`
로 그대로 디스패치할 수 있는 형태로 기록한다 (이 브랜치는 시뮬만 비행한다).

```json
{
  "meta": { "query": "...", "target_object": "refrigerator", "action": "goto",
            "return_home": false, "algo": "astar", "building": "00809_Qpor2mEya8F",
            "home_world": [], "start_world": [], "goal_world": [],
            "path_length_m": 7.4, "n_waypoints": 9, "speed": 40,
            "world_to_body": "fixed-heading ... meters*100 -> cm" },
  "waypoints_world_m": [[0, 0, 0]],
  "commands": [
    {"function": {"name": "takeoff",      "arguments": {}}, "sdk": "takeoff"},
    {"function": {"name": "go_xyz_speed", "arguments": {"x": 120, "y": -30, "z": 0, "speed": 40}},
     "sdk": "go 120 -30 0 40"},
    {"function": {"name": "land",         "arguments": {}}, "sdk": "land"}
  ]
}
```

`go_xyz_speed` 는 축당 상대 **cm**, `[-500, 500]`, 속도 `[10, 100]` cm/s. SDK가
거부하는 근접 세그먼트(모든 축 < 20 cm)는 병합하고, 500 cm 초과 축은 분할한다.
`action == "take_photo"` 면 `streamon` + `take_photo` 마커를 덧붙인다.
`return_home` 이면 `home_world` 로 돌아오는 역방향 구간을 덧붙인다.

## 한계

- **건물 전체가 하나의 복셀 그리드**이고 모든 포인트가 장애물이다 (시맨틱
  자유공간 카빙 없음). 층간 A*/RRT*는 포인트 클라우드에 실제 수직 통로(계단)가
  있어야 성공한다.
- **월드→바디 헤딩 고정** (드론이 +world-x 를 본다고 가정). 실제 yaw 추종과
  문 그래프 라우팅은 미구현.
- 탐색 클래스는 ScanNet-20 **폐쇄 집합**이다 (LitePT 출력). tv·모니터 등은
  `otherfurniture` 로 들어간다.
- `location_hint` ("옆 방" 같은 상대 위치)는 파싱만 하고 아직 쓰지 않는다.
