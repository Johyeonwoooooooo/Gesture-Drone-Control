# patrol — 자연어 → 탐색/순찰 → 경로 계획 → 시뮬레이터 비행

실행 방법·접속 절차는 저장소 루트 `README.md`. 이 문서는 **모듈 내부** 설명이다.

```
자연어 (웹 검색창 또는 이 REPL)
  → 순찰 구역 해석    (patrol_intent.py → GPU 서버의 llm_server/, room_index.py)
  → 방마다 A* 구간 비행 + 360° 스캔, 탐지 반응        (patrol_mission.py, planner.py)
  → 복귀·착륙 → 보고서                              (patrol_report.py)
```

비행 자체는 `simulator/bridge` 가 담당한다 (UDP → Unity). 이 패키지는 좌표를
계산해 넘기고 상태/이벤트를 기다린다. 360° 스캔과 사람 탐지는 **Unity 가**
한다 — 이쪽은 `scan` 을 보내고 `detect`/`scan_done` 을 받는다.

**확인 단계는 없다.** 순찰 구역은 웹 콘솔의 평면도에서 이륙 전에 고른다.

일반적인 진입점은 저장소 루트의 `api_server.py`(웹 콘솔용, `API.md`)이고
`server.py` 의 REPL 은 디버그용이다. **둘을 동시에 띄우면 안 된다** — 브리지를
서로 뺏는다.

## 모듈

| 파일 | 역할 |
|---|---|
| `server.py` | 메인 REPL. 모든 인자·상태(드론 위치, 마지막 순찰)를 들고 있다 |
| `remote_llm.py` | 저장소의 유일한 `generate()` — `--llm-url` 의 OpenAI 호환 서버에 HTTP로 물어본다. urllib만 씀 |
| `patrol_intent.py` | 순찰 구역 해석 — LLM 프롬프트 + 별칭·타입·층 5단 폴백 |
| `litept_backend.py` | `detections.json` 로드, 방별 포인트 병합, 홈 좌표 |
| `room_index.py` | 방 인덱스(코드·타입·중심·바닥높이) + 별칭 + 스캔 포즈. `out/room_index.json` 캐시 |
| `planner.py` | 복셀 그리드 A* / RRT*. `chaewon` 브랜치 `comparison/3D.py` 에서 가져와 순수 numpy로 정리 |
| `patrol_mission.py` | 순찰 실행 루프 — 구간 비행, 스캔 지휘, 탐지 반응(정지·라이트·사진), 복귀. `on_progress` 로 구조화된 진행 이벤트를 낸다 |
| `detect_events.py` | UDP 9004 탐지 수신기 (외부 디텍터 프로세스용 — **기본 경로 아님**). ARM/DISARM 게이팅 |
| `patrol_report.py` | 순찰 보고서 md/html/json + 이벤트 사진 |
| `room_aliases.json` | 방 별칭 ("현우방" → `002_012`), `floor_offset` |

## 주요 인자

**LLM** `--llm-url http://<host>:8000/v1` (**필수**) `--llm-model
Qwen/Qwen2.5-3B-Instruct` `--llm-api-key` `--llm-timeout 60`.
모델은 이 프로세스에 절대 올라오지 않는다 — **`patrol/` 은 torch를 안 쓴다.**
서버 쪽은 `llm_server/`.

**플래너** `--algo astar|rrt` `--resolution 0.15`(복셀 크기 m) `--margin 1`(장애물
팽창 셀) `--point-stride 4`(포인트 병합 스트라이드) `--rrt-iter 8000`

**시뮬** `--sim --unity-host <IP>` `--sim-speed 2.0`(Unity u/s, 집 scale 5라
2.0 u/s ≈ 0.4 m/s) `--sim-rc-limit 30` `--sim-transform`(기본
`simulator/bridge/transforms/<building>.json`)

**순찰** `--scan-mode auto|unity|rc` `--patrol-labels person` `--patrol-min-conf`
`--hover-height 1.2` `--scan-deg-per-sec 50` `--scan-turns 1` `--max-rooms 12`
`--no-light` `--room-aliases` `--report-dir` `--patrol-port 9004`(외부 디텍터)

**출력** `--viz-dir`(Unity `Assets/Resources` 를
가리키면 계획 경로·궤적이 씬에 그려짐)

## 연속 미션

각 쿼리는 **드론의 현재 시뮬레이터 위치**에서 시작한다 (상태를 못 받으면 직전
목표 → 홈 순으로 폴백). `home` 을 치면 홈으로 텔레포트하며 리셋.

## 자가 테스트 (서버 없이)

```bash
python patrol/litept_backend.py                # 방별 인스턴스 수 + home 좌표
python -m patrol.room_index --list             # 방 목록/별칭 (cwd = repo root)
python -m patrol.detect_events --emit --label person --conf 0.9 --image /abs/x.jpg

# LLM 왕복만 점검 (llm_server/serve.py 가 떠 있어야 함)
python patrol/remote_llm.py --llm-url http://<host>:8000/v1 "2층 전부 순찰해줘"
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
- **초소형 방은 들어간 뒤 못 나올 수 있다.** `margin`(기본 1셀 = 0.15 m) 팽창이
  좁은 방을 막아버려 A*가 `max_iters exceeded` 로 실패한다. 실측: 안방
  `000_003`(1.3 × 0.6 m)은 진입은 되는데 복귀 경로 계산이 실패한다. 회피책은
  `--margin 0` 또는 `--resolution` 을 낮추는 것.
- **월드→바디 헤딩 고정** (드론이 +world-x 를 본다고 가정). 실제 yaw 추종과
  문 그래프 라우팅은 미구현.
- 탐색 클래스는 ScanNet-20 **폐쇄 집합**이다 (LitePT 출력). tv·모니터 등은
  `otherfurniture` 로 들어간다.
- `location_hint` ("옆 방" 같은 상대 위치)는 파싱만 하고 아직 쓰지 않는다.
