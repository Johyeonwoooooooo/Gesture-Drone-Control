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
| `patrol_intent.py` | 순찰 구역 해석 — LLM 프롬프트 + 별칭·타입·층·개수 폴백 (아래) |
| `litept_backend.py` | `detections.json` 로드, 방별 포인트 병합, 홈 좌표 |
| `room_index.py` | 방 인덱스(코드·타입·중심·바닥높이) + 별칭 + 스캔 포즈. `out/room_index.json` 캐시 |
| `planner.py` | 복셀 그리드 A* / RRT*. `chaewon` 브랜치 `comparison/3D.py` 에서 가져와 순수 numpy로 정리 |
| `patrol_mission.py` | 순찰 실행 루프 — 구간 비행, 스캔 지휘, 탐지 반응(정지·라이트·사진), 복귀. `on_progress` 로 구조화된 진행 이벤트를 낸다 |
| `detect_events.py` | UDP 9004 탐지 수신기 (외부 디텍터 프로세스용 — **기본 경로 아님**). ARM/DISARM 게이팅 |
| `patrol_report.py` | 순찰 보고서 md/html/json + 이벤트 사진 |
| `room_aliases.json` | 방 별칭 ("현우" → 현우 소유 4개 방), `floor_offset`. **첫 별칭 = 웹 콘솔이 그 방에 쓰는 이름** |

### 구역 해석 순서 (`resolve_rooms`)

원문 스캔이 먼저고 LLM 은 마지막이다. 3B 모델은 이 중 최소 하나를 늘 틀린다.

1. **별칭** 부분일치 (긴 별칭 먼저). 층을 말했으면 그 층으로 좁힌다
   ("2층 복도만"). 모델이 준 방 코드와 **합치지 않는다** — 별칭은 사용자가
   실제로 쓴 글자에서 나오고 22개 방을 다 덮으므로 그 자체로 완전한 답이다.
2. **방 코드** — 모델이 준 id 중 원문이 뒷받침하는 것만.
3. **방 종류** (모델 목록 또는 `ROOM_KW_MAP` 키워드) → 4. **층 전체**
   → 5. **건물 전체**.
6. 아무것도 안 걸리면 **모델이 고른 방을 그대로** ("LLM 추정"). 규칙이 답을
   낼 수 있을 때는 여전히 규칙이 이긴다.

**개수 표현은 위치와 직교한다.** "어디"를 1~5 로 고르고, "몇 개"를 원문의
개수 단어로 자른다 (`_quantity` / `_take`):

| 말 | 결과 |
|---|---|
| 전부 · 전체 · 모든 방 · 싹 · 다 돌 | 후보 **전부**. `--max-rooms` 를 적용하지 않는다 |
| 절반 · 반만 · half | 후보의 **무작위 절반** |
| 아무거나 · 아무 방 · 랜덤 | 무작위 `ANY_ROOMS`(3)개 |

무작위 선택은 **층 오름차순**으로 정렬해 돌려준다 — 층 이동이 한 방향만
남는 게 그 조합에서 최소다 (`room_index.order_rooms` 도 층이 1순위지만, 그
성질이 바깥 정렬에 의존하면 안 된다).

주의 두 가지. **개수만 말하고 층은 안 말했으면 모델이 준 층을 버린다** —
"절반만 해줘" 에 `floors:[1]` 을 답하는 일이 잦고, 그러면 집 전체의 절반이
1층의 절반으로 조용히 쪼그라든다. 그리고 `ROOM_KW_MAP` 의 `'방' → bedroom`
때문에 "전체 **방** 순찰해줘" 가 침실만 골랐었다 — 개수 질의에서는 진짜
종류 단어(거실·화장실·주방…)가 원문에 없으면 종류를 통째로 버린다.

넓은 질의는 2 m² 미만 방을 뺀다(`min_area_m2`). 00809 에서는 안방 `000_003`
(1.3 × 0.6 m) 하나가 빠져 "전체"가 21개다 — 이름으로 부르면 그대로 간다.

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
python -m patrol.patrol_intent                 # 구역 해석 (전부/절반/아무거나)
python -m patrol.patrol_mission                # plan_leg 3단 폴백
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
- **초소형 방은 `margin` 팽창이 문틈을 막는다.** 여유(clearance) =
  `resolution × margin` 이고 기본이 0.15 m 다. 이 팽창이 좁은 방·문을 닫아버려
  A*가 `no path` / `max_iters exceeded` 로 실패한다.

  `patrol_mission.plan_leg` 이 **막힌 그 구간만** `clearance_ladder()` 를 한 칸씩
  내려가며 다시 푼다. **`margin` 을 0으로 내리지는 않는다** — 팽창을 빼면 계획은
  성공하지만 경로가 벽에 붙어 드론이 그대로 긁는다. 대신 해상도를 줄여 필요한
  통로 폭만 좁힌다:

  | rung | res × margin | 여유 | 통과 가능한 최소 통로 폭 |
  |---|---|---|---|
  | 0 | 0.15 × 2 | 0.30 m | 0.75 m |
  | 1 | 0.15 × 1 | 0.15 m | 0.45 m |
  | 2 | 0.075 × 1 | 0.075 m | 0.225 m |
  | 3 | 0.05 × 1 | 0.05 m | 0.15 m |

  다 내려가도 못 닿으면 **뚫고 가지 않고 갈 수 있는 데까지만 간다**
  (`fallback="nearest"`, `planner.astar(best_effort=True)`). 직선(`"direct"`)은
  출발점조차 자유공간이 아닐 때만 나온다.

  00809 실측 23구간(홈→방 22→복귀): rung0 2구간 · 조여서 푼 구간 20 ·
  앞까지만 접근 1(`002_021`, 드론이 못 들어가는 공간) · **직선 0**.
  경로에서 점군까지의 최소거리 실측 최악값 0.078 m (안방 `000_003`).

- **느린 건 A* 자체가 아니라 "실패하는" A* 다.** 탐색 루프가 순수 파이썬이라
  실측 **초당 약 2.5만 노드 확장**이고, 확장 수가 곧 실행 시간이다. 성공하는
  탐색은 목표에서 멈추지만 **실패하는 탐색은 도달 가능한 복셀을 전부 펼쳐야**
  "없다"고 말할 수 있다 — 0.15 m 격자에서 13.9만 확장 = 5.4 s, 0.05 m 격자
  (1700만 셀)에서는 `max_iters`(20만)를 그냥 다 쓰고 8 s. 격자 만드는 시간은
  0.3~0.7 s 로 무시해도 된다.

  그래서 `plan_leg` 은 각 rung 에서 A* 를 돌리기 **전에**
  `planner.connected()` 로 목표가 그 여유에서 도달 가능한지 먼저 본다.
  연결 성분 라벨(`scipy.ndimage.label`) 한 번이면 격자 전체를 답하고 그 뒤
  질의는 공짜다. scipy 가 없으면 항상 True 를 돌려주므로 검사만 생략되고
  결과는 같다. **실측 133 s → 58 s** (경로·여유는 완전히 동일).

  남은 58 s 중 40 s 는 두 구간이다: `000_003`(rung2 가 `max_iters` 를 다 쓰고
  rung3 에서 성공) 17.6 s, `002_021`(전 rung 도달 불가 → `nearest` 의
  best_effort 전개 + 긴 원시 경로의 `smooth_path` O(n²)) 22.4 s. 더 줄이려면
  격자를 구간 bbox 로 잘라 만드는 게 다음 수순이다.

  > 가중 A*(heuristic × w)는 **여기서 안 통한다**. 3층 건물이라 직선 휴리스틱이
  > 천장을 가리키고, w 를 올리면 그 방향으로 더 확신해서 판다. 실측 6구간 중
  > 1구간만 개선되고 `002_016` 은 6.4만 → 13.3만 확장으로 오히려 2배 나빠졌다.

  동작 검사: `python -m patrol.patrol_mission` (유니티·시뮬레이터 불필요).
- **월드→바디 헤딩 고정** (드론이 +world-x 를 본다고 가정). 실제 yaw 추종과
  문 그래프 라우팅은 미구현.
- 탐색 클래스는 ScanNet-20 **폐쇄 집합**이다 (LitePT 출력). tv·모니터 등은
  `otherfurniture` 로 들어간다.
- `location_hint` ("옆 방" 같은 상대 위치)는 파싱만 하고 아직 쓰지 않는다.
