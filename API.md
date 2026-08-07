# API — 웹 콘솔 ↔ 순찰 백엔드

`api_server.py` 가 노출하는 계약. 웹 콘솔(`web/`, HAUNTED OPS)이 붙을 자리다.

```bash
# 로컬 PC (Unity 옆). 한 포트에 web/ 정적 + API 가 같이 뜬다.
# LLM 은 기본이 이 PC 의 Ollama (qwen2.5:3b-instruct) — 플래그가 필요 없다.
python api_server.py --port 8123

# 학교 GPU 서버를 쓸 때 (VPN 필요). "gpu" 가 주소+모델 별칭이고, 토큰은
# PATROL_LLM_API_KEY 환경변수로 줘도 된다 (--llm-api-key 와 같은 값).
python api_server.py --port 8123 --llm-url gpu

# LLM 자체를 빼고 (오프라인 모드). Unity 도 없으면 없는 대로 뜬다
python api_server.py --port 8123 --llm-url ""
```

> **오프라인 모드** (`--llm-url ""`): LLM 호출 두 개만 빠지고 방 인덱스·
> 플래너·미션 루프·보고서·콘솔은 그대로 돈다. `/api/intent` 는 별칭/`N층`/방
> 종류 키워드로 구역을 찾고 — 모델이 방 코드를 제안하는 첫 단계 하나만 못 쓴다
> — 보고서 요약문은 템플릿 문장으로 나온다. `GET /api/status` 의
> `llmOffline: true` 로 판별할 수 있다.

살아있는 규격서는 **`http://localhost:8123/docs`** (FastAPI 자동 생성). 이
문서는 그걸로는 안 보이는 것 — 각 엔드포인트를 콘솔의 어느 함수에서 부르면
되는지 — 를 적는다.

> 포트는 `--port` 로 정한다. 기본값 8000은 로컬에서 이미 쓰이는 일이 잦아
> 권하지 않는다 (README §0 참고). 콘솔이 API를 **상대 경로**로 부르므로 포트를
> 바꿔도 프론트는 손댈 게 없다.

## 왜 한 프로세스인가

드론이 하나고 UDP 링크도 하나다. `web_server.py` 와 patrol 서버가 동시에
9000/9002 를 잡을 수 없다. 게다가 콘솔이 `'api/drone'`, `'plan'` 을 **상대
경로**로 부르므로 정적 파일과 API 가 같은 오리진이어야 한다. 그래서 하나로
합쳤다. `patrol/server.py`(터미널 REPL)는 디버그용으로 남아 있지만 **API 서버와
동시에 띄우면 안 된다** — 둘 다 브리지를 잡으려 한다.

---

## 전체 라우트

| | |
|---|---|
| `GET /api/drone` | `{x, y, z, source:"sim"\|"home", flying, connected}` — 1초 폴링 |
| `GET /api/status` | `{ready, engine, building, rooms, model, llmOffline, mission:{state,id,busy,seq}}` |
| `POST /plan` | `{start?, goal}` → `{engine, success, fallback, clearanceM, steps, bumps, dist, flown, ms, start, goal, path}` |
| `POST /api/intent` | 자연어 → 방 id 목록 (§1) |
| `GET /api/rooms` | 방 22개 + 별칭 + 스캔 포즈 (§1-b) |
| `POST /api/patrol/start` | 순찰 시작 (§2) |
| `GET /api/patrol/events` | 진행 로그, `since` 커서 (§3) |
| `POST /api/patrol/abort` | 중단 (§4) |
| `GET /api/patrol/report/{id}` | 보고서 JSON (§5) |
| `GET /` | `HAUNTED OPS.dc.html` 로 리다이렉트 |
| `GET /<path>` | `web/` 정적 |

`/plan` 의 `path` 는 3D 궤적, `goal` 은 **실제로 쓰인 (스냅된)** 좌표라 다음
구간의 `start` 로 그대로 넘기면 된다.

**경로는 항상 나온다** (`success` 는 이제 늘 true). 미션과 같은 플래너
(`patrol_mission.plan_leg`)를 타서, 여유가 넉넉한 격자에서 막히면 그 구간만
좁혀 다시 풀고 그래도 안 되면 갈 수 있는 데까지 간다. 목표에 닿았는지는
`fallback` 으로 봐야 한다:

| `fallback` | 뜻 | 목표 도달 |
|---|---|---|
| `null` | 여유 0.30 m 격자에서 그대로 풀림 | O |
| `"tightened"` | 여유를 좁혀 그 구간만 재계획 | O |
| `"nearest"` | 드론이 못 들어가는 목표 — 문 앞까지만 | X |
| `"direct"` | 출발점이 막힘. 직선(벽 통과) 최후 수단 | X |

`clearanceM` 은 그 경로가 확보한 장애물 여유 [m]. 자세한 사다리는
`patrol/README.md` §한계.

> **고도 주의.** 콘솔은 `meta.rooms[id].center[2]`(bbox 중간 높이)를 goal z 로
> 보내는데, 그건 우리가 호버하는 높이보다 1 m 쯤 높고 가구 안에 떨어질 수 있다.
> 그래서 서버가 그 점이 속한 방을 찾아 **자기 스캔 포즈로 스냅**한다. 응답의
> `goal` 이 보낸 값과 다른 건 그래서다 — 그 값이 드론이 실제로 갈 곳이다.

---

## 순찰 관련 라우트 (전부 콘솔에 배선돼 있다)

아래 다섯은 `6e8298f` 에서 콘솔에 연결됐다. 각 절이 **어느 함수에 붙어 있는지**
와 응답 규격을 적는다. 프론트를 다시 쓸 때 이 자리들을 유지하면 된다.

### 1. 자연어 → 순찰 구역 · `POST /api/intent`

붙는 자리는 **`드론 관제.dc.html` 의 `runSearch()`**. 방 id 목록을 받아
`setTargets(orderRooms(ids))` 를 부르면 순서·경로·payload 는 전부 기존 코드가
처리한다. 실패하거나 구역을 못 집으면 예전 부분일치 폴백(`searchByName`)으로
내려간다.

```
POST /api/intent   {"text": "2층 전부 순찰해줘"}
  → {"why":  "2층 전체",
     "rooms": ["011","012","014"],  // 방문 순서. 웹 id (3자리)
     "names": {"012": "3층 침실 002_012 (현우방)"},
     "returnHome": true}
```

```js
async runSearch() {
  const q = this.state.q.trim(); if (!q) return;
  const r = await fetch('api/intent', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text: q })
  }).then(r => r.json()).catch(() => null);

  if (!r || !r.rooms || !r.rooms.length) { /* 기존 부분일치 폴백 */ return; }
  this.setTargets(this.orderRooms(r.rooms));
  this.toast(`${r.why} — ${r.rooms.length}개 구역`);
}
```

`rooms` 가 비어 있으면 구역을 못 집은 것이고 `why` 에 이유가 들어온다
("구역을 특정하지 못함"). 그때는 기존 부분일치 폴백을 그대로 쓰면 된다.
빈 `text` 면 **400**. 오프라인 모드(`--llm-url ""`)에서도 이 라우트는 그대로
답한다 — 별칭/`N층`/방 종류로 풀리는 문장은 똑같이 풀리고, 안 풀렸을 때 `why`
뒤에 "(오프라인 모드 …)" 가 붙어 이유가 구분된다.

> **빈 목록이 정상 동작이다.** "냉장고 찾아줘" 처럼 구역과 무관한 문장에도 3B
> 모델은 그럴듯한 방 코드를 채워 넣는다. 서버는 **사용자가 실제로 쓴 말이
> 뒷받침하지 않는 방 코드를 버린다** (방 번호·별칭·방 종류·층 중 하나가 문장에
> 있어야 한다). 그래서 엉뚱한 방 3개를 순찰하는 대신 0개가 돌아온다.

> **별칭이 어긋나 있다.** 콘솔의 하드코딩 `LABELS` 는 `"012"` 를 "채원의 금고가
> 있다는 소문의 방"이라 부르고, `patrol/room_aliases.json` 은 같은 방을
> **현우방**이라 부른다. 자연어는 후자로 해석되므로 "현우방 순찰해줘"가 콘솔에서
> 다른 이름의 방으로 표시된다. `names` 를 같이 보내는 게 그 때문이다 — 표시에
> 이걸 쓰면 어긋남이 사라진다. (`web_meta.json` 의 `rooms[*].label` 은 22개
> 전부 null 이라 자리는 비어 있다.)

### 1-b. 방 목록 · `GET /api/rooms`

별칭을 한 곳에서 받아가는 통로. **아직 콘솔이 안 부른다** — `LABELS` 가 여전히
두 파일에 하드코딩돼 있다. 이걸로 바꾸면 위의 어긋남이 사라진다.

```
GET /api/rooms
  → {"building": "00809_Qpor2mEya8F",
     "rooms": {"012": {"roomName":"002_012", "floor":2, "type":"bedroom",
                       "typeKr":"침실", "display":"3층 침실 002_012 (현우방)",
                       "aliases":["현우방","현우 방"],
                       "center":[2.04,4.89,4.15],
                       "scanPose":[2.04,4.89,4.3]}, ...}}
```

`scanPose` 는 드론이 그 방에서 실제로 호버할 지점이다 — 지도에 순찰 지점을
찍고 싶으면 `center` 말고 이걸 쓰면 실제 비행과 맞는다.

별칭을 고치려면 `patrol/room_aliases.json` 을 편집하면 된다. 이름이 여러 군데
하드코딩돼 있는 것보다 한 파일에 모여 있는 편이 낫다고 보지만, 콘솔이 자기
`LABELS` 를 유지하겠다면 자연어 쪽만 그쪽 이름에 맞춰 `room_aliases.json` 을
고쳐도 결과는 같다. 어느 쪽이든 **한쪽으로 통일만 되면** 된다.

### 2. 순찰 실행 · `POST /api/patrol/start`

붙는 자리는 **`HAUNTED OPS.dc.html` 의 `startMission()`** — 화면 전환만 하던
자리에 `startPatrolFeed()` 가 붙어 있다. 여기가 실제 비행이 시작되는 지점이다.

```
POST /api/patrol/start
  {"targets":[{"room":"012","order":1},{"room":"014","order":2}],
   "returnHome": true,
   "cmd": "2층 전부 순찰해줘"}
  → {"missionId":"20260803_133027", "rooms":["002_012","002_014"], "seq":0}
```

`live` payload 를 그대로 보내도 된다 — 서버는 `targets[].room` 과
`returnHome` 과 `cmd` 만 본다. **배열 순서가 방문 순서**이고 서버는 다시 정렬하지
않는다 (콘솔의 `orderRooms()` 결과를 존중한다).

`cmd` 는 **선택**이고, 사용자가 콘솔에 친 문장이다. 보고서 요약 프롬프트의
`사용자_명령` 으로 들어가므로 이걸 보내면 요약문이 "무엇을 시켰는지"를 안다.
생략하면 예전처럼 `"web console"` 로 적힌다.

| | |
|---|---|
| `409` | 이미 순찰이 돌고 있다 |
| `400` | `targets` 가 비었거나 (`targets 가 비어 있습니다`) 모르는 방 id (`모르는 구역: ...`) |

응답의 `seq` 는 **이 미션의 첫 이벤트 직전 값**이다. 그대로 첫 폴링의 `since`
로 넘기면 `mission_start` 부터 하나도 안 놓친다.

### 3. 진행 로그 · `GET /api/patrol/events?since=<seq>`

붙는 자리는 **`HAUNTED OPS.dc.html` 의 `pollPatrol()`** (1.2초 간격). 보고서
화면이 지어내지 않고 비워둔 값들(방별 도착 시각, 실제 비행 거리)이 여기서 나온다.

```
GET /api/patrol/events?since=0
  → {"seq": 42, "state": "running", "missionId": "...",
     "events": [{"seq":1, "kind":"mission_start", "t":1785730509.6, ...}, ...]}
```

받은 `seq` 를 다음 요청의 `since` 로 넘긴다. 폴링이라 탭이 백그라운드로 가도
놓치는 이벤트가 없다. (콘솔에 WebSocket/EventSource 가 없어서 폴링으로 맞췄다.
1~2초 간격이면 충분하다.)

`state` 는 **`idle` | `running` | `done` | `error`** 넷이다. 콘솔은
`state !== 'running'` 이면 폴링을 멈춘다 — 마지막 응답에 `mission_end` 와
`report_ready` 가 같이 실려 오므로 하나 더 돌 필요가 없다.

`seq` 는 **미션이 바뀌어도 리셋되지 않는다.** 새 순찰이 시작되면 이전 이벤트는
버려지지만 번호는 계속 올라가므로, 폴링 중이던 클라이언트가 커서를 되감는 일이
없다. 로그는 최근 **500건**만 들고 있다 (1~2초 폴링이면 넘칠 일이 없다).

| `kind` | 실린 것 |
|---|---|
| `mission_start` | `rooms[{room,display,floor}]`, `returnHome` |
| `leg_start` | `room`, `display`, `order`, `of` |
| `arrived` | `room`, `display`, `legMeters` |
| `leg_failed` | `room`, `reason` |
| `scan_start` | `room`, `display` |
| `scan_done` | `room`, `degrees`, `detections`, `completed` |
| `detect` | `room`, `display`, `n`, `label`, `conf`, `box{l,t,w,h}`, `image` |
| `returning` / `landed` | — |
| `mission_end` | `flownMeters`, `durationSec`, `roomsReached`, `roomsPlanned`, `detections`, `collisions`, `returnedHome`, `abortedReason` (§4 — 사용자 중단은 안 채운다) |
| `report_ready` | `missionId` — **콘솔이 이걸 받고 §5 를 부른다** |
| `abort_requested` | — `POST /api/patrol/abort` 를 받은 순간 |
| `status` | `text` — 사람이 읽는 한국어 한 줄. 화면에 그대로 흘려도 된다 |
| `error` | `message` — 미션이 예외로 죽었다. `state` 도 `error` 가 된다 |

`room` 은 **우리 표기**(`"002_012"`)다. 콘솔 id 로는 `room.split('_').pop()`.

**`detect` 는 그대로 `__patrolDetect` 에 넘기면 된다** — `box` 가 이미 프레임
대비 % 다. Unity 가 자기 캡처 해상도로 나눠서 보내므로 중간에 아무도 환산하지
않는다.

```js
// 폴링 루프 안
for (const e of ev.events) {
  if (e.kind === 'detect') {
    window.__patrolDetect({
      label: e.label === 'person' ? undefined : e.label,  // 생략하면 "생존자 N"
      room: e.display, conf: e.conf * 100, box: e.box,    // photo 는 생략 →
    });                                                   // 화면 캡처가 채운다
  } else if (e.kind === 'arrived') {
    /* 타임라인에 방 도착 시각 */
  } else if (e.kind === 'mission_end') {
    this.finishPatrol();            // 운용자 버튼 대신 실제 완료로
  }
}
```

`conf` 는 0..1 이고 콘솔은 % 로 표시하므로 100을 곱한다.

### 4. 중단 · `POST /api/patrol/abort`

```
POST /api/patrol/abort → {"ok": true}
                       → {"ok": false, "reason": "진행 중인 순찰이 없습니다"}
```

스캔을 멈추고 rc 를 0 으로 두고 착륙시킨다. 붙는 자리는 **`abortPatrol()`**
(「작전 중단」 버튼). 순찰이 안 돌고 있어도 **HTTP 는 200** 이므로 `ok` 를 봐야
한다 — 화면만 전환하는 경우에도 부르게 돼 있어서 이 편이 다루기 쉽다.

> **중단은 즉시 끝나지 않고, `abortedReason` 도 안 채운다.** 미션 스레드는
> 중단 요청을 직접 보지 않는다 — 드론을 착륙시켜 남은 구간을 실패시키는
> 방식이라, `abort_requested` 뒤에 `leg_failed` → `returning` → `landed` →
> `mission_end` 가 몇 초에 걸쳐 이어진다. 그리고 그 `mission_end` 의
> `abortedReason` 은 **빈 문자열**이다 (그 필드는 `simulator_unreachable` /
> `state_lost` / 예외에만 찬다). 사용자가 중단했다는 사실은 부른 쪽이 알고
> 있으므로 이벤트로 되받을 필요가 없다 — 다만 `mission_end` 만 보고
> "정상 완료" 로 판단하면 안 된다.

### 5. 기록 · `GET /api/patrol/report/<missionId>`

`report.json` 을 그대로 돌려준다. 붙는 자리는 **`HAUNTED OPS.dc.html` 의
`fetchReport()`** — `onPatrolEvent()` 가 `report_ready` 를 보면 한 번 부른다.
진행 이벤트에는 요약도 실측치도 없으므로 **이 한 번이 그 둘의 유일한 통로**다.

콘솔이 쓰는 키는 셋이다:

| 키 | 쓰는 곳 |
|---|---|
| `summary` | 보고서 화면 좌측 「AI 요약」 패널, 순찰 완료 화면 우측 패널. `patrol_report._llm_summary()` 가 쓴 한국어 3~5문장 |
| `facts` | 개요 표 — `비행_거리_m`(실측), `소요_시간_초`, `복귀_완료`, `충돌_횟수`, `미도달_구역`, `중단_사유`, `탐지_건수` |
| `rooms[]` | 구역별 결과 — `room_name`(뒤 3자리가 콘솔 id), `reached`, `reason`, `scan_degrees`, `events` |

> **여기 숫자가 화면의 계획값과 다른 게 정상이다.** 브리핑의 「비행 거리」는 A\* 가
> 뽑은 계획 경로 길이(`live.dist`)고 `facts.비행_거리_m` 는 실제로 난 거리다.
> 콘솔은 보고서가 오면 실측치로 덮고, 못 받으면 계획값 그대로 그린다 — 요약문과
> 표가 서로 다른 얘기를 하지 않도록.

> **요약은 LLM 이 없어도 나온다.** 오프라인이거나 모델이 죽으면 서버가
> `_fallback_summary()` 템플릿 문장으로 채워 보내므로 `summary` 가 비는 일은 없다.
> 콘솔은 둘을 구분하지 않는다 (구분이 필요하면 `GET /api/status` 의 `llmOffline`).

「기록」 화면의 하드코딩된 6건을 실제 기록으로 바꿀 때도 이걸 쓴다.

> **가장 최근 순찰 하나만 꺼낼 수 있다.** 서버는 마지막 보고서 폴더만 들고
> 있어서, `missionId` 가 현재 미션과 다르면 **404** 다. 지난 기록 목록이
> 필요하면 `patrol/out/reports/` 를 훑는 라우트를 따로 열어야 한다.

---

## 아직 비어 있는 것

- **탐지 사진.** 웹이 화면 공유 프레임을 떠서 쓰는 게 기본이고, 공유가 꺼져
  있으면 `NO FRAME` 이 된다. Unity 가 저장한 파일 경로(`image`)는 로컬 절대
  경로라 브라우저가 못 읽는다 — 필요하면 그 폴더를 정적 서빙하는 라우트를 하나
  더 열어야 한다.
- **`scan` verb 를 구현한 Unity 빌드 + 씬 안의 사람.** 배관은 다 있다
  (`PatrolPersonDetection.cs` → TCP 9100 `person_detector_tcp.py` → `detect`).
  사람은 `test.unity` 의 NPC 프리팹 3개뿐이고 **그 에셋은 저장소에 없다**
  (README §8). **`detect` 가 한 건도 안 오는 경우는 둘 — NPC 에셋을 안 받았거나
  YOLO 프로세스를 안 띄웠거나**이고, 둘 다 결과는 빈 집과 같다(전 방 "아무도
  없음"). 서버가
  `scan` 무응답을 감지하면 rc 회전으로 내려가는데 그 경로에는 탐지가 없다
  (외부 디텍터를 UDP 9004 로 띄우면 있다). 웹 배선만 확인하려면
  `simulator/bridge/fake_unity_sim.py --detect-per-scan 1` 을 쓴다.
- **경로 엔진.** `/plan` 은 지금 A\* 다. 콘솔이 기대하던 SAC 정책(`rl_planner`)과
  통일할지는 아직 결정 전 — 응답 `engine` 필드가 어느 쪽이 답했는지 알려준다.
- **`GET /api/rooms` 를 콘솔이 안 쓴다.** `LABELS` 하드코딩이 두 파일에 남아
  있어 **표시되는** 방 이름이 자연어 쪽과 어긋난 채다 (§1-b). 탐지를 방에 되짚는
  쪽은 `detect` 의 `room`("002_012") 뒤 3자리로 맞추도록 고쳐서 이름 어긋남의
  영향을 안 받는다 — 보고서의 요약문·구역별 결과에 뜨는 **이름**만 두 체계가
  섞여 있다.
- **지난 순찰 기록.** 최근 하나만 꺼낼 수 있다 (§5). 그래서 새 순찰을 시작하면
  이전 보고서는 못 꺼낸다 — 콘솔도 `startMission()` 에서 `report` 를 비운다.
- **중단한 순찰의 요약.** 서버는 중단된 미션도 보고서를 쓰지만, 콘솔의
  `abortPatrol()` 이 폴링을 즉시 끊어 `report_ready` 를 못 본다. 그래서 「순찰
  중단」 화면에는 요약이 없다 (그 화면 문구도 "보고서가 생성되지 않았습니다"인
  채다).
