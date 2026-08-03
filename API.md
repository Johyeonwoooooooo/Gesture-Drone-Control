# API — 웹 콘솔 ↔ 순찰 백엔드

`api_server.py` 가 노출하는 계약. 웹 콘솔(`web/`, HAUNTED OPS)이 붙을 자리다.

```bash
# 로컬 PC (Unity 옆). 한 포트에 web/ 정적 + API 가 같이 뜬다.
python api_server.py --port 8123 --llm-url http://<GPU서버>:8000/v1 --llm-api-key <토큰>
```

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
| `GET /api/status` | `{ready, engine, building, rooms, model, mission:{state,id,busy,seq}}` |
| `POST /plan` | `{start?, goal}` → `{engine, success, steps, bumps, dist, flown, ms, start, goal, path}` |
| `POST /api/intent` | 자연어 → 방 id 목록 (§1) |
| `GET /api/rooms` | 방 22개 + 별칭 + 스캔 포즈 (§1-b) |
| `POST /api/patrol/start` | 순찰 시작 (§2) |
| `GET /api/patrol/events` | 진행 로그, `since` 커서 (§3) |
| `POST /api/patrol/abort` | 중단 (§4) |
| `GET /api/patrol/report/{id}` | 보고서 JSON (§5) |
| `GET /` | `HAUNTED OPS.dc.html` 로 리다이렉트 |
| `GET /<path>` | `web/` 정적 |

`/plan` 응답 키는 기존과 같다. `path` 는 3D 궤적, `goal` 은 **실제로 쓰인
(스냅된)** 좌표라 다음 구간의 `start` 로 그대로 넘기면 된다. 경로를 못 찾으면
`{"success": false, "error": "<이유>", "path": []}` 이고 HTTP 는 200이다 —
**`success` 를 봐야 한다.**

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
빈 `text` 면 **400**.

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
   "returnHome": true}
  → {"missionId":"20260803_133027", "rooms":["002_012","002_014"], "seq":0}
```

`live` payload 를 그대로 보내도 된다 — 서버는 `targets[].room` 과
`returnHome` 만 본다. **배열 순서가 방문 순서**이고 서버는 다시 정렬하지 않는다
(콘솔의 `orderRooms()` 결과를 존중한다).

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
| `report_ready` | `missionId` |
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

`report.json` 을 그대로 돌려준다. 콘솔은 이미 자기 화면에서 보고서를 그리므로
아직 안 부른다. 「기록」 화면의 하드코딩된 6건을 실제 기록으로 바꿀 때 쓴다.

> **가장 최근 순찰 하나만 꺼낼 수 있다.** 서버는 마지막 보고서 폴더만 들고
> 있어서, `missionId` 가 현재 미션과 다르면 **404** 다. 지난 기록 목록이
> 필요하면 `patrol/out/reports/` 를 훑는 라우트를 따로 열어야 한다.

---

## 아직 비어 있는 것

- **탐지 사진.** 웹이 화면 공유 프레임을 떠서 쓰는 게 기본이고, 공유가 꺼져
  있으면 `NO FRAME` 이 된다. Unity 가 저장한 파일 경로(`image`)는 로컬 절대
  경로라 브라우저가 못 읽는다 — 필요하면 그 폴더를 정적 서빙하는 라우트를 하나
  더 열어야 한다.
- **`scan` verb 를 구현한 Unity 빌드.** 아직 `PatrolPersonDetection.cs` 가 이
  브랜치에 없다. 그 전까지 서버는 `scan` 무응답을 감지해 rc 회전으로 내려가고,
  그 경로에는 탐지가 없다(외부 디텍터를 UDP 9004 로 띄우면 있다). **실제
  Unity로 돌리면 `detect` 가 한 건도 안 온다** — 배선을 확인하려면
  `simulator/bridge/fake_unity_sim.py --detect-per-scan 1` 을 쓴다.
- **경로 엔진.** `/plan` 은 지금 A\* 다. 콘솔이 기대하던 SAC 정책(`rl_planner`)과
  통일할지는 아직 결정 전 — 응답 `engine` 필드가 어느 쪽이 답했는지 알려준다.
- **`GET /api/rooms` 를 콘솔이 안 쓴다.** `LABELS` 하드코딩이 두 파일에 남아
  있어 방 이름이 자연어 쪽과 어긋난 채다 (§1-b).
- **지난 순찰 기록.** 최근 하나만 꺼낼 수 있다 (§5).
