# 순찰 에이전트 (LLM agent) — 경로 계획 · 탐지 반응 · 보고서

`patrol` 패키지의 **순찰 모드**. 자연어 명령으로 방을 지정하면 드론이 그 방까지
날아가 360° 스캔하고, 2D detection이 사람을 찾으면 정지 → 라이트 온 → 사진 기록
→ 알림 후 복귀하며, 마지막에 순찰 보고서를 생성한다.

담당 분리:

| 부분 | 담당 | 위치 |
|---|---|---|
| 경로 계획 / 순찰 실행 / 탐지 반응 / 보고서 | 이 문서 | `patrol/patrol_*.py` |
| **2D person detection 모델** | 팀원 | 별도 프로세스 |
| **Unity 드론 카메라 → Python 사진 전송(TCP)** | 팀원 | Unity C# + 별도 프로세스 |

---

## 1. 팀원용 계약 — 탐지 이벤트 (UDP 9004)

detector 프로세스는 사람을 탐지할 때마다 **JSON 한 덩어리를 UDP로 1개** 보낸다.
그게 전부다. 응답을 기다릴 필요도, 연결을 유지할 필요도 없다.

```python
import json, socket, time
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.sendto(json.dumps({
    "label":      "person",                  # 필수
    "conf":       0.87,                      # 선택 0..1
    "bbox":       [x, y, w, h],              # 선택, 이미지 픽셀
    "image_path": "/abs/path/evt_0031.jpg",  # 선택(강력 권장) — 촬영한 사진
    "ts":         time.time(),               # 선택
    "source":     "yolo26n",                 # 선택
}).encode(), ("127.0.0.1", 9004))
```

- **`label`만 필수.** 나머지가 없어도 이벤트는 기록된다.
- `image_path`는 **에이전트가 읽을 수 있는 절대 경로**여야 한다(같은 머신 또는
  공유 마운트). 파일은 보고서 폴더로 **복사**되므로, 보낸 뒤 지워도 된다.
  없으면 "사진 없음"으로 기록되고 순찰은 계속된다.
- `label` 필터/최소 신뢰도/중복 억제(cooldown)는 서버 쪽 인자로 조절한다:
  `--patrol-labels person --patrol-min-conf 0.5 --patrol-cooldown 3.0`

### 언제 보내야 하나 — 게이팅

에이전트는 **드론이 순찰 구역 안에서 스캔 중일 때만** 이벤트를 받아들인다
(플로우의 "이동 중에는 탐지하지 않고 지정 구역에서만"). 이동 중에 도착한
이벤트는 버려지고 통계(`ignored_in_transit`)에만 잡힌다. detector는 이 상태를
알 필요가 없다 — **계속 보내도 된다.** 게이팅은 에이전트가 한다.

### 사진은 누가 찍나

이벤트에 `image_path`를 실어 보내는 방식으로 합의됨. 즉 **탐지한 쪽이 그 프레임을
저장하고 경로를 알려준다.** 에이전트는 별도의 촬영 명령을 보내지 않는다.
(에이전트가 `capture`를 요청하는 방식이 필요해지면 `detect_events.py`에 요청
채널을 추가하면 되지만, 지금은 불필요하다.)

### 테스트

detector 없이 반응 시퀀스를 확인하려면:

```bash
python -m patrol.detect_events --emit --label person --conf 0.9 \
    --image /path/to/any.jpg
# 수신만 확인:
python -m patrol.detect_events            # 9004에서 armed 상태로 대기
```

---

## 2. 순찰 시퀀스

```
사용자: "현우방만 탐색해줘"
   │
   ├─ patrol_intent.parse_patrol   LLM(Qwen) → mode=patrol
   ├─ patrol_intent.resolve_rooms  별칭 "현우방" → 002_012
   ├─ room_index.order_rooms       여러 방이면 층 우선 + 최근접 순서
   │
   ├─ Unity preview + [이동] 확인   (--no-patrol-confirm 이면 건너뜀)
   │
   ├─ takeoff                       ← 미션 전체에서 딱 1회
   │  for room in rooms:
   │      planner.plan_path  →  follow_path.follow_path   (detector DISARMED)
   │      listener.arm(room)
   │      scan_360:  rc 0 0 0 <yaw>, 실제 yaw 누적으로 360° 확인
   │          탐지 이벤트 도착 시:
   │              rc 0 0 0 0        1. 정지(호버)
   │              light on          2. 라이트 온
   │              settle 0.8s       3. 조명 반영 대기
   │              사진 복사          4. events/evt_NNN.jpg
   │              배너 + 터미널 알림 5. 알림
   │              light off, 스캔 재개
   │      listener.disarm()
   │  복귀 경로 → land
   │
   └─ patrol_report.build_report → out/reports/<ts>_patrol/
```

`follow_path.fly_mission`이 아니라 저수준 `follow_path.follow_path`를 쓰는 이유:
`fly_mission`은 호출마다 `takeoff`하고 `finally`에서 `land`하는데,
`TelloSimulator.cs`는 `isFlying`일 때만 rc(yaw 포함)를 적용한다. 방마다 착륙하면
360° 스캔이 조용히 아무것도 하지 않게 된다.

## 3. 실행

```bash
source /data1/workspaces/jgshin22/miniconda3/etc/profile.d/conda.sh
conda activate patrol           # 또는 unidet3d (README.md §1 — numpy<2 필요)

python patrol/server.py \
    --sim --unity-host <UNITY-IP> --llm-device cuda:0 \
    --patrol-port 9004 --patrol-labels person \
    --viz-dir simulator/tello_simulator/Assets/Resources

query> 현우방만 탐색해줘        # 순찰 + 보고서
query> 2층 전부 순찰해줘
query> 거실 소파 찾아줘         # 기존 물체 찾기 (동작 불변)
query> rooms                    # 순찰 가능한 방 목록
query> report                   # 마지막 순찰 보고서 재생성
```

주요 인자: `--hover-height 1.2`(스캔 고도, 방 바닥 기준 m),
`--scan-deg-per-sec 50`, `--scan-turns 1`, `--max-rooms 12`(집 전체 순찰 상한),
`--no-light`(라이트 verb 끄기), `--no-patrol-confirm`(확인 없이 바로 출발),
`--room-aliases`(별칭 파일 경로), `--report-dir`.

### 방 별칭

LitePT 데이터에는 `002_012` 같은 코드와 `bedroom` 같은 타입만 있다. "현우방"은
**`patrol/room_aliases.json`에만** 존재하므로, 사람 이름 방을 쓰려면 이
파일을 편집해야 한다. 별칭은 LLM 프롬프트의 방 목록에 주입되고, LLM이 놓쳐도
원문 부분 문자열 매칭으로 다시 잡힌다.

```json
{"floor_offset": 1,
 "aliases": {"002_012": ["현우방"], "002_014": ["규철방"]}}
```

`floor_offset`: 폴더 접두사 `002_`는 층 인덱스 2이고, 화면/명령에서는
`2 + floor_offset` = **3층**으로 표시된다.

## 4. 보고서

`out/reports/<YYYYmmdd_HHMMSS>_patrol/`

| 파일 | 용도 |
|---|---|
| `report.md` | 상대경로 이미지. git/PR 리뷰용 |
| `report.html` | 이미지 base64 내장(600 KB 초과 시 JPEG 축소). 브라우저로 바로 열림 |
| `report.json` | 기계 판독용. 웹 UI가 렌더링할 수 있는 형태 |
| `events/evt_NNN.jpg` | 탐지 시점에 복사된 사진 |

내용: 날짜 · 순찰 구역 · 총 탐지 건수 · 이벤트별 상세(시각/방/신뢰도/드론 좌표/
사진) · 구역별 결과 표 · 비행 통계. **요약 문단만 LLM이 쓰고**, 나머지는 결정적
으로 생성된다. LLM 출력이 비거나 깨지면 템플릿 문장으로 대체되므로 보고서가
생성되지 않는 경우는 없다.

## 5. Unity 쪽 남은 작업 (팀원)

### (a) `light` verb — 6줄

에이전트는 탐지 시 `light on` / `light off`를 UDP 9000으로 보낸다. Unity는 현재
미지 명령을 로그만 남기고 `ok`로 ack하므로 **지금도 무해**하고, 아래 분기를
`TelloSimulator.ProcessCommand`에 추가하면 즉시 동작한다:

```csharp
if (cmd == "light on" || cmd == "light off")
{
    var horror = FindObjectOfType<HorrorAtmosphere>();
    if (horror != null) horror.SetFlashlight(cmd.EndsWith("on"));
    return;
}
```

`HorrorAtmosphere`에는 아직 public setter가 없으므로 (`flashlightOn` 필드와 `F`
키 토글만 있음) 다음도 함께 필요하다:

```csharp
public void SetFlashlight(bool on)
{
    flashlightOn = on;
    if (flashlight != null) flashlight.enabled = horrorEnabled && on;
}
```

### (b) 화면 카메라 회전 — **적용 완료**, 촬영 카메라는 별도

`CameraFollow.cs`의 heading은 드론의 회전이 아니라 **속도 벡터**에서 계산되므로,
원래는 제자리 360° yaw 회전 시 드론 본체만 돌고 화면은 정지해 있었다.
`LateUpdate`에 분기를 추가해 해결했다 — **호버 중이면서 실제로 회전 중일 때**
드론의 `target.forward`를 heading으로 따라간다:

```csharp
else if (followYawWhileHovering && Mathf.Abs(yawRate) > minYawRate)
{
    Vector3 forward = target.forward; forward.y = 0f;
    if (forward.sqrMagnitude > 0.001f)
        heading = Vector3.Slerp(heading, forward.normalized,
                                Time.deltaTime * headingSmoothSpeed);
}
```

"호버 중"만이 아니라 **yaw rate**로 게이팅한 이유: 드론은 yaw를 0으로 고정한 채
비행하므로(`follow_path`가 `rc … 0` 송신), 단순히 정지 상태에서 `target.rotation`을
채택하면 **웨이포인트에 도착해 멈출 때마다 카메라가 휙 돌아간다.** 인스펙터 필드
`followYawWhileHovering`(끄면 이전 동작), `minYawRate`(기본 5 deg/s)로 조절한다.
`setpos` 텔레포트 프레임은 yaw 변화를 함께 무시한다.

> 이 C# 변경은 **Unity 컴파일러로 검증하지 못했다** (개발 서버에 Unity/csc 없음).
> Play 모드에서 한 번 확인해 주기 바란다. 확인 포인트: 스캔 중 **C 키(1인칭)**로
> 보면 시야가 방을 훑고, 이동 중 웨이포인트 도착 시 카메라가 튀지 않아야 한다.

촬영/탐지용 카메라는 이와 별개로 **드론 transform의 자식**으로 새로 다는 것을
권장한다 — 화면 카메라(`CameraFollow`)는 3인칭에서 드론 주위를 공전하므로
드론이 보는 화면과 다르다.

### (c) 경로 시각화는 이미 있다

`PlannedPathRenderer.cs` / `FlightReportRenderer.cs`는 이미 씬에 있고
`Assets/Resources/planned_path_3d.json`, `flight_trajectory_3d.json`을 1초마다
폴링한다. 서버를 `--viz-dir simulator/tello_simulator/Assets/Resources`로 띄우면
**C# 수정 없이** 순찰 경로와 탐지 지점(빨간 구)이 씬에 그려진다.
두 파일은 Unity 좌표계 + `{x,y,z}` 객체 배열이어야 한다(JsonUtility 제약).

## 6. 헤드리스 테스트 (Unity 없이)

```bash
# 1) Unity 대역 프로토콜 스텁
python simulator/bridge/fake_unity_sim.py --verbose --auto-confirm-sec 3

# 2) 서버
python patrol/server.py --sim --unity-host 127.0.0.1 \
    --llm-device cuda:0
query> 현우방만 탐색해줘

# 3) 스캔 중에 다른 셸에서 사람 탐지 흉내
python -m patrol.detect_events --emit --label person --conf 0.9 \
    --image image.png --repeat 5 --interval 2
```

확인 사항:
- `fake-sim` 로그에 `light on` → `light off` 쌍이 탐지 건수만큼 찍힌다
- **이동 중**에 emit하면 기록되지 않고 `ignored_in_transit`만 증가한다
- 스캔이 `360°`로 완료된다(`scan_degrees`)
- `out/reports/<ts>_patrol/report.html`이 열리고 사진이 보인다
