# 🚁📦 Cargo Simulation — 커스텀 오브젝트 배치 + 드론 운반 상호작용

기존 Tello 시뮬레이터 환경 위에 **커스텀 물건을 배치하고, 드론이 잡아서(grab) 운반하고,
배달 구역(Drop Zone)에 내려놓는(drop)** 상호작용 시뮬레이션입니다.

**씬 파일을 수정하지 않습니다.** Play를 누르면 `SimPropManager`가
`RuntimeInitializeOnLoadMethod`로 자동 부트스트랩되어 오브젝트를 배치하고
드론에 카고 기능(`DroneCargo`)을 붙입니다.

---

## 구성 파일

| 파일 | 역할 |
|---|---|
| `Assets/CargoSim/SimPropManager.cs` | 오브젝트 배치 총괄 + UDP 명령 처리 (자동 부트스트랩) |
| `Assets/CargoSim/CarryableProp.cs` | 운반 가능한 화물 (Rigidbody 물리, 낙하 안전망 포함) |
| `Assets/CargoSim/DropZone.cs` | 배달 구역 패드 — 화물이 안에서 정지하면 배달 완료 처리 |
| `Assets/CargoSim/DroneCargo.cs` | 드론의 grab/drop 훅 + 운반 테더(줄) 시각화 |
| `Assets/StreamingAssets/cargo_layout.json` | **커스텀 배치 설정** (이 파일만 고치면 배치 변경) |
| `../digital_twin_test/delivery_mission.py` | 자율 배달 미션 데모 (Python) |
| `../digital_twin_test/python_rc_test.py` | 수동 조종기 (G:잡기 / H:놓기 / B:박스 생성 키 추가) |

`TelloSimulator.cs`에는 확장 훅 2개(`commandHook`, `stateExtraProvider`)와
`SendJson()`만 추가되었고 기존 동작은 그대로입니다.

---

## 실행 방법

### 1) 수동 조종으로 운반해 보기
1. Unity에서 `tello_simulator` 프로젝트 열기 → **Play**
   (콘솔에 `[Cargo] Simulation ready: 3 prop(s), 1 zone(s)` 로그 확인)
2. ```bash
   cd playground/digital_twin_test
   python python_rc_test.py
   ```
3. `T` 이륙 → 화물(주황/파랑 박스, 보라 배럴) 위로 접근 → `G` 잡기
   → 초록 패드 위로 이동 → `H` 놓기 → 패드 안에 안착하면 **배달 완료**

### 2) 자율 배달 미션 (전 과정 자동)
1. Unity **Play** 상태에서:
   ```bash
   cd playground/digital_twin_test
   python delivery_mission.py
   ```
2. 드론이 스스로: 이륙 → 화물 위치 조회(`objects`) → 화물별로
   접근 → grab → 배달구역 비행 → drop → 전부 배달 후 착륙 + 미션 리포트 출력

---

## 커스텀 배치 바꾸기 — `cargo_layout.json`

```json
{
  "relativeToDrone": true,
  "props": [
    { "name": "package_orange", "shape": "box", "x": 1.5, "y": 0.4, "z": 3.0,
      "scale": 0.35, "color": "#FF8C26", "mass": 0.4 }
  ],
  "zones": [
    { "name": "dropzone", "x": 0.0, "y": 0.0, "z": 8.5, "radius": 1.5, "color": "#26FF73" }
  ]
}
```

- `relativeToDrone: true`면 좌표가 **드론 시작 포즈 기준** (`+z` = 드론이 바라보는 방향).
  `false`면 월드 좌표.
- `shape`: `box` | `sphere` | `barrel`
- 화물은 스폰 후 물리로 바닥에 안착하고, 배달 패드는 아래 바닥면에 자동 스냅됩니다.
- 파일이 없거나 파싱에 실패하면 내장 기본 배치가 사용됩니다.

---

## UDP 명령 (기존 포트 9000 그대로)

| 명령 | 동작 |
|---|---|
| `grab` | 반경 2.5m 내 가장 가까운 화물을 집어 드론 아래에 매닮 |
| `drop` | 들고 있는 화물을 놓음 (물리 낙하) |
| `objects` | 상태 포트(9002)로 화물/구역 목록 JSON 응답 |
| `spawn <shape> <x> <y> <z> [scale] [rel]` | 화물 즉석 생성. `rel`이면 드론 로컬 좌표 |
| `clearobjects` | 스폰된 화물 전부 제거 |

`objects` 응답 예시:
```json
{"type":"objects",
 "props":[{"name":"package_orange","shape":"box","x":-7.6,"y":0.35,"z":-33.0,
           "carried":false,"delivered":false}],
 "zones":[{"name":"dropzone","x":-2.1,"y":0.02,"z":-34.5,"radius":1.50}]}
```

기존 상태 스트림(9002, 20Hz)에는 필드가 추가되었습니다:
```json
{"x":..,"y":..,"z":..,"yaw":..,"flying":true,
 "had_collision":false,"collision_count":0,
 "carrying":"package_orange","delivered":1,"props_total":3,"time":12.3}
```

---

## 동작 세부

- **잡기**: 화물은 드론 1.1m 아래에 매달리며 노란 테더 라인으로 표시.
  운반 중에는 화물 콜라이더가 꺼져 드론 충돌 판정을 오염시키지 않습니다.
- **배달 판정**: 화물이 배달 패드 트리거 안에서 **정지**하면 완료.
  화물 라벨이 `이름 [OK]`로 바뀌고 색이 구역 색으로 물듭니다.
- **HUD**: 좌측 상단 기존 텔레메트리 아래에 `[Cargo] Carrying / Delivered` 표시.
- 배달된 화물을 다시 잡으면 배달 상태가 해제됩니다 (상태는 항상 현재 배치를 반영).
- 화물이 맵 밖으로 떨어지면(-20m 이하) 원래 자리로 리스폰됩니다.
