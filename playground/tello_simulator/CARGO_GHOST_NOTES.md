# 화물 운반 개선 + 유령 자유 배회 (CargoSim)

## 1. "선에 매달려 데롱데롱" 원인

화물이 흔들리며 매달려 보인 것은 물리 로프/조인트 때문이 아니라 **의도적으로 그렇게 그려진 윈치(sling-load) 연출** 때문이었다. 원인은 세 가지:

| 원인 | 위치 | 내용 |
|------|------|------|
| ① 노란 줄(테더) | `DroneCargo.cs` (구버전) | `LineRenderer`로 드론→화물 사이에 노란 선을 매 프레임 그림 = 눈에 보이는 "선" |
| ② 1.1m 아래 오프셋 | `DroneCargo.carryDistance = 1.1f` | 화물을 드론 **중심에서 1.1m 아래**에 배치 → 공중에 매달린 것처럼 보임 |
| ③ 요(yaw)만 정렬 | `CarryableProp.Attach` | 화물을 월드 기준 수직으로 세워 매달아 슬링로드 느낌 강조 |

즉 화물은 사실 `SetParent` + `isKinematic`으로 드론에 **강체로 부모-자식 고정**되어 있었지만(물리적으로 안 흔들림), 1.1m 간격 + 노란 줄 때문에 "줄에 매달려 데롱데롱"하는 것처럼 **보였던** 것.

## 2. 수정: 드론에 딱 붙어서 이동 + 깔끔하게 내려놓기

### `DroneCargo.cs`
- **테더(LineRenderer) 완전 제거** — 더 이상 줄이 안 그려짐.
- `grab` 시 화물을 **드론 배면에 딱 붙게 클램프**. 드론 모델의 실제 밑면 높이를 렌더러 바운즈로 측정하고, 화물의 윗면이 그 밑면에 닿도록 위치를 계산(`ComputeMountPosition`). `clampClearance`(기본 0.02m)만큼만 띄움.
- 여전히 부모-자식 + 키네마틱이라 **드론과 완전히 한 몸으로 이동**(흔들림/지연 0).

### `CarryableProp.cs`
- `Attach`: 드론 헤딩에 맞춰 세운 채 강체 고정(주석 보강).
- `Release` → **`PlaceOnSurfaceBelow`** 추가: 놓을 때 바로 아래 표면(바닥/패드)을 레이캐스트로 찾아 화물을 그 위에 **똑바로 스냅해서 내려놓음**. 속도 0으로 두어 튕기거나 날아가지 않음. = "내려놓기"가 정확해짐. (드론/자기 콜라이더는 무시)

### `delivery_mission.py`
- 드롭 고도를 `zone.y + 1.8` → `zone.y + 1.2`로 낮춤(화물이 더 이상 1.1m 아래에 매달리지 않으므로). 주석 갱신.

## 3. 유령(Ghost) 자유 배회

### `GhostRoamer.cs` (신규)
- 반투명 "유령"이 스캔된 씬을 **스스로 자유 배회**.
- NavMesh가 없어서 **레이캐스트 기반**으로 동작: 바닥 위 일정 높이 유지, 전방 스피어캐스트로 벽/가구 회피, 주기적으로 새 방향 선택, 스폰 지점 반경(`wanderRadius`) 밖으로 나가면 복귀. 살짝 상하로 떠다니는 bob 연출.
- **콜라이더 없음** → 드론 충돌 카운트에 안 걸리고, 드론이 집어들 수도 없음(순수 배회 존재).

### `SimPropManager.cs`
- 새 UDP 명령:
  - `ghost [count]` — 유령 임포트(자유 배회 시작). 반투명 청백색 구체(외피 + 코어) + "유령" 라벨.
  - `clearghosts` — 유령 전부 제거.

## 3-b. 유령 동적 회피 (앞의 움직이는 물체 피하기)

### `DroneAvoidance.cs` (신규)
- 비행 중 매 프레임, 감지 반경(`safeDistance`, 기본 4m) 안의 **유령(GhostRoamer)** 을 찾아 드론을 **옆으로+위로 비켜** 회피시킴. 정면에 있을수록(`frontBias`) 더 강하게 반응해 "브레이크"가 아니라 "돌아서 지나감".
- `TelloSimulator`의 rc 이동(Update) 이후 `LateUpdate`에서 `CharacterController.Move`로 회피 변위를 **덧씌움** → 수동 조종·자율 미션 양쪽에서 동작.
- 유령은 콜라이더가 없어 물리로 안 막히므로, 위치 기반 능동 회피로 처리(충돌 카운트 오염 없음). 회피 중엔 화면에 주황색 `[Avoid] EVADING ...` 표시.
- `TelloSimulator.IsFlying` 프로퍼티를 추가해 착륙 상태에선 회피가 드론을 밀지 않도록 함.

### `SimPropManager.cs`
- 새 UDP 명령: `avoid on` / `avoid off` — 회피 어시스트 토글(기본 ON). `Start`에서 드론에 `DroneAvoidance` 자동 부착.
- 상태 스트림(20Hz)에 유령 위치 `"ghosts":[{x,y,z},...]` 추가(`GhostsJson`) → 자율 제어기가 읽어서 우회.

### 회피는 두 겹으로 동작
| 계층 | 위치 | 언제 | 역할 |
|------|------|------|------|
| 유도(퍼텐셜 필드) | `delivery_mission.py` `ghost_avoidance()` | **자율주행 시** | goto가 목표 인력 + 유령 반발을 합쳐 rc 계산 → 이동 중 유령을 크게 우회(감지 5m). 화물 접근/투하 하강은 `avoid=False`로 제외 |
| 반사(안전망) | `DroneAvoidance.cs` | 수동/자율 공통 | rc 이동 뒤 근접(4m) 유령을 마지막 순간에 밀어냄 |

즉 **자율 배달 미션은 유도 계층이 주도**해 유령을 미리 돌아가고, Unity 반사 계층은 최후 안전망으로 남는다. `delivery_mission.py`는 시작 시 유령 2기를 자동 임포트해 회피를 바로 시연한다.

## 4. 사용법

Unity에서 `tello_simulator`(`Assets/test.unity`) 열고 **Play**. 그다음:

### 수동 조종 (`playground/digital_twin_test/python_rc_test.py`)
```
python python_rc_test.py
```
- `T` 이륙 → `WASD/RF/QE` 비행
- `B` 앞에 박스(물체) 생성 → `G` 잡기(**드론에 딱 붙음**) → 이동 → `H` 놓기(**바닥에 내려놓음**)
- `N` 유령 임포트(자유 배회) / `M` 유령 제거
- `V` 유령 회피 어시스트 On/Off — `N`으로 유령 여러 개 띄우고 그쪽으로 `W` 전진하면 드론이 알아서 비켜감

### 자율 배달 미션 (`delivery_mission.py`)
```
python delivery_mission.py
```
화물을 자동으로 픽업(딱 붙임)해 배달구역에 내려놓음. 시작 시 유령 2기를 임포트하고,
이동 구간에서 유령을 **스스로 우회**하며 비행한다(회피 세기: `GHOST_AVOID_RADIUS/GAIN`).

## 5. 조정 포인트 (Inspector / 필드)
- `DroneCargo.clampClearance` — 드론-화물 간격(0이면 완전 밀착).
- `GhostRoamer.moveSpeed / turnSpeed / hoverHeight / wanderRadius / lookAhead` — 유령 배회 성향.
- 유령의 시각(색/투명도/개수)은 `SimPropManager.SpawnGhost` / `MakeGhostMaterial`에서.
- `DroneAvoidance.safeDistance / avoidStrength / upwardBias / frontBias` — 회피 민감도·세기. `avoid off`로 끄면 순수 조종만.
