# Git Change Report

작성 기준: 현재 워크트리의 `git status`, `git diff --stat`, `git diff --name-status` 기준

## 요약

- 추적 중인 변경 파일: 7개
- 미추적 파일: 466개
- 추적 파일 diff 통계: 3,054 insertions, 7,254 deletions
- 큰 변경 대부분은 Unity 씬 파일과 에셋 추가/교체에서 발생
- 집게발 관련 신규 구현은 `DroneGripper.cs`에 포함됨

## 추적 중인 변경

| 상태 | 파일 | 변경 요약 |
| --- | --- | --- |
| Modified | `simulator/tello_simulator/Assets/CameraFollow.cs` | 3인칭 카메라 오프셋을 더 멀고 낮은 위치로 변경 |
| Deleted | `simulator/tello_simulator/Assets/TEEsavR23oF.glb` | 기존 GLB 에셋 삭제 |
| Deleted | `simulator/tello_simulator/Assets/TEEsavR23oF.glb.meta` | 기존 GLB 메타 파일 삭제 |
| Modified | `simulator/tello_simulator/Assets/TelloSimulator.cs` | 홈 위치, 키보드 조종, 카메라 기준 이동, pickup/putdown, carrying 상태, 유령 충돌 등록, Pickable 충돌 제외 추가 |
| Modified | `simulator/tello_simulator/Assets/test.unity` | Unity 씬 대규모 변경 |
| Modified | `simulator/tello_simulator/Packages/packages-lock.json` | Git 기반 Unity 패키지 의존성 revision 고정 |
| Modified | `simulator/tello_simulator/ProjectSettings/TagManager.asset` | `Pickable` 태그와 레이어 추가 |

## 주요 코드 변경

### `CameraFollow.cs`

- 기존 3인칭 오프셋:

```csharp
new Vector3(0, 1.5f, -4.0f)
```

- 변경 후:

```csharp
new Vector3(0, 0.5f, -10f)
```

효과:

- 드론을 더 멀리서 보는 3인칭 시점으로 변경
- 시야 확보가 쉬워짐

### `TelloSimulator.cs`

추가된 기능:

- `homePosition` 추가
- `cargoOffset` 추가
- `T`, `L`, `H` 키 입력 처리
- `WASD`, `Q/E`, 방향키 기반 수동 조종 추가
- 카메라 기준 이동 방식으로 변경
- UDP `pickup <objectName>` 명령 추가
- UDP `putdown` 명령 추가
- 상태 JSON에 `carrying` 필드 추가
- 유령 충돌 등록용 `RegisterGhostCollision()` 추가
- `Pickable` 태그 오브젝트는 비행 충돌 기록에서 제외

의도:

- 시뮬레이터 수동 테스트 편의성 개선
- 드론이 오브젝트를 들고 내려놓는 상태를 UDP 상태 스트림에서 확인 가능하게 함
- Pickable 물체를 집거나 접촉할 때 불필요하게 장애물 충돌로 기록되는 문제 완화

### `DroneGripper.cs`

신규 파일이며 집게발 기능을 담당함.

주요 기능:

- 드론 하위에 4개 L자형 손가락 집게발 자동 생성
- 스페이스바로 집기/놓기 토글
- `Pickable` 태그 오브젝트 탐색
- 집을 때 오브젝트 콜라이더 비활성화 후 `HoldPoint`에 부착
- 놓을 때 오브젝트 콜라이더 즉시 복원
- 놓는 직후 드론/집게발과의 순간 충돌만 짧게 무시
- 자식 콜라이더가 잡힌 경우 `attachedRigidbody` 루트를 우선 대상으로 선택
- `UNITY_EDITOR` 가드로 에디터 전용 `OnValidate()` 코드 보호

집게발 오브젝트 사라짐 문제 대응:

- 기존 문제 원인: 콜라이더가 꺼진 상태에서 중력을 켜면 바닥 충돌 없이 아래로 떨어질 수 있음
- 수정 방향: release 직후 콜라이더를 바로 켜고, 집게발 충돌만 임시 무시

## Unity 설정/패키지 변경

### `TagManager.asset`

추가됨:

- Tag: `Pickable`
- Layer: `Pickable`

사용처:

- 집게발이 집을 수 있는 오브젝트 필터링
- 시뮬레이터 충돌 기록에서 Pickable 오브젝트 제외

### `packages-lock.json`

Git 패키지 revision이 명시적으로 고정됨:

- `com.unity.robotics.urdf-importer`
- `org.khronos.unitygltf`

효과:

- Unity 패키지 재해석 시 같은 커밋을 사용하게 되어 의존성 재현성이 좋아짐

## 에셋/씬 변경

### 삭제된 기존 에셋

- `simulator/tello_simulator/Assets/TEEsavR23oF.glb`
- `simulator/tello_simulator/Assets/TEEsavR23oF.glb.meta`

### 추가된 신규 GLB

- `simulator/tello_simulator/Assets/Qpor2mEya8F.glb`
- `simulator/tello_simulator/Assets/Qpor2mEya8F.glb.meta`

### 대규모 미추적 에셋

현재 미추적 파일 466개가 존재함.

폴더별 주요 분포:

| 경로 | 파일 수 |
| --- | ---: |
| `simulator/tello_simulator/Assets/HorrorPuzzleItems` | 288 |
| `simulator/tello_simulator/Assets/Little_GhostLP(FREE)` | 86 |
| `simulator/tello_simulator/Assets/InstantZoomies` | 28 |
| `simulator/tello_simulator/Assets/Art` | 22 |
| `simulator/tello_simulator/Assets/DarkDropStudio` | 21 |
| `simulator/tello_simulator/.vscode` | 3 |

주요 추가 에셋 종류:

- 퍼즐 아이템 프리팹/모델/텍스처
- 유령 캐릭터 에셋
- 버킷 에셋
- 공포 소품 에셋
- UnityGLTF/URP 관련 생성 파일로 보이는 CommandBuffer 래퍼 코드

## 변경 규모

`git diff --numstat` 기준:

| 파일 | 추가 | 삭제 |
| --- | ---: | ---: |
| `CameraFollow.cs` | 1 | 1 |
| `TelloSimulator.cs` | 146 | 12 |
| `test.unity` | 2,903 | 6,969 |
| `packages-lock.json` | 2 | 2 |
| `TagManager.asset` | 2 | 1 |
| `TEEsavR23oF.glb.meta` | 0 | 269 |
| `TEEsavR23oF.glb` | binary | binary |

## 주의 사항

- `DroneGripper.cs`는 현재 미추적 파일이라 아직 git에 추가되지 않은 상태임.
- `test.unity` 변경량이 매우 커서 씬 내부 변경은 Unity Editor에서 직접 확인하는 것이 필요함.
- 미추적 에셋이 466개로 많아 커밋 전 포함할 에셋과 제외할 에셋을 분리하는 것이 좋음.
- `.vscode` 설정 파일은 팀 공유가 필요한 경우만 커밋하는 것이 좋음.
- Unity Play 모드 검증은 이 리포트 작성 시점에 수행하지 않음.

## 커밋 전 권장 정리

1. `DroneGripper.cs`, `DroneGripper.cs.meta`, `TagManager.asset`, 관련 씬 변경이 함께 들어가는지 확인
2. `TEEsavR23oF.glb` 삭제와 `Qpor2mEya8F.glb` 추가가 의도된 교체인지 확인
3. 대규모 에셋 폴더 중 실제 씬에서 참조하는 것만 커밋 대상에 포함
4. `test.unity`를 Unity Editor에서 열어 누락된 prefab/material reference가 없는지 확인
5. Play 모드에서 집기/놓기, Pickable 충돌 제외, carrying 상태 스트림을 확인
