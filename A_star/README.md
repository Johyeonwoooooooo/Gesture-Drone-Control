# pathway.py

3D 포인트클라우드(`.npy` / `.ply`) 위에서 **A\*** 알고리즘으로 시작점 → 목표점 최적 경로를 찾는 독립 실행 스크립트입니다.
다른 의존성 파일(`3D.py` 등) 없이 이 파일 하나로 동작합니다.

## 1. 설치

Python 3.9 이상 필요. 의존 라이브러리:
- `numpy` (필수)
- `trimesh` (`--plot`으로 GLB 시각화 만들 때만 필요)

```bash
pip install numpy trimesh
```

## 2. 빠른 시작

```bash
python pathway.py --input scene.npy --start -3.0 -0.5 1.5 --goal 3.5 0.5 2.0
```

장애물이 표시된 포인트클라우드와 시작점/목표점 좌표(x y z)만 있으면 바로 실행됩니다.

`--input`은 세 가지 형태를 모두 지원합니다:

| 입력 형태 | 예시 | 동작 |
|---|---|---|
| 단일 파일 | `scene.npy`, `scene.ply` | 그 파일을 그대로 사용 |
| 방 폴더 (coord.npy 직접 포함) | `npy/room3/` | `room3/coord.npy`를 로드 (color.npy, normal.npy는 무시) |
| 여러 방을 담은 상위 폴더 | `npy/` (안에 `room1/coord.npy`, `room2/coord.npy`, ...) | 모든 방의 `coord.npy`를 하나로 합쳐서 전체 씬으로 사용 |

```bash
# 방 한 칸만 (npy/room3/coord.npy 사용)
python pathway.py --input npy/room3 --start -2 0 0.5 --goal 1 0 0.5

# 여러 방을 합친 전체 씬에서 경로 찾기 (npy/ 하위 모든 coord.npy 병합)
python pathway.py --input npy --start -2 0 0.5 --goal 8 3 1.5
```

> `color.npy`, `normal.npy`는 좌표 정보가 아니므로 읽지 않습니다. 경로 탐색에는 `coord.npy`(xyz 좌표)만 사용됩니다.

## 3. 동작 원리 (간단히)

이 스크립트는 두 가지 모드로 동작합니다.

### 단순 모드 (기본)
1. **포인트클라우드 로드** — `.npy` / `.ply` / 폴더 통합 입력에서 장애물 좌표를 읽습니다.
2. **Voxelize** — `--resolution` 크기의 격자로 변환. 포인트 있는 칸=장애물(1), 없는 칸=자유(0).
3. **A\* 탐색** — 26방향 이웃을 허용하는 A*로 시작 → 목표 최단 경로 탐색.
4. **경로 단순화** — `--no-smooth`로 끌 수 있음.

### 계층적 모드 (`--rooms-json` 사용 시) — *권장*
하나의 거대한 격자가 아니라, **방 단위로 쪼개서** 따로 A*를 돌리는 방식.

1. `rooms_graph.json` 로드 (방별 bbox, passages, edges).
2. `find_room_by_point()`로 **시작점/목표점이 속한 방** 찾기.
3. `edges`로 방 그래프를 만들고 **BFS로 방 시퀀스** 탐색 (예: `002 → 007 → 008`).
4. 각 방 안에서만 `coord.npy`를 로드해 A* 실행:
   - 첫 방: 글로벌 시작점 → 다음 방으로 가는 `door_center`
   - 중간 방: 이전 `door_center` → 다음 `door_center`
   - 마지막 방: 마지막 `door_center` → 글로벌 목표점
5. 모든 구간 경로를 이어붙여 글로벌 경로 완성.

**장점**: 방마다 작은 격자만 다루니까 메모리 절약 + 빠르고, 좁은 문도 `door_center`로 강제 통과해서 "도달 불가" 실패가 거의 없음.

## 4. 옵션 설명

| 옵션 | 필수 | 기본값 | 설명 |
|---|---|---|---|
| `--input` | ✅ | - | 포인트클라우드 입력. `.npy`/`.ply` 파일, `coord.npy`가 있는 방 폴더, 또는 여러 방 폴더의 상위 폴더 모두 가능 |
| `--rooms-json` | | - | `rooms_graph.json` 경로. **지정 시 계층적 모드**로 동작 (방 그래프 BFS + 방별 A*) |
| `--start` | ✅ | - | 시작점 좌표 `x y z` (예: `--start -2 0 0.5`) |
| `--goal` | ✅ | - | 목표점 좌표 `x y z` |
| `--resolution` | | `0.15` | voxel 한 칸의 크기(m). 작을수록 정밀하지만 느려짐 |
| `--margin` | | `0` | 장애물 팽창 마진(voxel 단위). 로봇/드론 크기만큼 여유를 두고 싶을 때 사용 |
| `--sample` | | `10` | 포인트클라우드 다운샘플링 간격. `1`이면 전체 포인트 사용(느림) |
| `--no-smooth` | | off | 경로 단순화(string-pulling)를 끄고 격자 단위 경로 그대로 반환 |
| `--save` | | - | 경로 저장 파일. 확장자로 형식 결정: `.npy` / `.csv` / `.json` |
| `--plot` | | off | 3D 경로를 **GLB 파일**(.glb)로 저장. 회전/줌 가능한 인터랙티브 3D 모델 |
| `--plot-output` | | `astar_path.glb` | GLB 저장 경로 |

## 5. 사용 예시

```bash
# ★ 권장: rooms_graph.json을 활용한 계층적 A* (방마다 따로 A* + 결과 이어붙임)
python pathway.py --input npy --rooms-json rooms_graph.json \
                  --start -1.5 -4.0 0.5 --goal 14.0 6.5 4.0 \
                  --plot --plot-output result.glb

# 단순 모드: 모든 방의 coord.npy를 하나로 합쳐서 단일 A*
python pathway.py --input npy --start -1.5 -4.0 0.5 --goal 14.0 6.5 4.0

# 경로를 json으로 저장
python pathway.py --input npy --rooms-json rooms_graph.json \
                  --start -1.5 -4.0 0.5 --goal 14.0 6.5 4.0 --save path.json

# 3D 시각화(GLB) 저장 — 웹 뷰어/Preview에서 회전·줌 가능
python pathway.py --input npy --start -1.5 -4.0 0.5 --goal 14.0 6.5 4.0 --plot --plot-output result.glb

# 로봇 크기를 고려해 장애물을 2칸(voxel)만큼 부풀리기
python pathway.py --input npy --start -1.5 -4.0 0.5 --goal 14.0 6.5 4.0 --margin 2

# 전체 포인트 사용 (다운샘플링 없이, 더 정밀하지만 느림)
python pathway.py --input npy --start -1.5 -4.0 0.5 --goal 14.0 6.5 4.0 --sample 1
```

## 6. 콘솔 출력 예시

```
입력 로드 중: scene.npy
  포인트 수: 12,400
격자 크기: (40, 28, 15), 자유 공간 비율: 96.3%
시작점: (-2.0, 0.0, 0.5)  목표점: (4.0, 1.0, 1.5)

[성공] 경로 탐색 완료
  소요 시간:    142.3 ms
  탐색 노드 수: 8,213
  경로 길이:    7.42 m
  Waypoint 수:  5
  저장됨:       path.json
  시각화 저장:  astar_path.png
```

경로를 찾지 못하면 `[실패]`와 함께 이유(도달 불가 / max_iters 초과 / 시작·목표 근처에 자유 공간 없음)를 출력하고 종료 코드 1을 반환합니다.

## 7. 저장 파일 형식

**경로 데이터 (`--save`)**
- `.json`: `{"waypoints": [[x,y,z], [x,y,z], ...]}`
- `.csv`: 헤더 `x,y,z` + 좌표 행
- `.npy`: `(N, 3)` numpy 배열

**시각화 (`--plot`)**
- `.glb` (glTF binary) — 인터랙티브 3D 모델. 회전/줌이 자유로워서 matplotlib PNG보다 훨씬 보기 좋음.

GLB 파일을 보는 방법:
- **웹 뷰어** (가장 간편): https://gltf-viewer.donmccurdy.com/ 에 파일 드래그
- **macOS**: Finder에서 더블클릭 → Preview/Quick Look으로 자동 열림
- **Blender**: File → Import → glTF 2.0
- **Three.js**: `GLTFLoader`로 바로 로드 가능

GLB 안 구성:
- 회색 점들 = 장애물 (포인트클라우드, 너무 많으면 80,000개로 다운샘플링)
- 파란색 튜브 = A* 경로
- 파란색 작은 구 = 중간 waypoint
- 녹색 큰 구 = 시작점
- 빨간색 큰 구 = 목표점

## 8. 다른 스크립트에서 함수로 사용하기

CLI 대신 코드에서 직접 호출하고 싶다면 `find_path()`를 import해서 쓸 수 있습니다.

```python
from pathway import find_path

path, info, grid_meta = find_path(
    input_path="scene.npy",
    start=(-2.0, 0.0, 0.5),
    goal=(4.0, 1.0, 1.5),
    resolution=0.15,
    margin=0,
    sample=10,
)

if path is None:
    print("실패:", info["reason"])
else:
    print("경로 길이:", info["distance_m"], "m")
    print("waypoints:", path)
```

## 9. 주의사항 / 팁

- `--resolution`이 너무 작으면 격자가 커져서 메모리/속도 부담이 커집니다. 처음엔 `0.15~0.3` 정도로 시작해보는 걸 추천합니다.
- 시작점/목표점이 장애물 내부거나 격자 밖이어도, 가장 가까운 자유 공간 칸을 자동으로 찾아 보정합니다(`find_nearest_free`, 최대 반경 20칸).
- `.ply`는 ascii / binary_little_endian 포맷만 지원합니다(binary_big_endian 불가).
- 큰 씬에서 A*가 느리면 `--sample`을 늘리거나 `--resolution`을 키워서 격자 크기를 줄이세요.
