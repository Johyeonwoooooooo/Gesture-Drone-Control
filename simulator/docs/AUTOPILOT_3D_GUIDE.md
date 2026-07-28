# 3D 자동주행 실행 가이드

Unity Tello 시뮬레이터에서 3D 복셀맵 기반 자동비행을 실행·검증·시각화하는 명령 모음.

## 사전 준비

- 모든 명령은 아래 폴더에서 실행:
  ```
  cd C:\Prometheus\Gesture-Drone-Control\playground\path
  ```
- **실제 비행(`--execute`)이 필요한 명령만** Unity가 Play 모드여야 함. 나머지는 Unity 없이 동작.
- Unity Play 시 Console에 `[Tello] UDP server listening on 9000` 확인.

### Unity 씬 1회 셋업 (시각화용, 최초 한 번만)

1. Hierarchy 우클릭 → Create Empty → 이름 예: `Visualizers`
2. Inspector에서 컴포넌트 3개 추가:
   - `PlannedPathRenderer` — 계획 경로 (빨간 라인)
   - `FlightReportRenderer` — 실제 비행 궤적 (하늘색 라인) + 침범 지점 (빨간 구)
   - `VoxelMapRenderer` — 장애물 복셀 (와이어 큐브). `Voxel Map Path`에 입력:
     ```
     C:\Prometheus\Gesture-Drone-Control\playground\path\TEEsavR23oF_voxel_map_3d.json
     ```
3. 씬 저장 (Ctrl+S)

드론(TelloSimulator)에는 비행 트레일과 충돌 마커가 내장되어 있어 별도 셋업 불필요.

---

## 1. 실제 비행 (Unity Play 필요)

드론이 시작 좌표로 순간이동 → 이륙 → A* 경로 자동비행 → 착륙.

```bash
# 수평+수직 복합 (18m 상승)
python unity_autopilot_3d.py --map-json TEEsavR23oF_voxel_map_3d.json --start-x 53.6 --start-y 7.8 --start-z -8.9 --goal-x 17.6 --goal-y 25.8 --goal-z -28.9 --execute --trajectory-out traj.json

# 수직 위주 (1.8m -> 25.8m, 24m 상승 데모)
python unity_autopilot_3d.py --map-json TEEsavR23oF_voxel_map_3d.json --start-x -4.4 --start-y 1.8 --start-z -6.9 --goal-x 17.6 --goal-y 25.8 --goal-z -10.9 --execute --trajectory-out traj_climb.json

# 시작/도착 생략 가능: 시작=현재 드론 위치, 도착=자동 선정(위층 휴리스틱)
python unity_autopilot_3d.py --map-json TEEsavR23oF_voxel_map_3d.json --execute --trajectory-out traj.json
```

**산출물**
| 파일 | 의미 |
|---|---|
| 터미널 JSON / `unity_autopilot_3d_result.json` | 비행 지표 요약. `success`(목표 도달), `collision_count`(Unity 물리 충돌 횟수), `trajectory_intrusion_steps`(궤적이 장애물 복셀에 들어간 횟수 — 0이면 깨끗), `trajectory_min_clearance_m`(비행 중 장애물과의 최소 거리, **1.0m 미만이면 벽 접촉**), `mean/max_tracking_error`(경로 추종 오차) |
| `traj.json` (`--trajectory-out`) | 원시 데이터: 실제 궤적 좌표열 + 계획 경로 + 침범 지점 좌표. 아래 리포트 생성기의 입력 |
| `Assets/Resources/planned_path_3d.json` | 계획 경로 → Unity `PlannedPathRenderer`가 자동으로 빨간 라인 표시 |
| `Assets/Resources/flight_trajectory_3d.json` | 비행 궤적+침범 지점 → Unity `FlightReportRenderer`가 착륙 직후 자동 표시 |

**Unity 화면에서 보이는 것**: 비행 중 하늘색 트레일, 충돌 순간 빨간 구(TelloSimulator 내장), 착륙 후 궤적 라인+침범 지점(FlightReportRenderer).

---

## 2. 비행 리포트 생성 (Unity 불필요)

1번에서 만든 `traj.json`을 그림으로 변환.

```bash
python visualize_autopilot_3d.py --map-json TEEsavR23oF_voxel_map_3d.json --trajectory-json traj.json --output-prefix flight_report
```

**산출물** (`flight_report_*`)
| 파일 | 의미 |
|---|---|
| `_3d.png` | 3D 조감도: 장애물(회색 점) + 계획 경로(빨강) + 실제 궤적(파랑) + 침범 지점(X) |
| `_topdown.png` | 위에서 본 X-Z 평면도. 비행 고도 범위의 벽만 표시 |
| `_altitude.png` | 시간축 고도 그래프: 계획 고도 vs 실제 고도, 침범 스텝 표시 |
| `_clearance.png` | **충돌 여부 최종 판정 그래프.** 매 스텝 장애물까지 거리. 주황 점선(1.0m) 아래로 내려가면 벽 접촉, 위에 머물면 무접촉 |
| `_slices.png` | 층별 단면도: 드론이 지난 각 2m 고도층마다 그 층의 벽 배치 + 그 층에서의 궤적 |
| `_viewer.html` | **인터랙티브 3D 뷰어.** 브라우저에서 열기 → 드래그 회전, 휠 줌, 높이 슬라이더로 천장 잘라내기. 외부 라이브러리 없음(파일만 공유해도 열림) |

---

## 3. 성능 벤치마크 (Unity 불필요, 수초)

집 안 무작위 시작/도착 30쌍에 대한 플래닝 성능 측정.

```bash
python benchmark_autopilot_3d.py --map-json TEEsavR23oF_voxel_map_3d.json --cases 30
```

**산출물** (`benchmark_3d_results/`)
| 파일 | 의미 |
|---|---|
| `benchmark_planning.png` | 6패널 요약: ①플래닝 시간 vs 거리 ②경로 길이 vs 직선거리(점선=이상적) ③우회율 분포(1.0=직선) ④경로 곡률 ⑤계획 경로의 장애물 여유(전부 2m = 항상 안전거리 유지) ⑥웨이포인트 수. 제목에 **성공률** |
| `benchmark_planning.csv` | 케이스별 원시 지표 (엑셀에서 열어 추가 분석 가능) |

실제 비행까지 포함하려면 (Unity Play 필요, 케이스당 ~15초):
```bash
python benchmark_autopilot_3d.py --map-json TEEsavR23oF_voxel_map_3d.json --cases 5 --execute
```
→ `benchmark_execute.png/csv` 추가 생성: 추적 오차 분포, **케이스별 Unity 충돌 vs 복셀 침범 비교** 차트 포함.

---

## 4. 예외 케이스 검증 (Unity 불필요, 수초)

일부러 잘못된/어려운 입력 7종을 던져 견고성 증명.

```bash
python benchmark_autopilot_3d.py --map-json TEEsavR23oF_voxel_map_3d.json --suite edge --cases 10
```

**카테고리**: 벽 속 목표(자유 공간으로 스냅) / 맵 밖 목표(클램프+스냅) / 도달 불가 목표(크래시 없이 실패 보고) / 시작=도착 / 8m+ 수직 이동 / 최장거리 / 맵 경계 시작

**산출물** (`benchmark_3d_results/`)
| 파일 | 의미 |
|---|---|
| `benchmark_edge.png` | 카테고리별 통과율 막대그래프 (발표용 "예외 대응 증명" 자료) |
| `benchmark_edge.csv` | 케이스별 상세: `passed`(판정), `crashed`(예외 발생 여부), `path_found`(경로 발견 여부) |

---

## 5. 복셀맵 재생성 (집 배치를 바꿨을 때만)

Unity에서: 집 루트 선택 → **Tools → Add Mesh Colliders to Selected** → 같은 선택 상태로 **Tools → Export 3D Voxel Map (Selected Root)** → `playground\path\TEEsavR23oF_voxel_map_3d.json` 덮어쓰기.

---

## 문제 해결

| 증상 | 원인/해결 |
|---|---|
| `can't open file ...` | 실행 폴더가 다름 → `cd C:\Prometheus\Gesture-Drone-Control\playground\path` |
| `[WinError 10048] 포트 사용 중` | 이전 실행이 안 죽고 남음 → 그 터미널에서 Ctrl+C 또는 작업관리자에서 python 종료 |
| `No state after takeoff` / 드론이 안 움직임 | Unity가 Play 모드가 아니거나 Console에 컴파일 에러 → Play 확인, 빨간 에러 해결 |
| 드론이 추락 | Console에 `[Tello] Removed N leftover ArticulationBody` 로그 확인 — 없으면 스크립트 재컴파일 필요 (Unity 창 포커스) |
