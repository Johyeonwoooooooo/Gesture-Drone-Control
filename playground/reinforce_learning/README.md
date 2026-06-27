# 강화학습(RL) 자율비행 — demo_house

demo_house 점군 위에서 드론이 **시작점→도착점**으로 날아가는 정책을 강화학습으로 배운다.
RRT*/A*(`../demo_house/hierarchical_plan.py`)의 대안/비교 실험용.

## RL 한 줄 요약 (입문)

- 강화학습 = **시행착오로 배우기**. "데이터셋"을 미리 안 만들고, **환경**을 만들면
  에이전트(드론)가 그 안에서 수백만 번 직접 날아보며 **경험(데이터)을 스스로 생성**해 학습한다.
- 우리가 정의하는 건 환경의 3요소뿐:
  - **관측(state)** = 목표 방향(3) + 목표 거리(1) + 14방향 거리센서(레이캐스트)
  - **행동(action)** = 속도벡터 (vx, vy, vz), 각 -1~1
  - **보상(reward)** = 가까워지면 +, 매 스텝 작은 -, 벽 박으면 -10, 도착하면 +100
- 충돌검사·전역 점군은 기존 코드(`rrt_star_drone.py`, `hierarchical_plan.py`)를 그대로 재활용.

## 설치

```powershell
conda activate tello
# GPU torch (tello=py3.9 → cu128).  py3.10+ 면 cu130 가능
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install gymnasium stable-baselines3 tensorboard tqdm rich
```
- GPU 확인: `python -c "import torch; print(torch.cuda.is_available())"` → True
- `tensorboard`(학습곡선), `tqdm`+`rich`(진행바)는 없으면 자동 생략되지만 있으면 편함
- 점군/충돌검사용 numpy<2, scipy, open3d 는 tello 환경에 이미 있음

## 사용 순서

```powershell
python check_env.py                       # 1. 환경 검증 (학습 전 필수)
python train.py --easy --timesteps 300000 # 2. 쉬운 버전부터 (첫 성공 빨리 보기) ★권장 시작
python train.py --timesteps 1000000       # 3. 본 학습(전체 난이도), model.zip 저장
python evaluate.py --pick                 # 4. 점 2개 찍어 비행 결과 3D 확인
```

### 난이도 (`--easy`)

전체 집(3층 22방) 아무 두 점이나는 단순 정책엔 너무 어려워 `success_rate` 가 0 근처에
머물기 쉽다. RL 은 **한 번도 성공 못 한 행동은 못 배우므로**, 먼저 쉬운 문제로 첫 성공을
만들어 신호를 키운 뒤 어렵게 가는 게 정석(커리큘럼). `--easy` 가 그 쉬운 설정:

| 항목 | 기본 | --easy | 효과 |
|---|---|---|---|
| 시작·도착 위치 | 집 전체 랜덤 | **같은 방 안** | 문/계단 횡단 제거 |
| 목표 거리 상한 | 없음 | **4 m** | 가까운 목표만 |
| 도착 판정 `goal_radius` | 0.30 m | **0.60 m** | 근처까지 가면 성공 |
| 시간초과 페널티 | 0 | **10** | "안전하게 배회" 국소최적 방지 |

`--easy` 로 success_rate 가 충분히 오르면(>0.5), `--easy` 빼고 본 학습으로 넘어간다.
(`drone_env.py` 의 `sample_same_room`/`max_goal_dist`/`goal_radius`/`timeout_penalty`
인자를 직접 조절하면 그 중간 난이도도 가능)

평가 때 좌표 직접 지정도 가능:
```powershell
python evaluate.py --start 6 6 4.2 --goal 2 0 1.2
```

## 네가 실제로 만지는 것

대부분 SB3 기본값이라 건드릴 게 별로 없다. 실질적으로 둘:

1. **`--timesteps`** (train.py): 학습량. 성공률 낮으면 늘린다.
2. **보상함수** (drone_env.py 상단 인자): `progress_scale`, `collision_penalty`,
   `goal_bonus`, `step_penalty`. 행동이 이상하면 여기를 조정.

신경망 크기(`net_arch`)나 `learning_rate` 같은 건 보통 그대로 둔다.

## 파일

| 파일 | 역할 |
|---|---|
| `drone_env.py` | ★ 환경 (관측/행동/보상, KDTree 충돌검사). RL의 핵심 |
| `check_env.py` | 학습 전 환경 검증 + 랜덤 롤아웃 |
| `train.py` | SAC/PPO 학습 → `model.zip` |
| `evaluate.py` | 점 2개 → 학습된 정책 비행 → open3d 시각화 |
| `obstacles_cache.npz` | 전역 장애물 점군 캐시 (자동 생성, gitignore 대상) |

## 한계 / 솔직한 메모

- 고정 맵 점A→B는 RRT*가 더 빠르고 확실. RL은 **학습/비교 실험**으로 의미.
- 학습 병목은 신경망이 아니라 **매 스텝 KDTree 충돌검사**. 100만 스텝에 수십 분~수 시간.
- 관측이 "위치+목표+로컬센서"라 한 맵 안 일반화는 되지만, 아주 좁은 계단 통과는 어려울 수 있음.
- 실제 Tello 비행 연결(좌표 m↔cm, RC 변환)은 아직 미연결 — RRT* 쪽과 동일 과제.
