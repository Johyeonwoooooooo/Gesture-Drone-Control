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

## 사용 순서 (커리큘럼 ★권장)

전 집을 한 번에 학습하면 sparse reward 함정으로 성공률 0에 머문다.
**쉬움→어려움 단계별**로, 각 단계가 이전 모델을 `--init-from` 으로 이어받는다:

```powershell
python check_env.py          # 0. 환경 검증 (학습 전 필수)
python train_stage1.py       # 1.  같은 방            → model_s1
python train_stage2a.py      # 2a. 문턱 넘기(문 통과) → model_s2a
python train_stage2b.py      # 2b. 인접 방(문 1개)    → model_s2b
python train_stage2c.py      # 2c. 같은 층 ≤8m        → model_s2c
python train_stage2d.py      # 2d. 같은 층 무제한     → model_s2d
python evaluate.py --pick --model model_s2d --goal-radius 0.5   # 3D 비행 확인
```

| 단계 | 시작·도착 샘플링 | 새로 배우는 것 | 진급 기준(간이평가) |
|---|---|---|---|
| 1 | 같은 방, ≤4m | 기본 비행/장애물 회피 | ≥ 0.7 |
| 2a | 문 양쪽 1.5m 이내 | **문 통과** (2단계 병목 스킬) | ≥ 0.8 |
| 2b | 문 하나로 연결된 두 방, ≤5m | 방 비행 + 문 통과 결합 | ≥ 0.7 |
| 2c | 같은 층 아무 방, ≤8m | 문 1~2개 라우팅 | ≥ 0.6 |
| 2d | 같은 층, 거리 무제한 | 2단계 최종 목표 | — |

- 각 단계 도중 성과 확인: `python plot_progress.py` (success_rate 곡선)
- 2b부터는 이전 단계 에피소드를 20~30% 섞어(`easy_mix`) 앞 단계 실력 망각을 방지
- 문 경유 에피소드(2a/2b, 혼합 포함)는 **문 경유 진척 보상**: 문을 지나기 전엔
  (현위치→문)+(문→목표) 거리가 줄어야 +보상 — 직선거리 보상이 벽 쪽으로 유인하는
  문제를 고침. 문 통과(중심 근접 or 목표 직선 가시) 후엔 직선 진척으로 전환
- 수동 표시된 문 중심(door_center)이 문틀에 박힌 경우가 있어(5/20개 통과 불가였음)
  환경이 자동으로 개구부 중앙(주변 최대 여유 점)으로 스냅해서 사용
- 정체되면(성공률 안 오름) 그 단계 `--timesteps` 를 늘리거나 이전 단계로 돌아가 더 학습
- 단일 스크립트로 직접 난이도를 지정하고 싶으면 구버전 `train.py` (아래) 사용

### 난이도 (`--easy`, 구버전 train.py)

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
| `drone_env.py` | ★ 환경 (관측/행동/보상, KDTree 충돌검사, 난이도별 샘플링). RL의 핵심 |
| `check_env.py` | 학습 전 환경 검증 + 랜덤 롤아웃 |
| `train_stage1.py` `train_stage2a~2d.py` `train_stage3.py` | ★ 커리큘럼 단계별 학습. 난이도는 각 파일 맨 위 `ENV` 만 수정 |
| `train_common.py` | 단계 스크립트 공통 학습 함수 (SAC 보일러플레이트) |
| `train.py` | (구버전) 인자로 난이도를 직접 지정하는 단일 스크립트 |
| `evaluate.py` | 점 2개 → 학습된 정책 비행 → open3d 시각화 |
| `plot_progress.py` | 학습곡선(success_rate/보상) PNG 저장 |
| `obstacles_cache.npz` | 전역 장애물 점군 캐시 (자동 생성, gitignore 대상) |

## 한계 / 솔직한 메모

- 고정 맵 점A→B는 RRT*가 더 빠르고 확실. RL은 **학습/비교 실험**으로 의미.
- 학습 병목은 신경망이 아니라 **매 스텝 KDTree 충돌검사**. 100만 스텝에 수십 분~수 시간.
- 관측이 "위치+목표+로컬센서"라 한 맵 안 일반화는 되지만, 아주 좁은 계단 통과는 어려울 수 있음.
- 실제 Tello 비행 연결(좌표 m↔cm, RC 변환)은 아직 미연결 — RRT* 쪽과 동일 과제.
