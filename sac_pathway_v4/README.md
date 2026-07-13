# sac_pathway_v4.py — 관측·환경 layer 개선 SAC

`sac_pathway_v3.py`의 진단 결과(α 고착 / rollback 반복 / -1000 return 폭주)를 바탕으로,
문제가 SAC 알고리즘이 아닌 **환경·관측 설계**에 있다는 가정 하에 세 가지 layer를 개선.

## v3 → v4 3가지 개선

| 개선 | 목적 |
|---|---|
| **환경 안정화** (연속 충돌 조기 종료 + 페널티 상한) | -1000 return 폭주 제거 |
| **관측 강화** (raycast 3m→5m, frame stack 3, 이전 행동 스택) | 부분 관측 완화, 코너 회피 능력 |
| **Target entropy annealing** (-3 → -6) | α 고착 문제 해결, 후반 결정적 policy |

## 관측/행동 (v3와 다름)

- **관측 (88차원)**: 목표 상대벡터(3) + 목표 거리(1) + 26방향 raycast × **3프레임**(78) + 이전 행동 × **2프레임**(6)
- **행동 (3차원)**: v3와 동일 (tanh 스쿼시 x,y,z 이동, max_step = 0.3m)
- **보상**: v3 + 충돌 페널티 상한(30) + 5연속 충돌 시 조기 종료

## 사용법

```bash
# 학습 (모든 v4 개선 기본 켜짐)
python3 sac_pathway_v4.py train --input npy/ --total-steps 300000

# 관측 확장 (26방향 → ~98방향, 계산량 증가)
python3 sac_pathway_v4.py train --input npy/ --ray-density 2

# 계층적 planning + Open3D
 python3 ~/Desktop/astar_rrt_project/SAC/sac_pathway_v4.py plan \
    --input ~/Desktop/astar_rrt_project/npy/ \
    --rooms-json ~/Desktop/astar_rrt_project/A_star/rooms_graph.json \
    --model ~/Desktop/sac_pathway_v4/sac_model_best.pt \
    --start 4.4 0.1 -1.0 \
    --goal 11.2 5.0 4.3 \
    --show

# 학습 곡선
python sac_pathway_v4.py curve
```

## HM3D 데이터셋 실험 결과 (300k steps, CPU)

| | v1 | v2 | v3 | **v4 (density 1)** | **v4 (density 2)** |
|---|---|---|---|---|---|
| **best 성공률** | 42% | 60% | 66.7% | **72%** | **76%** |
| 최종 성공률 | 40% | 36% | 48% | ~50% | ~35% |
| rollback | — | — | 9회 | **35회** | **48회** |
| 학습 시간 | ~50분 | ~90분 | ~90분 | ~100분 | **~163분** |
| best 도달 시점 | ep 700 | ep 720 | ep 350 | **ep 480** | **ep 310** |

## 관찰 (중요)

**긍정:**
- best 성공률 **72~76%** 도달 (v3 대비 +6~10%p)
- α가 0.025~0.045대까지 지속 하락 (v3의 0.10 고착 해결)
- Stuck 비율 60% → 40%로 감소 (환경 안정화 효과 확인)
- H_t anneal 정상 동작 (-3 → -6)

**부정 (예상 밖):**
- **rollback이 오히려 크게 증가** (v3 9회 → v4 35~48회) — 붕괴는 여전히 반복
- ray-density 2가 계산 시간만 60% 늘리고 안정성엔 도움 안 됨
- Curriculum이 6.6m에서 정체 → 큰 씬을 여전히 못 커버

## 병목 재진단 (v5 개선 방향)

v3에서 관측 layer 병목이라 진단했지만, 실제로는 **다른 곳**임이 드러남:

1. **HER 재라벨링의 부작용** — 프레임 스택 도입 후 HER이 만들어내는 "가짜 궤적"이 실제 시간 흐름과 안 맞을 가능성. 스택 정보(과거 rays/actions)와 재라벨된 goal이 논리적으로 불일치.

2. **Curriculum + rollback의 상호 파괴** — 확장 → 실패 → rollback → 확장 리셋 사이클이 v3보다 더 빨리 돌아감. 아마 target entropy anneal이 policy를 확정적으로 만들어서 확장 목표에 취약해진 것.

3. **CPU에서 100분+ 학습의 근본 한계** — 어느 알고리즘이든 이 예산에선 3층 22방 전체 커버가 어려움. GPU 300분(=CPU 100분) 또는 imitation pretraining이 실질 해결책.

## v1~v4 스토리

> **v1~v3까지** 알고리즘 층위(SAC 원본 → auto-α → LayerNorm/PER/n-step)에서 42% → 66.7% 개선.
> **v4는** 관측/환경 layer(frame stacking, ray range 5m, 충돌 안전장치, entropy anneal)로 **best 76%** 달성하지만,
> **rollback 폭증(9→48회)** 이라는 예상 밖 결과를 얻음.
>
> 이는 관측 layer의 문제라기보다 **HER + curriculum + entropy anneal의 상호작용 문제**로 해석되며,
> 다음 단계는 알고리즘·관측 튜닝이 아닌 **GPU 예산 확장** 또는 **imitation pretraining**임을 시사.

발표에선 이 negative result를 오히려 강조하는 게 좋음 — "실험 기반 진단은 예상을 벗어날 수 있고, 이것이 왜 diagnostic-driven approach가 중요한지의 실증".

## v4 신규 CLI 옵션

- `--ray-range 5.0` (v3은 3.0)
- `--ray-density 1` — 1=26방향, 2=~98방향 (권장: 1)
- `--frame-stack 3`
- `--max-consec-collisions 5`
- `--collision-cap 30.0`
- `--target-entropy-end -6.0`
- `--target-entropy-frac 0.7`

## Ablation 명령어 (v3 재현)

```bash
python sac_pathway_v4.py train --input npy/ --total-steps 300000 \
    --frame-stack 1 --ray-range 3.0 \
    --max-consec-collisions 999 --collision-cap 9999 \
    --target-entropy-end -3.0
```

## 산출물 (`~/Desktop/sac_pathway_v4/`)

- `sac_model_best.pt` — best 모델 (plan 시 이걸 쓰세요)
- `training_log.csv` — 기존 컬럼 + `target_entropy`, `stuck`
- `learning_curve.png` — 4패널 학습 곡선

