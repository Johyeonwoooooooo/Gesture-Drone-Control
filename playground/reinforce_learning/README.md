# 강화학습(RL) 자율비행 — demo_house

demo_house(Matterport 22방·3층) 점군 위에서 드론이 **시작점→도착점**으로 나는 정책을
SAC로 학습한다. 보상이 길을 알고(측지거리 shaping), 환경이 스스로 난이도를 올리는
(자동 커리큘럼) **v2 구조** — 수동 단계 없이 계단(층 넘기) 포함 집 전체를 한 번에 커버.

설계 배경·Before/After 는 `NOTION_SUMMARY.md` 참고.

## 핵심 구조

- **보상** = 측지거리(벽을 돌아가는 실제 최단거리, Dijkstra) 진척 × 5
  − 0.02/스텝 + 도착 +100 − 충돌 범프 −2 (potential-based shaping, Ng 1999)
- **자동 커리큘럼**: 시작점을 측지거리 밴드 [d_max−3, d_max]에서 샘플.
  최근 30ep 성공률 >60% → d_max +0.25m / <30% → −0.25m
- **서브골(carrot)**: 관측의 목표가 측지 내리막 2.5m 앞 지점을 가리킴(전역 안내는
  필드, 국소 조종은 RL 의 계층 구조)
- **Asymmetric actor-critic**: critic 만 특권 정보(측지거리·하강방향)를 봄
- **guide_clearance**: 안내 그래프는 통로 여유 ≥0.2m 만 경유(물리는 0.12m) —
  스캔 틈새(벽 속 공동)로 새는 유령 경로 차단

## 설치

```powershell
conda activate tello
pip install torch --index-url https://download.pytorch.org/whl/cu128   # py3.9 → cu128
pip install gymnasium stable-baselines3 tensorboard tqdm rich
```

## 사용법

현재 최종 모델 기준 학습 설정: `--ray-max 4 --rays horiz14 --subgoal 2.5 --bump 2
--clearance 0.12 --max-steps 700` (평가·시연도 같은 값이어야 함 — fly_pick 은 내장됨).

```powershell
# ── 시연 (모델 있으면 바로) ──────────────────────────────
python fly_pick.py --demo            # 1층 아무데나 → 3층 아무데나 자동 비행 + 3D
python fly_pick.py --demo --loop     # 반복 데모
python fly_pick.py                   # 3D에서 Shift+클릭으로 시작/도착 직접 찍기

# ── 평가 ────────────────────────────────────────────────
python _test_eval500.py model_geo_best 500       # 500ep 균일(4~32m) 확정 평가
python _test_anatomy.py 200                      # 200ep 해부(실패 모드/클러스터)
python _test_stair_eval.py model_geo_best 40     # 계단 과제 회귀 체크

# ── 학습 (fresh, 시간 예산) ──────────────────────────────
python train_geo.py --hours 12 --ray-max 4 --rays horiz14 --subgoal 2.5 `
    --bump 2 --clearance 0.12 --max-steps 700 --stair-mix 0.15

# 이어받기(같은 물리 체제만! 버퍼 자동 로드) / 학습 곡선
python train_geo.py --init-from model_geo --hours 8 --d-init <직전 d_max> ...
tensorboard --logdir runs            # rollout/d_max 우상향 + eval/success_rate 확인
```

## 현재 성능 (2026-07-17, model_geo_best)

- 균일·결정론·정책 단독 200ep: **0.980** (같은층 1.000 / 층넘기 0.973)
- 무작위 500ep (측지 4~32m): **0.968** (전 거리대 0.95~1.0, 성공 평균 49스텝)
- 계단 과제(양방향): **0.97**

## 파일

| 파일 | 역할 |
|---|---|
| `geo_env.py` | ★ v2 환경 (측지 보상·자동 커리큘럼·carrot·범퍼/슬라이드·guide) |
| `asym_policy.py` | Asymmetric actor-critic (특권 정보 경계 강제) |
| `train_geo.py` | ★ 학습 (시간 예산·병렬 env·발산 가드·best 저장) |
| `evaluate_geo.py` | 배치 평가(층넘기 분리)·경계 증명·점 찍기 시연 |
| `fly_pick.py` | ★ 3D 시연 (점 찍기 / --demo 무작위 층간 비행, 궤적 npy 저장) |
| `inject_stair_demos.py` | 계단 오라클 시범을 리플레이 버퍼에 주입 (재학습용) |
| `_test_eval500.py` `_test_anatomy.py` `_test_stair_eval.py` | 평가 도구 3종 |
| `drone_env.py` `evaluate.py` | (v1 잔존) v2 가 장애물 로딩·계단 체인·3D 픽킹을 재사용 |

생성물(gitignore): `model_geo.zip`(최종) / `model_geo_best.zip`(★시연용) /
`model_geo_replay.pkl`(이어받기) / `*_cache.npz`(격자·장애물 캐시) / `runs/`

## 주의 (사고 이력에서 나온 규칙)

- **물리/보상 체제가 바뀌면 이어받기 금지, fresh 학습** (섞인 버퍼 → SAC 발산 4회)
- 이어받기는 반드시 `_replay.pkl` 있는 모델로 (빈 버퍼 워밍업이 정책 파괴)
- env 를 만들 땐 항상 `clearance=0.12` 명시 (기본 0.235로 만들면 격자 캐시가 덮임)
- 곡선 읽기: success_rate 톱니는 정상(온도조절기). 진짜 지표는 d_max 우상향과
  eval/success_rate
