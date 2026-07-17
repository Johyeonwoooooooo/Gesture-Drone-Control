"""
train_stage3.py — 3단계: 전체집(층 넘기 포함) 밤샘 학습
────────────────────────────────────────────────────────
집 전체 아무 두 방(층이 달라도 됨)으로 비행을 학습한다. 순수 reactive 정책은
계단을 못 뚫는 게 확인됐으므로, 이 버전은 구조를 바꿔서 도전한다:

  · waypoint_obs=True — 관측이 '최종 목표' 대신 '다음 웨이포인트'(문/계단점)를
    가리킴. 정책이 할 일이 "가까운 지점까지 비행"으로 단순해짐 (obs 18차원 불변).
  · sample_full_house=True — 방그래프 BFS 라우팅에 계단 엣지 포함. 계단 구간은
    grid A* 로 미리 깔아둔 웨이포인트 체인(최초 1회 계산, 캐시)을 경유하고,
    체인 근처에서만 여유반경을 0.12m 로 완화해 통과 가능하게 함.

  python train_stage3.py                          # model_s2c 이어받아 model_s3 저장
  python train_stage3.py --timesteps 3000000      # 더 길게 (약 60 fps → 1M ≈ 4.5h)

★ 이어받기는 반드시 `_replay.pkl` 이 있는 모델로 (버퍼 없는 이어받기 = 무작위
  워밍업 → 정책 파괴 사고 2회 이력). 기본값 model_s2c 는 2c 학습 종료 시 버퍼가
  같이 저장돼 있음. model_s2c_best 는 버퍼가 없으므로 --init-from 으로 주지 말 것.
"""
import argparse
from train_common import run_training

# 전체집 난이도: 층 넘기 포함, 서브골 관측, 도착판정 0.5 + 20% 인접방 혼합(망각 방지)
ENV = dict(goal_radius=0.5, sample_full_house=True, waypoint_obs=True,
           min_separation=1.5, timeout_penalty=10.0, max_steps=400,
           easy_mix=0.2, easy_mix_mode='adjacent_rooms')

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='3단계: 전체집(층 넘기) 밤샘 학습')
    ap.add_argument('--timesteps', type=int, default=2_000_000,
                    help='약 60fps 기준 2M ≈ 9시간 (밤샘용)')
    ap.add_argument('--out', default='model_s3')
    ap.add_argument('--init-from', default='model_s2c',
                    help='이어받을 모델 (반드시 _replay.pkl 있는 것)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--lr', type=float, default=1e-4,
                    help='관측 의미가 바뀌는 이어받기라 낮은 학습률 기본')
    a = ap.parse_args()
    run_training(ENV, a.timesteps, a.out, init_from=a.init_from, seed=a.seed,
                 label='3단계 · 전체집(층 넘기, 웨이포인트 관측)', lr=a.lr,
                 eval_freq=40_000)
