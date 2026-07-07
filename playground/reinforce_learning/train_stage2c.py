"""
train_stage2c.py — 커리큘럼 2c단계: 같은 층 근거리 (문 1~2개 라우팅)
────────────────────────────────────────────────────────────────
2b(model_s2b)를 이어받아 '같은 층 아무 두 방, 8m 이내' 비행을 학습.
(구 train_stage2.py 에 해당 — 8m 직행은 0.25 정체였어서 2a/2b를 거쳐 온다)
에피소드의 30%는 2b(인접 방)를 섞어 망각을 방지.

  python train_stage2c.py                        # model_s2b 이어받아 model_s2c 저장
  python train_stage2c.py --timesteps 700000

진급 기준: 간이평가 성공률 ≥ 0.6 이면 → train_stage2d.py
"""
import os
import argparse
from train_common import run_training

_BASE = os.path.dirname(os.path.abspath(__file__))

# 2c 난이도: 같은 층(방 달라도 됨), 8m 이내, 도착판정 0.5 + 30% 인접방 혼합
ENV = dict(goal_radius=0.5, sample_same_floor=True, max_goal_dist=8.0,
           min_separation=1.5, timeout_penalty=10.0,
           easy_mix=0.3, easy_mix_mode='adjacent_rooms')

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='커리큘럼 2c단계: 같은 층 근거리')
    ap.add_argument('--timesteps', type=int, default=500_000)
    ap.add_argument('--out', default='model_s2c')
    ap.add_argument('--init-from', default='model_s2b', help='이어받을 2b 모델')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--lr', type=float, default=None,
                    help='학습률 재설정(발산 후 재시작 시 1e-4 권장). 미지정=기존 유지')
    a = ap.parse_args()
    run_training(ENV, a.timesteps, a.out, init_from=a.init_from, seed=a.seed,
                 label='2c단계 · 같은 층 근거리', lr=a.lr)
