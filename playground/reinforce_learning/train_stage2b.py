"""
train_stage2b.py — 커리큘럼 2b단계: 인접 방 (문 1개 통과)
──────────────────────────────────────────────────────
2a(model_s2a)를 이어받아 '문 하나로 연결된 두 방' 사이 비행을 학습.
방 안 비행(1단계) + 문 통과(2a)를 결합하는 단계.
에피소드의 30%는 2a(문턱 넘기)를 섞어 문 통과 실력을 까먹지 않게 한다.

  python train_stage2b.py                        # model_s2a 이어받아 model_s2b 저장
  python train_stage2b.py --timesteps 700000

진급 기준: 간이평가 성공률 ≥ 0.7 이면 → train_stage2c.py
"""
import os
import argparse
from train_common import run_training

_BASE = os.path.dirname(os.path.abspath(__file__))

# 2b 난이도: 인접한 두 방, 5m 이내, 도착판정 0.6 + 30% 문턱넘기 혼합
ENV = dict(goal_radius=0.6, sample_adjacent_rooms=True, max_goal_dist=5.0,
           min_separation=1.5, timeout_penalty=10.0,
           easy_mix=0.3, easy_mix_mode='door_cross')

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='커리큘럼 2b단계: 인접 방')
    ap.add_argument('--timesteps', type=int, default=500_000)
    ap.add_argument('--out', default='model_s2b')
    ap.add_argument('--init-from', default='model_s2a', help='이어받을 2a 모델')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    run_training(ENV, a.timesteps, a.out, init_from=a.init_from, seed=a.seed,
                 label='2b단계 · 인접 방')
