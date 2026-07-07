"""
train_stage2d.py — 커리큘럼 2d단계: 같은 층 무제한 (2단계 최종 목표)
────────────────────────────────────────────────────────────────
2c(model_s2c)를 이어받아 '같은 층 아무 두 점(거리 제한 없음)' 비행을 학습.
에피소드의 20%는 인접 방을 섞어 망각을 방지.

  python train_stage2d.py                        # model_s2c 이어받아 model_s2d 저장
  python train_stage2d.py --timesteps 800000

완료 후: evaluate.py 로 실비행 확인 → train_stage3.py (전 층, 하이브리드 권장)
  python evaluate.py --pick --model model_s2d --goal-radius 0.5
"""
import os
import argparse
from train_common import run_training

_BASE = os.path.dirname(os.path.abspath(__file__))

# 2d 난이도: 같은 층, 거리 제한 없음, 도착판정 0.5 + 20% 인접방 혼합
ENV = dict(goal_radius=0.5, sample_same_floor=True, max_goal_dist=None,
           min_separation=1.5, timeout_penalty=10.0,
           easy_mix=0.2, easy_mix_mode='adjacent_rooms')

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='커리큘럼 2d단계: 같은 층 무제한')
    ap.add_argument('--timesteps', type=int, default=500_000)
    ap.add_argument('--out', default='model_s2d')
    ap.add_argument('--init-from', default='model_s2c', help='이어받을 2c 모델')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    run_training(ENV, a.timesteps, a.out, init_from=a.init_from, seed=a.seed,
                 label='2d단계 · 같은 층 무제한')
