"""
train_stage3_tqc.py — TQC 커리큘럼 3단계: 같은 층 거리 무제한
────────────────────────────────────────────────────────────────
train_stage2_tqc.py(model_tqc_s2) 를 이어받아 '같은 층, 거리 제한 없음' 비행 학습.
에피소드의 20% 는 2단계(same_floor + 8m) 를 섞어 망각 방지.

  /opt/conda/bin/python train_stage3_tqc.py
  /opt/conda/bin/python train_stage3_tqc.py --timesteps 800000

완료 후: train_stage4_tqc.py (전체 집, 층간 포함)
"""
import os
import argparse
from train_common_tqc import run_training

_BASE = os.path.dirname(os.path.abspath(__file__))

ENV = dict(goal_radius=0.5, sample_same_floor=True, max_goal_dist=None,
           min_separation=1.5, timeout_penalty=10.0,
           easy_mix=0.2, easy_mix_mode='same_room')

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='TQC 3단계: 같은 층 무제한')
    ap.add_argument('--timesteps', type=int, default=500_000)
    ap.add_argument('--out',       default='model_tqc_s3')
    ap.add_argument('--init-from', default='model_tqc_s2')
    ap.add_argument('--seed',      type=int, default=0)
    ap.add_argument('--lr',        type=float, default=None)
    a = ap.parse_args()
    run_training(ENV, a.timesteps, a.out, init_from=a.init_from, seed=a.seed,
                 label='TQC 3단계 · 같은 층 무제한', lr=a.lr)
