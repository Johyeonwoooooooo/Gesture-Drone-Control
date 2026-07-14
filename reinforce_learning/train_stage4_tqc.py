"""
train_stage4_tqc.py — TQC 커리큘럼 4단계: 전체 집 (층간 포함)
────────────────────────────────────────────────────────────────
train_stage3_tqc.py(model_tqc_s3) 를 이어받아 집 전체 아무 두 점 비행 학습.
에피소드의 20% 는 3단계(same_floor) 를 섞어 망각 방지.

  /opt/conda/bin/python train_stage4_tqc.py
  /opt/conda/bin/python train_stage4_tqc.py --timesteps 1000000
"""
import os
import argparse
from train_common_tqc import run_training

_BASE = os.path.dirname(os.path.abspath(__file__))

ENV = dict(goal_radius=0.5, sample_same_floor=False, max_goal_dist=None,
           min_separation=1.5, timeout_penalty=10.0,
           easy_mix=0.2, easy_mix_mode='same_floor')

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='TQC 4단계: 전체 집')
    ap.add_argument('--timesteps', type=int, default=800_000)
    ap.add_argument('--out',       default='model_tqc_s4')
    ap.add_argument('--init-from', default='model_tqc_s3')
    ap.add_argument('--seed',      type=int, default=0)
    ap.add_argument('--lr',        type=float, default=None)
    a = ap.parse_args()
    run_training(ENV, a.timesteps, a.out, init_from=a.init_from, seed=a.seed,
                 label='TQC 4단계 · 전체 집', lr=a.lr)
