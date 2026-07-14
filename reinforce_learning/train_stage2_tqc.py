"""
train_stage2_tqc.py — TQC 커리큘럼 2단계: 같은 층, 8m 이내
────────────────────────────────────────────────────────────────
train_stage1_tqc.py(model_tqc_s1) 을 이어받아 '같은 층 아무 두 방, 8m 이내' 비행 학습.
문 엣지 없이 동작하는 버전 (door_cross/adjacent_rooms 모드 미사용).
에피소드의 30% 는 1단계(same_room) 를 섞어 기본 비행 실력 유지.

  /opt/conda/bin/python train_stage2_tqc.py
  /opt/conda/bin/python train_stage2_tqc.py --timesteps 700000

진급 기준: 성공률 >= 0.6 → train_stage3_tqc.py
"""
import os
import argparse
from train_common_tqc import run_training

_BASE = os.path.dirname(os.path.abspath(__file__))

ENV = dict(goal_radius=0.5, sample_same_floor=True, max_goal_dist=8.0,
           min_separation=1.5, timeout_penalty=10.0,
           easy_mix=0.3, easy_mix_mode='same_room')

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='TQC 2단계: 같은 층 8m 이내')
    ap.add_argument('--timesteps', type=int, default=500_000)
    ap.add_argument('--out',       default='model_tqc_s2')
    ap.add_argument('--init-from', default='model_tqc_s1')
    ap.add_argument('--seed',      type=int, default=0)
    ap.add_argument('--lr',        type=float, default=None)
    a = ap.parse_args()
    run_training(ENV, a.timesteps, a.out, init_from=a.init_from, seed=a.seed,
                 label='TQC 2단계 · 같은 층 8m', lr=a.lr)
