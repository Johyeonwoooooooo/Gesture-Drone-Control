"""
train_stage1_tqc.py — TQC 커리큘럼 1단계: 같은 방, 가까운 목표 (첫 성공 만들기)
────────────────────────────────────────────────────────────────
train_stage1.py 의 TQC 버전. drone_env_npy.py 사용 (demo_house 의존성 없음).

  /opt/conda/bin/python train_stage1_tqc.py
  /opt/conda/bin/python train_stage1_tqc.py --timesteps 500000

다음: train_stage2a_tqc.py  (이 결과 model_tqc_s1 을 이어받음)
"""
import os
import argparse
from train_common_tqc import run_training

_BASE = os.path.dirname(os.path.abspath(__file__))

ENV = dict(goal_radius=0.6, sample_same_room=True, max_goal_dist=4.0,
           min_separation=0.8, timeout_penalty=10.0)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='TQC 커리큘럼 1단계: 같은 방')
    ap.add_argument('--timesteps', type=int, default=300_000)
    ap.add_argument('--out',       default='model_tqc_s1')
    ap.add_argument('--seed',      type=int, default=0)
    a = ap.parse_args()
    run_training(ENV, a.timesteps, a.out, init_from=None, seed=a.seed,
                 label='TQC 1단계 · 같은 방')
