"""
train_stage1.py — 커리큘럼 1단계: 같은 방, 가까운 목표 (첫 성공 만들기)
────────────────────────────────────────────────────────────────
난이도를 바꾸려면 아래 ENV 딕셔너리만 수정. (train.py 는 건드리지 않는다)

  python train_stage1.py                      # 기본 30만 스텝, model_s1.zip 저장
  python train_stage1.py --timesteps 500000

다음: train_stage2a.py  (이 결과 model_s1 을 이어받음)
"""
import os
import argparse
from train_common import run_training

_BASE = os.path.dirname(os.path.abspath(__file__))

# 1단계 난이도: 같은 방 안에서만, 4m 이내, 도착판정 후하게(0.6), 배회 페널티 ON
ENV = dict(goal_radius=0.6, sample_same_room=True, max_goal_dist=4.0,
           min_separation=0.8, timeout_penalty=10.0)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='커리큘럼 1단계: 같은 방')
    ap.add_argument('--timesteps', type=int, default=300_000)
    ap.add_argument('--out', default='model_s1')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    run_training(ENV, a.timesteps, a.out, init_from=None, seed=a.seed,
                 label='1단계 · 같은 방')
