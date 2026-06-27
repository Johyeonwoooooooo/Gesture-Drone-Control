"""
train_stage2.py — 커리큘럼 2단계: 같은 층, 방 넘나들기
────────────────────────────────────────────────────
1단계(model_s1)를 이어받아 '같은 층 안에서 방을 넘나드는' 비행을 학습.
난이도를 바꾸려면 아래 ENV 만 수정.

  python train_stage2.py                         # model_s1 이어받아 model_s2 저장
  python train_stage2.py --init-from model       # 다른 1단계 모델로 시작(예: 기존 model.zip)
  python train_stage2.py --timesteps 700000

※ 8m로 했더니 success 0.25에서 정체(문 라우팅이 reactive 정책엔 어려움) →
  점프를 줄여 max_goal_dist=6 으로 시작. 잘 오르면 8로 키운 2차 학습을 이어붙여도 됨.

다음: train_stage3.py
"""
import os
import argparse
from train_common import run_training

_BASE = os.path.dirname(os.path.abspath(__file__))

# 2단계 난이도: 같은 층(방 달라도 됨), 6m 이내, 도착판정 0.5, 배회 페널티 ON
ENV = dict(goal_radius=0.5, sample_same_floor=True, max_goal_dist=6.0,
           min_separation=1.5, timeout_penalty=10.0)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='커리큘럼 2단계: 같은 층')
    ap.add_argument('--timesteps', type=int, default=500_000)
    ap.add_argument('--out', default='model_s2')
    ap.add_argument('--init-from', default='model_s1', help='이어받을 1단계 모델')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    run_training(ENV, a.timesteps, a.out, init_from=a.init_from, seed=a.seed,
                 label='2단계 · 같은 층')
