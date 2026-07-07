"""
train_stage3.py — 커리큘럼 3단계: 전 층 (집 전체, 층 넘기 포함)
─────────────────────────────────────────────────────────────
2단계 최종(model_s2d)을 이어받아 집 전체 아무 두 점으로 비행 학습.

  python train_stage3.py
  python train_stage3.py --timesteps 800000

⚠️ 솔직한 경고: 층 넘기(계단)는 reactive 정책(주변 2m·14레이, 지도/기억 없음)으론
   거의 안 됨 — 좁은 계단 구멍을 못 뚫어 성공 신호가 안 들어옴(실제로 success 0 확인됨).
   현실적 대안 = 하이브리드:
     · 층 내부 이동 = 이 RL 모델
     · 층간(계단)  = 기존 RRT*/격자A* (../demo_house/hierarchical_plan.py 의 grid_astar)
   이 스크립트는 "그래도 끝까지 RL로" 도전해보는 용도. 안 되면 하이브리드로 전환.
"""
import os
import argparse
from train_common import run_training

_BASE = os.path.dirname(os.path.abspath(__file__))

# 3단계 난이도: 제한 없음(집 전체, 층 넘기 포함), 도착판정 0.4, 배회 페널티 ON
ENV = dict(goal_radius=0.4, sample_same_floor=False, max_goal_dist=None,
           min_separation=1.5, timeout_penalty=10.0)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='커리큘럼 3단계: 전 층')
    ap.add_argument('--timesteps', type=int, default=600_000)
    ap.add_argument('--out', default='model_s3')
    ap.add_argument('--init-from', default='model_s2d', help='이어받을 2단계(2d) 모델')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    print("⚠️ 층 넘기(계단)는 reactive 정책으론 매우 어려움. README/CLAUDE.md 의 하이브리드 참고.")
    run_training(ENV, a.timesteps, a.out, init_from=a.init_from, seed=a.seed,
                 label='3단계 · 전 층')
