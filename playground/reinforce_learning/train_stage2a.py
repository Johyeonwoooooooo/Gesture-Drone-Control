"""
train_stage2a.py — 커리큘럼 2a단계: 문턱 넘기 (문 통과 집중훈련)
────────────────────────────────────────────────────────────
1단계(model_s1)를 이어받아 '문 앞 → 문 뒤'만 집중 반복.
시작·도착을 같은 층 문(door_center) 양쪽 1.5m 안에서만 뽑아서,
2단계 병목 스킬인 '문 통과'에 dense한 성공 신호를 준다.

  python train_stage2a.py                        # model_s1 이어받아 model_s2a 저장
  python train_stage2a.py --init-from model      # 다른 1단계 모델로 시작
  python train_stage2a.py --timesteps 400000

진급 기준: 간이평가 성공률 ≥ 0.8 이면 → train_stage2b.py
"""
import os
import argparse
from train_common import run_training

_BASE = os.path.dirname(os.path.abspath(__file__))

# 2a 난이도: 문 양쪽 1.5m 안에서만, 도착판정 후하게(0.7), 배회 페널티 ON.
# max_steps 120: 3m 과제에 300스텝은 배회 낭비 — 빨리 끊어 에피소드 밀도를 높임.
# 진척 보상은 문 경유 거리 기준(drone_env 의 door_route_reward, 기본 ON) — 문 방향으로 유도.
ENV = dict(goal_radius=0.7, sample_door_cross=True, door_range=1.5,
           min_separation=1.0, timeout_penalty=10.0, max_steps=120)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='커리큘럼 2a단계: 문턱 넘기')
    ap.add_argument('--timesteps', type=int, default=300_000)
    ap.add_argument('--out', default='model_s2a')
    ap.add_argument('--init-from', default='model_s1', help='이어받을 1단계 모델')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    run_training(ENV, a.timesteps, a.out, init_from=a.init_from, seed=a.seed,
                 label='2a단계 · 문턱 넘기')
