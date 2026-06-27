"""
check_env.py
────────────
★ 학습 '전에' 반드시 먼저 돌려보는 검증 스크립트.

강화학습에서 제일 흔한 실수: 환경이 잘못됐는데 모르고 몇 시간 학습 → 안 됨.
그래서 학습 전에 (1) 관측/행동 규격이 맞는지, (2) 랜덤 행동으로 에피소드가
정상적으로 굴러가는지(충돌/도착/시간초과가 일어나는지)를 먼저 확인한다.

사용:
  conda activate tello
  python check_env.py
"""
import numpy as np
from drone_env import DroneHouseEnv

env = DroneHouseEnv()

# 1) Gymnasium 규격 검사 (SB3 제공). 문제가 있으면 여기서 에러로 알려줌.
try:
    from stable_baselines3.common.env_checker import check_env
    check_env(env, warn=True)
    print("[OK] Gymnasium 환경 규격 통과")
except ImportError:
    print("[건너뜀] stable-baselines3 미설치 → 규격검사 생략 (pip install stable-baselines3)")

print(f"\n관측 차원: {env.observation_space.shape}  (목표단위3 + 거리1 + 레이{len(env.ray_dirs)})")
print(f"행동 차원: {env.action_space.shape}  (vx, vy, vz)")

# 2) 랜덤 행동으로 5 에피소드 굴려보기 — 환경이 '말이 되는지' 눈으로 확인
print("\n랜덤 정책 5 에피소드 (아직 학습 안 한 상태):")
for ep in range(5):
    obs, _ = env.reset()
    total_r, term, trunc = 0.0, False, False
    d0 = np.linalg.norm(env.goal - env.pos)
    while not (term or trunc):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        total_r += r
    end = "도착" if info["is_success"] else ("충돌/이탈" if term else "시간초과")
    print(f"  ep{ep}: 시작거리 {d0:5.2f}m → 끝거리 {info['dist']:5.2f}m | "
          f"{env.steps:3d}스텝 | 보상합 {total_r:7.2f} | {end}")

print("\n랜덤 정책이라 대부분 충돌/시간초과가 정상이야. "
      "에러 없이 여기까지 나오면 환경 OK → train.py 로.")
