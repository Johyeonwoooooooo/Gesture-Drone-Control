"""
train.py
────────
DroneHouseEnv 에서 SAC 로 자율비행 정책을 학습한다.

너가 직접 만지는 건 사실상 두 개뿐:
  --timesteps   : 총 학습 스텝 (제일 중요). 처음엔 작게 → 잘 되면 크게.
  보상함수      : drone_env.py 의 progress_scale / collision_penalty / goal_bonus
나머지(learning_rate, gamma, batch_size 등)는 SB3 기본값이 이미 잘 튜닝돼 있어서 그대로 둔다.

권장 순서:
  1) python check_env.py                 # 환경 검증 (먼저!)
  2) python train.py --timesteps 100000  # 빠른 확인 (수 분~수십 분)
  3) python train.py --timesteps 1000000 # 본 학습 (수십 분~수 시간)
  4) python evaluate.py --pick           # 점 2개 찍어 결과 확인

학습 곡선 보기(선택):  tensorboard --logdir runs
"""
import os
import argparse
import numpy as np

from drone_env import DroneHouseEnv

_BASE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--timesteps', type=int, default=300_000, help='총 학습 스텝')
    ap.add_argument('--algo', choices=['sac', 'ppo'], default='sac')
    ap.add_argument('--out', default=os.path.join(_BASE, 'model'), help='저장 경로(확장자 제외)')
    ap.add_argument('--resume', action='store_true', help='--out 모델을 이어서 학습')
    ap.add_argument('--init-from', default=None,
                    help='이 모델(.zip 경로, 확장자 제외)로 시작해 --out 에 저장 (커리큘럼용)')
    ap.add_argument('--easy', action='store_true',
                    help='난이도 낮춤 프리셋(같은 방+≤4m+goal_radius0.6+배회페널티). 1단계용')
    # 난이도 세부 조절 (커리큘럼 단계마다 직접 지정) — --easy 안 줄 때 사용
    ap.add_argument('--same-room', action='store_true', help='시작·도착을 같은 방에서만 뽑기')
    ap.add_argument('--max-goal-dist', type=float, default=None,
                    help='시작-도착 최대 거리(m). 미지정=집 전체')
    ap.add_argument('--goal-radius', type=float, default=0.3, help='도착 판정 반경(m)')
    ap.add_argument('--timeout-penalty', type=float, default=0.0, help='시간초과 페널티(배회 방지)')
    ap.add_argument('--min-sep', type=float, default=1.5, help='시작-도착 최소 거리(m)')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    from stable_baselines3 import SAC, PPO
    from stable_baselines3.common.monitor import Monitor

    if args.easy:   # 1단계 프리셋
        env_kw = dict(goal_radius=0.6, sample_same_room=True, max_goal_dist=4.0,
                      min_separation=0.8, timeout_penalty=10.0)
        print("[난이도] --easy: 같은 방/≤4m, goal_radius=0.6, 배회 페널티 ON")
    else:           # 세부 인자로 직접 (커리큘럼 2단계 이후)
        env_kw = dict(goal_radius=args.goal_radius, sample_same_room=args.same_room,
                      max_goal_dist=args.max_goal_dist, min_separation=args.min_sep,
                      timeout_penalty=args.timeout_penalty)
        print(f"[난이도] same_room={args.same_room} max_goal_dist={args.max_goal_dist} "
              f"goal_radius={args.goal_radius} timeout_penalty={args.timeout_penalty}")

    env = Monitor(DroneHouseEnv(**env_kw))
    Algo = SAC if args.algo == 'sac' else PPO

    # tensorboard 가 깔려 있으면 학습곡선 기록(runs/), 없으면 그냥 건너뜀
    try:
        import tensorboard  # noqa: F401
        tb = os.path.join(_BASE, 'runs')
    except ImportError:
        tb = None
        print("[안내] tensorboard 미설치 → 학습곡선 로그 생략 "
              "(보고 싶으면: pip install tensorboard)")
    model_zip = args.out + '.zip'

    if args.init_from and os.path.exists(args.init_from + '.zip'):
        print(f"[학습] 이전 모델로 시작(커리큘럼): {args.init_from}.zip → {model_zip} 에 저장")
        model = Algo.load(args.init_from, env=env, tensorboard_log=tb)
    elif args.resume and os.path.exists(model_zip):
        print(f"[학습] 기존 모델 이어서: {model_zip}")
        model = Algo.load(args.out, env=env, tensorboard_log=tb)
    else:
        print(f"[학습] 새 {args.algo.upper()} 모델 생성")
        if args.algo == 'sac':
            # 입문 권장값. net_arch=[256,256] 가 기본이라 따로 안 줘도 됨.
            model = SAC("MlpPolicy", env, verbose=1, seed=args.seed,
                        learning_rate=3e-4, buffer_size=300_000, batch_size=256,
                        gamma=0.99, tau=0.005, learning_starts=10_000,
                        train_freq=1, gradient_steps=1, tensorboard_log=tb)
        else:
            model = PPO("MlpPolicy", env, verbose=1, seed=args.seed,
                        learning_rate=3e-4, n_steps=2048, batch_size=64,
                        n_epochs=10, gamma=0.99, gae_lambda=0.95,
                        clip_range=0.2, tensorboard_log=tb)

    # 진행바는 tqdm+rich 있을 때만 (없으면 일반 로그로)
    try:
        import tqdm, rich  # noqa: F401
        pbar = True
    except ImportError:
        pbar = False

    print(f"[학습] timesteps={args.timesteps:,}  (Ctrl+C 로 중단해도 마지막에 저장)")
    try:
        model.learn(total_timesteps=args.timesteps, progress_bar=pbar)
    except KeyboardInterrupt:
        print("\n[학습] 중단됨 — 현재까지 모델 저장")
    model.save(args.out)
    print(f"[학습] 저장 완료: {model_zip}")

    # 학습된 정책 빠른 평가 (시각화 없이 성공률만)
    print("\n[평가] 학습된 정책으로 20 에피소드 (랜덤 시작/도착):")
    eval_env = DroneHouseEnv(**env_kw)
    succ, lens = 0, []
    for _ in range(20):
        obs, _ = eval_env.reset()
        term = trunc = False
        while not (term or trunc):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = eval_env.step(action)
        if info["is_success"]:
            succ += 1
            lens.append(eval_env.steps)
    msg = f"  성공률 {succ}/20"
    if lens:
        msg += f" | 성공 평균 {np.mean(lens):.0f}스텝"
    print(msg)
    print("  (성공률이 낮으면 --timesteps 를 늘리거나 보상함수를 조정해봐)")


if __name__ == '__main__':
    main()
