"""
train_common.py
───────────────
단계별 학습 스크립트(train_stage1/2/3.py)가 공통으로 쓰는 학습 함수.
SAC 보일러플레이트(모델 생성/이어받기/학습/저장/간이평가)를 여기 한 곳에 둔다.
→ 난이도를 바꿀 땐 train.py 를 고치지 말고, 각 단계 스크립트의 ENV 딕셔너리만 본다.
"""
import os
import numpy as np

from drone_env import DroneHouseEnv

_BASE = os.path.dirname(os.path.abspath(__file__))


def run_training(env_kw, timesteps, out, init_from=None, algo='sac', seed=0,
                 label='', eval_episodes=20):
    """env_kw 난이도로 SAC/PPO 학습.
       init_from(.zip 확장자 제외 경로)가 있으면 그 모델을 이어받아 시작(커리큘럼)."""
    from stable_baselines3 import SAC, PPO
    from stable_baselines3.common.monitor import Monitor

    if not os.path.isabs(out):
        out = os.path.join(_BASE, out)
    if init_from and not os.path.isabs(init_from):
        init_from = os.path.join(_BASE, init_from)

    print(f"\n{'='*55}\n[{label}] 학습 시작\n{'='*55}")
    print(f"  난이도: {env_kw}")
    print(f"  timesteps={timesteps:,}  out={os.path.basename(out)}.zip")

    env = Monitor(DroneHouseEnv(**env_kw))
    Algo = SAC if algo == 'sac' else PPO

    try:
        import tensorboard  # noqa: F401
        tb = os.path.join(_BASE, 'runs')
    except ImportError:
        tb = None

    if init_from and os.path.exists(init_from + '.zip'):
        print(f"  이전 모델 이어받기: {os.path.basename(init_from)}.zip")
        model = Algo.load(init_from, env=env, tensorboard_log=tb)
    else:
        if init_from:
            print(f"  [주의] init_from 모델({init_from}.zip) 없음 → 새 모델로 시작")
        print(f"  새 {algo.upper()} 모델 생성")
        if algo == 'sac':
            model = SAC("MlpPolicy", env, verbose=1, seed=seed,
                        learning_rate=3e-4, buffer_size=300_000, batch_size=256,
                        gamma=0.99, tau=0.005, learning_starts=10_000,
                        train_freq=1, gradient_steps=1, tensorboard_log=tb)
        else:
            model = PPO("MlpPolicy", env, verbose=1, seed=seed,
                        learning_rate=3e-4, n_steps=2048, batch_size=64,
                        n_epochs=10, gamma=0.99, gae_lambda=0.95,
                        clip_range=0.2, tensorboard_log=tb)

    try:
        import tqdm, rich  # noqa: F401
        pbar = True
    except ImportError:
        pbar = False

    print("  (Ctrl+C 로 중단해도 마지막에 저장)")
    try:
        model.learn(total_timesteps=timesteps, progress_bar=pbar)
    except KeyboardInterrupt:
        print("\n[학습] 중단됨 — 현재까지 모델 저장")
    model.save(out)
    print(f"[학습] 저장 완료: {out}.zip")

    # 간이 평가 (deterministic, 시각화 없음)
    print(f"\n[평가] 학습된 정책으로 {eval_episodes} 에피소드 (랜덤 시작/도착):")
    eval_env = DroneHouseEnv(**env_kw)
    succ, lens = 0, []
    for _ in range(eval_episodes):
        obs, _ = eval_env.reset()
        term = trunc = False
        while not (term or trunc):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = eval_env.step(action)
        if info["is_success"]:
            succ += 1
            lens.append(eval_env.steps)
    msg = f"  성공률 {succ}/{eval_episodes}"
    if lens:
        msg += f" | 성공 평균 {np.mean(lens):.0f}스텝"
    print(msg)
    return out
