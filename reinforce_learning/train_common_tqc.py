"""
train_common_tqc.py
───────────────────
train_common.py 의 TQC 버전.
drone_env_npy.py (demo_house 의존성 없음) + TQC 알고리즘 사용.
기존 train_common.py / drone_env.py 는 그대로 유지.
"""
import os
import shutil
import numpy as np

from drone_env_npy import DroneHouseEnv, load_obstacles_npy as load_obstacles

_BASE = os.path.dirname(os.path.abspath(__file__))


def run_training(env_kw, timesteps, out, init_from=None, seed=0,
                 label='', eval_episodes=20, eval_freq=20_000, lr=None):
    from sb3_contrib import TQC
    from stable_baselines3.common.monitor import Monitor

    if not os.path.isabs(out):
        out = os.path.join(_BASE, out)
    if init_from and not os.path.isabs(init_from):
        init_from = os.path.join(_BASE, init_from)

    print(f"\n{'='*55}\n[{label}] TQC 학습 시작\n{'='*55}")
    print(f"  난이도: {env_kw}")
    print(f"  timesteps={timesteps:,}  out={os.path.basename(out)}.zip")

    obs_pts = load_obstacles()
    env     = Monitor(DroneHouseEnv(obstacles=obs_pts, **env_kw))
    pure_kw = dict(env_kw, easy_mix=0.0)

    try:
        import tensorboard  # noqa: F401
        tb = os.path.join(_BASE, 'runs_tqc')
    except ImportError:
        tb = None

    if init_from and os.path.exists(init_from + '.zip'):
        print(f"  이전 모델 이어받기: {os.path.basename(init_from)}.zip")
        model = TQC.load(init_from, env=env, tensorboard_log=tb)
        if hasattr(model, 'learning_starts'):
            rb = init_from + '_replay.pkl'
            if os.path.exists(rb):
                model.load_replay_buffer(rb)
                model.learning_starts = 0
                print(f"  리플레이 버퍼 이어받기: {os.path.basename(rb)}")
            else:
                model.learning_starts = 10_000
                print("  리플레이 버퍼 없음 → 무작위 워밍업 1만 스텝 후 재개")
    else:
        if init_from:
            print(f"  [주의] init_from 모델({init_from}.zip) 없음 → 새 모델로 시작")
        print("  새 TQC 모델 생성")
        model = TQC("MlpPolicy", env, verbose=1, seed=seed,
                    learning_rate=3e-4, buffer_size=300_000, batch_size=256,
                    gamma=0.99, tau=0.005, learning_starts=10_000,
                    train_freq=1, gradient_steps=1, tensorboard_log=tb,
                    policy_kwargs=dict(n_quantiles=25, n_critics=2))

    if lr is not None:
        model.learning_rate = float(lr)
        model.lr_schedule   = lambda _: float(lr)
        print(f"  학습률 재설정: {lr:g}")

    try:
        import tqdm, rich  # noqa: F401
        pbar = True
    except ImportError:
        pbar = False

    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
    cbs = [CheckpointCallback(save_freq=100_000, save_path=os.path.join(_BASE, 'ckpt_tqc'),
                              name_prefix=os.path.basename(out))]
    best_dir = os.path.join(_BASE, 'best_tqc', os.path.basename(out))
    cbs.append(EvalCallback(Monitor(DroneHouseEnv(obstacles=obs_pts, **pure_kw)),
                            best_model_save_path=best_dir, eval_freq=eval_freq,
                            n_eval_episodes=30, deterministic=True, verbose=1))

    class StopOnDivergence(BaseCallback):
        def __init__(self):
            super().__init__()
            self.base = None

        def _on_step(self):
            if self.n_calls % 1000 != 0:
                return True
            m = self.model
            if getattr(m, 'log_ent_coef', None) is None:
                return True
            import torch as th
            v = float(th.exp(m.log_ent_coef.detach()).item())
            if self.base is None:
                self.base = max(v, 1e-3)
            elif v > 2.5 * self.base:
                print(f"\n[경고] ent_coef {self.base:.3f} → {v:.3f} 폭등 — 발산 감지, "
                      f"학습 자동 중단 (step {self.num_timesteps:,})")
                return False
            return True

    cbs.append(StopOnDivergence())

    print("  (Ctrl+C 로 중단해도 마지막에 저장)")
    try:
        model.learn(total_timesteps=timesteps, progress_bar=pbar, callback=cbs)
    except KeyboardInterrupt:
        print("\n[학습] 중단됨 — 현재까지 모델 저장")

    model.save(out + '_new')
    os.replace(out + '_new.zip', out + '.zip')
    print(f"[학습] 저장 완료: {out}.zip")
    if hasattr(model, 'save_replay_buffer'):
        model.save_replay_buffer(out + '_new_replay.pkl')
        os.replace(out + '_new_replay.pkl', out + '_replay.pkl')
        print(f"[학습] 리플레이 버퍼 저장: {os.path.basename(out)}_replay.pkl")
    best_zip = os.path.join(best_dir, 'best_model.zip')
    if os.path.exists(best_zip):
        shutil.copyfile(best_zip, out + '_best.zip')
        print(f"[학습] 학습 중 최고 모델: {os.path.basename(out)}_best.zip")

    def _quick_eval(m, tag):
        e = DroneHouseEnv(obstacles=obs_pts, **pure_kw)
        succ, lens = 0, []
        for _ in range(eval_episodes):
            obs, _ = e.reset()
            term = trunc = False
            while not (term or trunc):
                action, _ = m.predict(obs, deterministic=True)
                obs, _, term, trunc, info = e.step(action)
            if info["is_success"]:
                succ += 1
                lens.append(e.steps)
        msg = f"  [{tag}] 성공률 {succ}/{eval_episodes}"
        if lens:
            msg += f" | 성공 평균 {np.mean(lens):.0f}스텝"
        print(msg)

    print(f"\n[평가] 결정론 {eval_episodes} 에피소드:")
    _quick_eval(model, '최종')
    if os.path.exists(out + '_best.zip'):
        _quick_eval(TQC.load(out + '_best'), 'best')
        print("  → 성공률 높은 쪽을 다음 단계 --init-from 으로 쓸 것")
    return out
