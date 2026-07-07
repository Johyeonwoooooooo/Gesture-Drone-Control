"""
train_common.py
───────────────
단계별 학습 스크립트(train_stage1/2/3.py)가 공통으로 쓰는 학습 함수.
SAC 보일러플레이트(모델 생성/이어받기/학습/저장/간이평가)를 여기 한 곳에 둔다.
→ 난이도를 바꿀 땐 train.py 를 고치지 말고, 각 단계 스크립트의 ENV 딕셔너리만 본다.
"""
import os
import shutil
import numpy as np

from drone_env import DroneHouseEnv, load_obstacles

_BASE = os.path.dirname(os.path.abspath(__file__))


def run_training(env_kw, timesteps, out, init_from=None, algo='sac', seed=0,
                 label='', eval_episodes=20, eval_freq=20_000, lr=None):
    """env_kw 난이도로 SAC/PPO 학습.
       init_from(.zip 확장자 제외 경로)가 있으면 그 모델을 이어받아 시작(커리큘럼).
       학습 중 eval_freq 스텝마다 결정론 평가로 역대 최고 모델을 <out>_best.zip 에 저장."""
    from stable_baselines3 import SAC, PPO
    from stable_baselines3.common.monitor import Monitor

    if not os.path.isabs(out):
        out = os.path.join(_BASE, out)
    if init_from and not os.path.isabs(init_from):
        init_from = os.path.join(_BASE, init_from)

    print(f"\n{'='*55}\n[{label}] 학습 시작\n{'='*55}")
    print(f"  난이도: {env_kw}")
    print(f"  timesteps={timesteps:,}  out={os.path.basename(out)}.zip")

    obs_pts = load_obstacles()                     # 점군은 한 번만 만들어 모든 env가 공유
    env = Monitor(DroneHouseEnv(obstacles=obs_pts, **env_kw))
    # 평가는 '순수 과제' 기준: easy_mix(쉬운 에피소드 혼합)를 빼고 재서 과대평가 방지
    pure_kw = dict(env_kw, easy_mix=0.0)
    Algo = SAC if algo == 'sac' else PPO

    try:
        import tensorboard  # noqa: F401
        tb = os.path.join(_BASE, 'runs')
    except ImportError:
        tb = None

    if init_from and os.path.exists(init_from + '.zip'):
        print(f"  이전 모델 이어받기: {os.path.basename(init_from)}.zip")
        model = Algo.load(init_from, env=env, tensorboard_log=tb)
        # SB3는 learn() 재시작 시 스텝 카운터가 0으로 리셋되어 learning_starts(1만)
        # 스텝 동안 '완전 무작위 행동' 워밍업을 다시 하고, 리플레이 버퍼도 기본으론
        # 저장되지 않아 빈 버퍼의 소량 데이터로 업데이트하다 정책이 일시 망가진다
        # → 이어받기인데 성공률이 0부터 다시 오르는 것처럼 보이는 원인.
        if hasattr(model, 'learning_starts'):
            rb = init_from + '_replay.pkl'
            if os.path.exists(rb):
                # 이전 경험까지 통째로 이어받기 → 이음새 없이 바로 재개
                model.load_replay_buffer(rb)
                model.learning_starts = 0
                print(f"  리플레이 버퍼 이어받기: {os.path.basename(rb)}")
            else:
                # 버퍼가 없으면(다른 단계 모델을 이어받는 경우): 무작위 워밍업 1만 스텝.
                # '처음부터 정책 행동으로 버퍼 채우기'를 시도했다가 실패한 이력 있음 —
                # 잘 나는 궤적만 쌓이면 Q함수가 나쁜 행동의 가치를 과대평가해 발산
                # (critic_loss 폭주 → ent_coef 폭등 → 성공률 0.75→0.03 붕괴 실측).
                # 무작위 데이터가 Q함수를 접지시키므로 반드시 필요. 초반 로그의
                # 성공률 0은 무작위 행동 구간 표시일 뿐, 가중치는 정상 로드 상태.
                model.learning_starts = 10_000
                print("  리플레이 버퍼 없음 → 무작위 워밍업 1만 스텝 후 학습 재개"
                      " (초반 success_rate 0 은 정상)")
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

    if lr is not None:
        # 발산 재시작용: 낮은 학습률로 후반 진동/발산 스파이럴을 감쇠
        model.learning_rate = float(lr)
        model.lr_schedule = lambda _: float(lr)
        print(f"  학습률 재설정: {lr:g}")

    try:
        import tqdm, rich  # noqa: F401
        pbar = True
    except ImportError:
        pbar = False

    # 크래시 대비: 100k 스텝마다 ckpt/ 에 체크포인트 저장 (긴 학습 보호)
    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
    cbs = [CheckpointCallback(save_freq=100_000, save_path=os.path.join(_BASE, 'ckpt'),
                              name_prefix=os.path.basename(out))]
    # '마지막 모델 ≠ 최고 모델' 대책: eval_freq 스텝마다 순수 과제로 결정론 평가를
    # 돌려 역대 최고(평균 보상 기준) 모델을 따로 저장 — 성공률 밴드 상단을 확보.
    best_dir = os.path.join(_BASE, 'best', os.path.basename(out))
    cbs.append(EvalCallback(Monitor(DroneHouseEnv(obstacles=obs_pts, **pure_kw)),
                            best_model_save_path=best_dir, eval_freq=eval_freq,
                            n_eval_episodes=30, deterministic=True, verbose=1))

    # 발산 자동 감지: SAC 자동 엔트로피 계수가 초기값의 2.5배를 넘으면(critic 발산의
    # 선행 신호 — 두 차례 실측: 0.12→2.23, 0.11→0.27+) 학습을 스스로 멈춘다.
    # 정상 학습에서는 ent_coef 가 감소·횡보하므로 오탐 없음. best/ckpt 는 이미 저장돼 있음.
    from stable_baselines3.common.callbacks import BaseCallback

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
                print("       ckpt/·best 모델에서 --lr 1e-4 로 재시작 권장")
                return False
            return True

    cbs.append(StopOnDivergence())

    print("  (Ctrl+C 로 중단해도 마지막에 저장)")
    try:
        model.learn(total_timesteps=timesteps, progress_bar=pbar, callback=cbs)
    except KeyboardInterrupt:
        print("\n[학습] 중단됨 — 현재까지 모델 저장")
    # SB3 save()는 대상 파일을 먼저 비우고 쓰기 시작해서, 저장 중 크래시가 나면
    # 기존 모델까지 파괴된다(실제 사고 발생) → 임시 파일에 저장 후 원자적 교체.
    model.save(out + '_new')
    os.replace(out + '_new.zip', out + '.zip')
    print(f"[학습] 저장 완료: {out}.zip")
    if hasattr(model, 'save_replay_buffer'):     # SAC 등 off-policy만 해당
        model.save_replay_buffer(out + '_new_replay.pkl')
        os.replace(out + '_new_replay.pkl', out + '_replay.pkl')
        print(f"[학습] 리플레이 버퍼 저장: {os.path.basename(out)}_replay.pkl (이어받기용)")
    best_zip = os.path.join(best_dir, 'best_model.zip')
    if os.path.exists(best_zip):
        shutil.copyfile(best_zip, out + '_best.zip')
        print(f"[학습] 학습 중 최고 모델: {os.path.basename(out)}_best.zip")

    # 간이 평가 (deterministic, 순수 과제 기준, 시각화 없음) — 최종 vs best 비교
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

    print(f"\n[평가] 결정론 {eval_episodes} 에피소드 (순수 과제, 랜덤 시작/도착):")
    _quick_eval(model, '최종')
    if os.path.exists(out + '_best.zip'):
        _quick_eval(Algo.load(out + '_best'), 'best')
        print("  → 성공률 높은 쪽을 다음 단계 --init-from 으로 쓸 것")
    return out
