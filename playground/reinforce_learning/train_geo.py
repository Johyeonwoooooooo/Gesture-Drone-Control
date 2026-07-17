"""
train_geo.py — v2(측지 보상 + 자동 커리큘럼) 밤샘 학습
──────────────────────────────────────────────────────
수동 커리큘럼 없이 처음부터 전체집(층 넘기 포함)을 한 번에 학습한다.
난이도는 환경이 성공률을 보고 스스로 올린다(geo_env 의 d_max).

  python train_geo.py                    # 새 모델, 6개 병렬 env, 22시간(시간제한) 학습
  python train_geo.py --hours 10         # 시간 예산 변경
  python train_geo.py --n-envs 1         # 병렬 없이 (디버그)
  python train_geo.py --init-from model_geo   # 이어받기 (버퍼 자동 로드)

fps 는 머신 상태에 따라 변하므로 스텝 수가 아니라 '시간'으로 멈춘다(--hours, 기본 22h).
--timesteps 는 안전 상한일 뿐(기본 20M — 보통 시간제한이 먼저 온다).

병렬 env 로 CPU 병목(KDTree)을 우회 — 6 env ≈ 6배 처리량.
밤샘 안전장치: 100k마다 ckpt, 평가 최고 모델 자동 저장(model_geo_best),
ent_coef 폭등 시 발산 자동 중단, 원자적 저장(+리플레이 버퍼).
학습 곡선: rollout/success_rate 는 '현재 난이도(d_max)'에서의 성공률이므로
d_max(텐서보드 rollout/d_max)와 같이 봐야 한다 — 성공률이 0.5 부근을 유지하며
d_max 가 꾸준히 오르는 그림이 정상(커리큘럼이 난이도를 밀어올리는 중).
"""
import os
import argparse
import numpy as np

_BASE = os.path.dirname(os.path.abspath(__file__))


def make_env(rank, curriculum=True, d_init=4.0, ray_max=2.0, ray_layout='axes14',
             subgoal=None, bump=None, clearance=None, max_steps=400, stair_mix=0.0,
             guide=0.20, people=0, people_penalty=None):
    def _f():
        from stable_baselines3.common.monitor import Monitor
        from geo_env import DroneGeoEnv
        kw = {} if clearance is None else {'clearance': clearance}
        env = DroneGeoEnv(curriculum=curriculum, priv_obs=True, d_init=d_init,
                          ray_max=ray_max, ray_layout=ray_layout, subgoal_dist=subgoal,
                          bump_penalty=bump, max_steps=max_steps, stair_mix=stair_mix,
                          guide_clearance=guide, people=people,
                          people_penalty=people_penalty, **kw)
        env.reset(seed=1000 + rank)
        return Monitor(env, info_keywords=('d_max',))
    return _f


def main():
    ap = argparse.ArgumentParser(description='v2: 측지 보상 + 자동 커리큘럼 (전체집)')
    ap.add_argument('--timesteps', type=int, default=20_000_000,
                    help='스텝 상한(안전용). 보통 --hours 가 먼저 끊는다')
    ap.add_argument('--hours', type=float, default=22.0,
                    help='학습 시간 예산(시간). 지나면 저장하고 종료')
    ap.add_argument('--n-envs', type=int, default=6)
    ap.add_argument('--d-init', type=float, default=4.0,
                    help='커리큘럼 시작 난이도(m). 재개 시 직전 d_max 로 주면 워밍업 생략')
    ap.add_argument('--ray-max', type=float, default=2.0,
                    help='거리센서 사거리(m). 늘리면 문이 멀리서 보여 천장이 올라감'
                         ' (관측 차원 불변 → 이어받기 가능. 평가 시에도 같은 값 필요)')
    ap.add_argument('--rays', choices=['axes14', 'horiz14'], default='axes14',
                    help='레이 배치. horiz14=수평 12방향(30°)+상하 2 — 원거리 문 탐지에'
                         ' 유리 (차원 동일. 평가 시에도 같은 값 필요)')
    ap.add_argument('--subgoal', type=float, default=None,
                    help='설정(m) 시 계층 모드: 관측 목표를 측지 내리막 carrot 으로 교체.'
                         ' 순수 정책 천장(~11m) 해소. 2.5 권장 (평가 시에도 같은 값 필요)')
    ap.add_argument('--bump', type=float, default=None,
                    help='설정 시 범퍼 물리: 충돌해도 안 죽고 제자리 + 이 페널티.'
                         ' 계단 통과 학습용. 2 권장 (평가 시에도 같은 값 필요)')
    ap.add_argument('--max-steps', type=int, default=400,
                    help='에피소드 스텝 예산. 계단 포함 원거리 과제는 700 권장 —'
                         ' 예산이 짧으면 어려운 과제가 전부 타임아웃 처리되어'
                         ' 커리큘럼이 그 난이도에서 정체함')
    ap.add_argument('--clearance', type=float, default=None,
                    help='드론 여유반경(m) 전역 지정. 0.12 = 계단 포함 전 공간 통일'
                         ' (기본 0.235는 계단 통과 불가라 구역 완화에 의존).'
                         ' 평가 시에도 같은 값 필요')
    ap.add_argument('--guide-clearance', type=float, default=0.20,
                    help='안내(측지 그래프) 전용 최소 여유(m). 물리와 별도 — 여유가 이보다'
                         ' 좁은 틈새(키홀)는 필드가 경유하지 않음. 계단 체인 근처는'
                         ' 예외적으로 stair_clearance 완화. 평가 시에도 같은 값 필요')
    ap.add_argument('--target-entropy', type=float, default=None,
                    help='SAC target_entropy 직접 지정(기본 auto=-3). -1.5 등으로 올리면'
                         ' 탐험 붕괴(ent_coef 바닥 고착) 완화 — fresh 학습에서만 유효')
    ap.add_argument('--stair-mix', type=float, default=0.0,
                    help='이 확률로 계단 과외 에피소드(계단 끝→반대 끝 짧은 층간 과제)를'
                         ' 출제. 계단 통과를 집중 연습시킬 때 0.2 권장')
    ap.add_argument('--people', type=int, default=0,
                    help='에피소드마다 예정 경로 근처를 순찰하는 사람(동적 장애물) 수.'
                         ' 지도/carrot 은 모르고 레이·물리만 봄 — 회피 학습용. 2 권장')
    ap.add_argument('--people-penalty', type=float, default=None,
                    help='사람 접촉 페널티(기본 bump 와 동일). 회피를 가르치려면'
                         ' 벽 범프보다 크게 — 5 권장')
    ap.add_argument('--out', default='model_geo')
    ap.add_argument('--init-from', default=None, help='이어받을 모델(.zip 제외)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--lr', type=float, default=3e-4)
    a = ap.parse_args()

    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import (CheckpointCallback, EvalCallback,
                                                    BaseCallback)
    from geo_env import DroneGeoEnv
    from drone_env import load_obstacles

    out = os.path.join(_BASE, a.out)
    print(f"\n{'='*55}\n[v2 · 측지 보상 + 자동 커리큘럼] 학습 시작\n{'='*55}")
    print(f"  시간예산 {a.hours:.1f}h | 스텝상한 {a.timesteps:,} | n_envs={a.n_envs} | out={a.out}.zip")

    # 캐시(장애물/계단 체인)를 병렬 프로세스가 동시에 만들다 충돌하지 않게 먼저 생성
    load_obstacles()
    probe = DroneGeoEnv()
    del probe

    fns = [make_env(i, d_init=a.d_init, ray_max=a.ray_max, ray_layout=a.rays,
                    subgoal=a.subgoal, bump=a.bump, clearance=a.clearance,
                    max_steps=a.max_steps, stair_mix=a.stair_mix,
                    guide=a.guide_clearance, people=a.people,
                    people_penalty=a.people_penalty)
           for i in range(a.n_envs)]
    venv = SubprocVecEnv(fns) if a.n_envs > 1 else DummyVecEnv(fns)

    tb = os.path.join(_BASE, 'runs')
    if a.init_from:
        init = os.path.join(_BASE, a.init_from)
        print(f"  이어받기: {a.init_from}.zip")
        model = SAC.load(init, env=venv, tensorboard_log=tb)
        rb = init + '_replay.pkl'
        if os.path.exists(rb):
            model.load_replay_buffer(rb)
            model.learning_starts = 0
            print(f"  리플레이 버퍼 이어받기: {os.path.basename(rb)}")
        else:
            model.learning_starts = 10_000
            print("  버퍼 없음 → 무작위 워밍업 (v2 는 새 관측이라 새로 시작 권장)")
    else:
        from asym_policy import AsymSACPolicy
        model = SAC(AsymSACPolicy, venv, verbose=1, seed=a.seed,
                    learning_rate=a.lr, buffer_size=1_000_000, batch_size=256,
                    gamma=0.99, tau=0.005, learning_starts=20_000,
                    train_freq=1, gradient_steps=max(1, a.n_envs // 2),
                    target_entropy=('auto' if a.target_entropy is None
                                    else a.target_entropy),
                    policy_kwargs=dict(share_features_extractor=False),
                    tensorboard_log=tb)

    # d_max(자동 커리큘럼 난이도) 를 텐서보드에 기록 + 주기적으로 콘솔에 표시
    class LogDmax(BaseCallback):
        def __init__(self, n_envs):
            super().__init__()
            self.print_every = max(1, 20_000 // n_envs)   # 약 2만 env스텝마다 한 줄
            self.last_dm = None

        def _on_step(self):
            dm = [i.get('d_max') for i in self.locals.get('infos', []) if 'd_max' in i]
            if dm:
                self.last_dm = float(np.mean(dm))
                # goal/ 섹션은 알파벳순으로 rollout/ 위에 표시됨. 매 스텝 기록해야
                # 로그 표가 뜰 때마다 항상 실린다 (표는 몇십 스텝마다 갱신됨).
                self.logger.record('goal/d_max_m', round(self.last_dm, 2))
                self.logger.record('rollout/d_max', self.last_dm)   # 기존 텐서보드 곡선 유지
            if self.n_calls % self.print_every == 0 and self.last_dm is not None:
                sb = getattr(self.model, 'ep_success_buffer', None)
                sr = f" | 최근 성공률 {float(np.mean(sb)):.2f}" if sb else ""
                print(f"[진행] step {self.num_timesteps:,} | 출제 난이도 d_max "
                      f"{self.last_dm:.2f} m{sr}", flush=True)
            return True

    class StopOnWallClock(BaseCallback):
        """시간 예산이 다 되면 학습을 스스로 멈춘다 (이후 저장 루틴은 정상 진행)."""
        def __init__(self, hours):
            super().__init__()
            self.deadline = None
            self.hours = float(hours)

        def _on_training_start(self):
            import time
            self.deadline = time.time() + self.hours * 3600

        def _on_step(self):
            if self.n_calls % 200 == 0:
                import time
                if time.time() >= self.deadline:
                    print(f"\n[시간제한] {self.hours:.1f}h 경과, 학습 종료 후 저장"
                          f" (step {self.num_timesteps:,})")
                    return False
            return True

    class StopOnDivergence(BaseCallback):
        def __init__(self):
            super().__init__()
            self.base = None

        def _on_step(self):
            if self.n_calls % 1000 != 0 or getattr(self.model, 'log_ent_coef', None) is None:
                return True
            import torch as th
            v = float(th.exp(self.model.log_ent_coef.detach()).item())
            if self.base is None:
                self.base = max(v, 1e-3)
            elif v > 2.5 * self.base:
                print(f"\n[경고] ent_coef {self.base:.3f}->{v:.3f} 폭등: 발산 감지, 자동 중단")
                return False
            return True

    _ckw = {} if a.clearance is None else {'clearance': a.clearance}
    eval_env = Monitor(DroneGeoEnv(curriculum=False, priv_obs=True, ray_max=a.ray_max,
                                   ray_layout=a.rays, subgoal_dist=a.subgoal,
                                   bump_penalty=a.bump, max_steps=a.max_steps,
                                   guide_clearance=a.guide_clearance, people=a.people,
                                   people_penalty=a.people_penalty, **_ckw))
    cbs = [
        CheckpointCallback(save_freq=max(1, 100_000 // a.n_envs),
                           save_path=os.path.join(_BASE, 'ckpt'),
                           name_prefix=a.out),
        EvalCallback(eval_env, best_model_save_path=os.path.join(_BASE, 'best', a.out),
                     eval_freq=max(1, 60_000 // a.n_envs), n_eval_episodes=30,
                     deterministic=True, verbose=1),
        LogDmax(a.n_envs), StopOnWallClock(a.hours), StopOnDivergence(),
    ]

    try:
        import tqdm, rich  # noqa: F401
        pbar = True
    except ImportError:
        pbar = False
    print("  (Ctrl+C 로 중단해도 마지막에 저장)")
    try:
        model.learn(total_timesteps=a.timesteps, progress_bar=pbar, callback=cbs)
    except KeyboardInterrupt:
        print("\n[학습] 중단됨: 현재까지 모델 저장")

    model.save(out + '_new')                              # 원자적 저장 (사고 이력 대책)
    os.replace(out + '_new.zip', out + '.zip')
    model.save_replay_buffer(out + '_new_replay.pkl')
    os.replace(out + '_new_replay.pkl', out + '_replay.pkl')
    print(f"[학습] 저장: {a.out}.zip / {a.out}_replay.pkl")
    best_zip = os.path.join(_BASE, 'best', a.out, 'best_model.zip')
    if os.path.exists(best_zip):
        import shutil
        shutil.copyfile(best_zip, out + '_best.zip')
        print(f"[학습] 평가 최고 모델: {a.out}_best.zip")

    # 간이 평가: 전 난이도 균일 30 에피소드 (최종 vs best)
    def _quick(m, tag):
        e = DroneGeoEnv(curriculum=False, priv_obs=True, ray_max=a.ray_max,
                        ray_layout=a.rays, subgoal_dist=a.subgoal, bump_penalty=a.bump,
                        max_steps=a.max_steps, guide_clearance=a.guide_clearance,
                        people=a.people, people_penalty=a.people_penalty, **_ckw)
        succ = 0
        for i in range(30):
            obs, _ = e.reset(seed=7000 + i)
            term = trunc = False
            while not (term or trunc):
                act, _ = m.predict(obs, deterministic=True)
                obs, r, term, trunc, info = e.step(act)
            succ += int(info['is_success'])
        print(f"  [{tag}] 성공률 {succ}/30 = {succ/30:.2f}")

    print("\n[평가] 결정론 30 에피소드 (전체집, 전 난이도):")
    _quick(model, '최종')
    if os.path.exists(out + '_best.zip'):
        _quick(SAC.load(out + '_best'), 'best')
        print("  → 높은 쪽을 이어받기/시연에 사용")
    venv.close()


if __name__ == '__main__':
    main()
