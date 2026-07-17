# -*- coding: utf-8 -*-
"""오라클(carrot 직진 추종) 계단 통과 시범을 리플레이 버퍼에 주입.

정책이 계단 립 앞에서 '학습된 공포'로 멈추는 문제의 직접 처방: SAC 는 off-policy 라
버퍼에 든 경험이면 누구 것이든 배운다. 오라클은 계단을 통과할 수 있음이 검증돼
있으므로, 그 궤적(성공 보상 포함)을 버퍼에 넣어 Q 함수를 접지시킨다.

  python inject_stair_demos.py            # model_geo(.zip/_replay.pkl) 에 주입 후 저장
  python inject_stair_demos.py --dry-run  # 저장 없이 생성/형상 검증만

주입 후 train_geo.py --init-from model_geo ... 로 이어받으면 시범이 함께 학습된다.
(ASCII 출력 전용 — cp949 파이프 크래시 방지)
"""
import argparse
import os
import numpy as np

_BASE = os.path.dirname(os.path.abspath(__file__))

KW = dict(curriculum=False, priv_obs=True, ray_max=4.0, ray_layout="horiz14",
          subgoal_dist=2.5, bump_penalty=2.0, clearance=0.12, max_steps=150)

EPISODES_PER_TASK = 15     # 체인 2개 x 방향 2 x 15 = 60 에피소드
ACTION_NOISE = 0.15        # 시범 다양화 (탐험 흉내)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model_geo")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    from stable_baselines3 import SAC
    from geo_env import DroneGeoEnv

    path = os.path.join(_BASE, a.model)
    rb_path = path + "_replay.pkl"
    model = SAC.load(path)
    model.load_replay_buffer(rb_path)
    rb = model.replay_buffer
    n_envs = rb.n_envs
    print(f"buffer loaded: pos={rb.pos} full={rb.full} n_envs={n_envs}")

    env = DroneGeoEnv(**KW)
    rng = np.random.default_rng(0)

    def rep(x):
        return np.repeat(np.asarray(x)[None], n_envs, axis=0)

    added = 0
    succ = 0
    total = 0
    for ci, ch in enumerate(env._chains):
        for direction in (0, 1):
            ends = (ch[0], ch[-1]) if direction == 0 else (ch[-1], ch[0])
            for ep in range(EPISODES_PER_TASK):
                start = ends[0] + rng.uniform(-0.6, 0.6, 3)
                obs, _ = env.reset(options={"start": start, "goal": ends[1]})
                term = trunc = False
                while not (term or trunc):
                    c = env._carrot()
                    v = c - env.pos
                    nv = float(np.linalg.norm(v))
                    act = v / nv if nv > 1e-6 else np.zeros(3)
                    act = np.clip(act + rng.normal(0, ACTION_NOISE, 3), -1, 1)
                    nobs, r, term, trunc, info = env.step(act)
                    if not a.dry_run or added < 5:
                        obs_b = ({k: rep(v_) for k, v_ in obs.items()}
                                 if isinstance(obs, dict) else rep(obs))
                        nobs_b = ({k: rep(v_) for k, v_ in nobs.items()}
                                  if isinstance(nobs, dict) else rep(nobs))
                        rb.add(obs_b, nobs_b, rep(act),
                               np.full(n_envs, r, dtype=np.float32),
                               np.full(n_envs, term, dtype=bool),
                               [{"TimeLimit.truncated": bool(trunc and not term)}] * n_envs)
                    obs = nobs
                    added += 1
                total += 1
                succ += int(info["is_success"])
    print(f"demo episodes: {succ}/{total} success, {added} transitions "
          f"(x{n_envs} env slots = {added*n_envs} entries)")
    if a.dry_run:
        print("dry-run: not saved")
        return
    model.save_replay_buffer(rb_path + ".new")
    os.replace(rb_path + ".new", rb_path)
    print(f"saved: {os.path.basename(rb_path)}")


if __name__ == "__main__":
    main()
