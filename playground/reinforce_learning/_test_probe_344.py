# -*- coding: utf-8 -*-
"""Interrogate the (3,4,4) stall cluster: roll policy to the stall, then hand
control to a carrot-following oracle. Oracle succeeds -> policy problem.
Oracle stalls too -> env/carrot/geometry problem. (ASCII output)"""
import sys
import numpy as np
from stable_baselines3 import SAC
from geo_env import DroneGeoEnv

KW = dict(curriculum=False, priv_obs=True, ray_max=4.0, ray_layout="horiz14",
          subgoal_dist=2.5, bump_penalty=2.0, clearance=0.12, max_steps=700)

MODEL = sys.argv[1] if len(sys.argv) > 1 else "model_geo_best"
print(f"[probe] model = {MODEL}")
model = SAC.load(MODEL)

for seed in (7001, 7027, 7012):
    env = DroneGeoEnv(**KW)
    obs, _ = env.reset(seed=seed)
    print(f"\n===== seed {seed}: start z={env.pos[2]:.1f} goal z={env.goal[2]:.1f} "
          f"phi0={env._phi(env.pos):.1f}m =====")
    # 1) policy until stall (50 steps without phi improvement)
    best_phi = env._phi(env.pos)
    since = 0
    term = trunc = False
    step = 0
    while not (term or trunc) and since < 50 and step < 650:
        act, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(act)
        step += 1
        phi = env._phi(env.pos)
        if phi < best_phi - 0.05:
            best_phi = phi
            since = 0
        else:
            since += 1
    if term or trunc:
        print(f"  policy finished before stalling (success={info['is_success']})")
        continue
    p = env.pos
    print(f"  policy stalled at step {step}: pos=({p[0]:.2f},{p[1]:.2f},{p[2]:.2f})"
          f" phi={env._phi(p):.2f} bumps={info['bumps']}")
    # carrot state at the stall
    c = env._carrot()
    d = float(np.linalg.norm(c - p))
    print(f"  carrot=({c[0]:.2f},{c[1]:.2f},{c[2]:.2f}) dist={d:.2f} "
          f"phi(carrot)={env._phi(c):.2f}  (dist<0.05 = fallback-on-self)")
    rays = env._raycast(p) * env.ray_max
    print(f"  rays min={rays.min():.2f}m (min direction idx {int(np.argmin(rays))}),"
          f" down-ray={rays[-1]:.2f}m up-ray={rays[-2]:.2f}m")
    # 2) oracle takeover: fly straight at the carrot, re-query every step
    print("  --- oracle takeover (300 steps max) ---")
    ok = False
    for t in range(300):
        c = env._carrot()
        v = c - env.pos
        n = float(np.linalg.norm(v))
        a = v / n if n > 1e-6 else np.zeros(3)
        obs, r, term, trunc, info = env.step(a)
        if t % 50 == 0 or term:
            q = env.pos
            print(f"    t={t:3d} pos=({q[0]:.1f},{q[1]:.1f},{q[2]:.1f}) "
                  f"phi={env._phi(q):.2f} bumps={info['bumps']}")
        if term:
            ok = info["is_success"]
            break
    print(f"  oracle result: {'SUCCESS' if ok else 'did not reach goal'} "
          f"(final phi={env._phi(env.pos):.2f}, bumps={info['bumps']})")
