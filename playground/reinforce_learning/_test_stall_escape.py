# -*- coding: utf-8 -*-
"""Same 40 seeds, but with stall-escape: if no geodesic progress for 40 steps,
take 10 stochastic steps (jiggle), then back to deterministic. (ASCII output)"""
import numpy as np
from stable_baselines3 import SAC
from geo_env import DroneGeoEnv

MODEL = "model_geo_best_0716am"
N = 40
PATIENCE = 40   # steps without phi improvement -> escape
JIGGLE = 10     # stochastic steps per escape

env = DroneGeoEnv(curriculum=False, priv_obs=True, ray_max=4.0, ray_layout="horiz14",
                  subgoal_dist=2.5, bump_penalty=2.0, clearance=0.12, max_steps=600)
model = SAC.load(MODEL)

succ = 0
esc_used_in_succ = 0
for i in range(N):
    obs, _ = env.reset(seed=7000 + i)
    best_phi = env._phi(env.pos)
    since = 0
    jig = 0
    escapes = 0
    term = trunc = False
    steps = 0
    while not (term or trunc):
        det = jig <= 0
        act, _ = model.predict(obs, deterministic=det)
        obs, r, term, trunc, info = env.step(act)
        steps += 1
        jig -= 1
        phi = env._phi(env.pos)
        if phi < best_phi - 0.05:
            best_phi = phi
            since = 0
        else:
            since += 1
        if since >= PATIENCE and jig <= 0:
            jig = JIGGLE
            since = 0
            escapes += 1
    ok = info["is_success"]
    succ += int(ok)
    if ok and escapes:
        esc_used_in_succ += 1
    print(f"ep{i:02d} {'OK  ' if ok else 'FAIL'} steps={steps:3d} escapes={escapes:2d}"
          f" bumps={info['bumps']:3d}")

print()
print(f"success {succ}/{N}  (baseline was 9/{N})")
print(f"successes that needed escape: {esc_used_in_succ}")
