# -*- coding: utf-8 -*-
"""Functional check of guide_clearance fix: explicit cross-floor tasks between
known-real points. Policy (deterministic) and oracle, old traps included.
(ASCII output)

Usage: conda run -n tello python _test_guide_fix.py [model]
"""
import sys
import numpy as np
from stable_baselines3 import SAC
from geo_env import DroneGeoEnv

KW = dict(curriculum=False, priv_obs=True, ray_max=4.0, ray_layout="horiz14",
          subgoal_dist=2.5, bump_penalty=2.0, clearance=0.12, max_steps=700)
MODEL = sys.argv[1] if len(sys.argv) > 1 else "model_geo_best"
model = SAC.load(MODEL)
env = DroneGeoEnv(**KW)
print(f"[guide-fix check] model = {MODEL}")

TASKS = [
    ("top->bottom (old 7001 trap approach)", [9.6, 1.9, 4.4], [7.0, 1.6, -1.5]),
    ("top->mid", [3.6, 1.5, 4.4], [12.0, 5.0, 1.6]),
    ("top far->bottom far", [12.0, 1.0, 4.4], [1.0, 5.0, -1.6]),
    ("bottom->top", [7.0, 1.6, -1.5], [9.6, 1.9, 4.4]),
    ("mid->top (old freeze2 area on route)", [1.3, -0.4, 1.1], [7.6, 4.0, 4.1]),
]


def run(policy_fn, tag, start, goal):
    obs, _ = env.reset(options={'start': np.array(start, float),
                                'goal': np.array(goal, float)})
    term = trunc = False
    info = {}
    while not (term or trunc):
        a = policy_fn(obs)
        obs, r, term, trunc, info = env.step(a)
    ok = bool(info.get('is_success'))
    p = env.pos
    print(f"    [{tag}] {'OK ' if ok else 'FAIL'} steps={env.steps} "
          f"bumps={info['bumps']} end=({p[0]:.1f},{p[1]:.1f},{p[2]:.1f}) "
          f"remain={info['dist']:.1f}m", flush=True)
    return ok


def pol(obs):
    a, _ = model.predict(obs, deterministic=True)
    return a


def orc(obs):
    c = env._carrot()
    v = c - env.pos
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-6 else np.zeros(3)


okp = oko = 0
for name, s, g in TASKS:
    print(f"  {name}: start={s} goal={g}")
    okp += run(pol, "policy", s, g)
    oko += run(orc, "oracle", s, g)
print(f"\n[summary] policy {okp}/{len(TASKS)}  oracle {oko}/{len(TASKS)}")
