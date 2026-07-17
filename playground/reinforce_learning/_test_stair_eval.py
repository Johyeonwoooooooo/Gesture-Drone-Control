# -*- coding: utf-8 -*-
"""Direct stair-task success measurement: stair_mix=1.0 episodes, deterministic
policy, split by direction (descend/ascend). (ASCII output)

Usage: conda run -n tello python _test_stair_eval.py [model] [n]
"""
import sys
import numpy as np
from stable_baselines3 import SAC
from geo_env import DroneGeoEnv

MODEL = sys.argv[1] if len(sys.argv) > 1 else "model_geo_best"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 40

env = DroneGeoEnv(curriculum=True, priv_obs=True, ray_max=4.0, ray_layout="horiz14",
                  subgoal_dist=2.5, bump_penalty=2.0, clearance=0.12, max_steps=700,
                  stair_mix=1.0)
model = SAC.load(MODEL)
print(f"[stair eval] model = {MODEL}, n = {N}")

env.reset(seed=123)
res = {"down": [0, 0], "up": [0, 0]}
stalls = {}
for i in range(N):
    obs, _ = env.reset()
    start, goal = env.pos.copy(), env.goal.copy()
    d = "down" if goal[2] < start[2] - 0.5 else "up"
    term = trunc = False
    while not (term or trunc):
        act, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(act)
    ok = bool(info["is_success"])
    res[d][1] += 1
    res[d][0] += int(ok)
    tag = "OK " if ok else "FAIL"
    if not ok:
        key = tuple(int(round(v)) for v in env.pos)
        stalls[key] = stalls.get(key, 0) + 1
    print(f"  ep{i:02d} {d:4s} z {start[2]:5.2f}->{goal[2]:5.2f} phi0={env._phi(start):5.1f} "
          f"{tag} steps={env.steps} bumps={info['bumps']} end=({env.pos[0]:.1f},{env.pos[1]:.1f},{env.pos[2]:.1f})",
          flush=True)
print("\n[result]")
for k, (s, n) in res.items():
    if n:
        print(f"  {k:4s}: {s}/{n} = {s/n:.2f}")
tot_s = sum(v[0] for v in res.values()); tot_n = sum(v[1] for v in res.values())
print(f"  all : {tot_s}/{tot_n} = {tot_s/max(tot_n,1):.2f}")
if stalls:
    print("  fail end clusters:", sorted(stalls.items(), key=lambda kv: -kv[1])[:6])
