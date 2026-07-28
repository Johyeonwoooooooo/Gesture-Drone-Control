# -*- coding: utf-8 -*-
"""500-episode uniform evaluation, geodesic distance restricted to [4, 32] m.
Deterministic, solo policy. Rejection-samples seeds until 500 tasks in band.
(ASCII output)

Usage: conda run -n tello python _test_eval500.py [model] [n]
Writes eval500_results.json next to this file.
"""
import json
import os
import sys
import numpy as np
from stable_baselines3 import SAC
from geo_env import DroneGeoEnv

_BASE = os.path.dirname(os.path.abspath(__file__))
MODEL = sys.argv[1] if len(sys.argv) > 1 else "model_geo_best"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 500
D_LO, D_HI = 4.0, 32.0

KW = dict(curriculum=False, priv_obs=True, ray_max=4.0, ray_layout="horiz14",
          subgoal_dist=2.5, bump_penalty=2.0, clearance=0.12, max_steps=700)
env = DroneGeoEnv(**KW)
model = SAC.load(os.path.join(_BASE, MODEL))
print(f"[eval500] model={MODEL}  n={N}  band=[{D_LO},{D_HI}]m", flush=True)

eps = []
seed = 8000
tried = 0
while len(eps) < N:
    obs, _ = env.reset(seed=seed)
    seed += 1
    tried += 1
    phi0 = float(env._phi(env.pos))
    if not (D_LO <= phi0 <= D_HI):
        continue
    start, goal = env.pos.copy(), env.goal.copy()
    term = trunc = False
    dmin = float(np.linalg.norm(goal - env.pos))
    traj_last = [env.pos.copy()]
    while not (term or trunc):
        act, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(act)
        dmin = min(dmin, float(np.linalg.norm(goal - env.pos)))
        traj_last.append(env.pos.copy())
        if len(traj_last) > 100:
            traj_last.pop(0)
    bbox = float(np.linalg.norm(np.max(traj_last, 0) - np.min(traj_last, 0)))
    ok = bool(info["is_success"])
    ep = dict(seed=seed - 1, success=ok, steps=env.steps, phi0=round(phi0, 2),
              cross=bool(abs(start[2] - goal[2]) > 1.5), bumps=int(info["bumps"]),
              dmin=round(dmin, 2),
              end=[round(float(x), 1) for x in env.pos])
    if not ok:
        ep["mode"] = ("wedge" if ep["bumps"] >= 150 else
                      "freeze" if bbox < 1.0 else "wander")
    eps.append(ep)
    if len(eps) % 50 == 0:
        sr = np.mean([e["success"] for e in eps])
        print(f"  {len(eps)}/{N} (tried {tried})  running success {sr:.3f}",
              flush=True)

ok = [e for e in eps if e["success"]]
ng = [e for e in eps if not e["success"]]
print(f"\n===== {MODEL}: {len(ok)}/{len(eps)} = {len(ok)/len(eps):.3f} =====")
for label, flt in (("same-floor", lambda e: not e["cross"]),
                   ("cross-floor", lambda e: e["cross"])):
    sub = [e for e in eps if flt(e)]
    if sub:
        s = sum(e["success"] for e in sub)
        print(f"  {label:11s} {s}/{len(sub)} = {s/len(sub):.3f}")
print("  by distance band:")
for lo, hi in ((4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32)):
    sub = [e for e in eps if lo <= e["phi0"] < hi + (1 if hi == 32 else 0)]
    if sub:
        s = sum(e["success"] for e in sub)
        print(f"    {lo:2d}-{hi:2d}m  {s:3d}/{len(sub):3d} = {s/len(sub):.3f}")
if ok:
    st = [e["steps"] for e in ok]
    print(f"  success steps: mean {np.mean(st):.0f} / p90 {np.percentile(st, 90):.0f}"
          f" / max {max(st)}")
if ng:
    print("  failures:")
    for e in ng:
        print(f"    seed={e['seed']} phi0={e['phi0']:5.1f} cross={e['cross']} "
              f"mode={e['mode']} bumps={e['bumps']} dmin={e['dmin']:.1f} end={e['end']}")

with open(os.path.join(_BASE, "eval500_results.json"), "w") as f:
    json.dump(eps, f)
print("\nsaved eval500_results.json")
