# -*- coding: utf-8 -*-
"""Definitive evaluation + failure anatomy: 200 episodes, uniform difficulty,
deterministic, solo policy. Compares final vs best model. (ASCII output)

Usage: conda run -n tello python _test_anatomy.py [n_episodes]
Writes anatomy_results.json next to this file.
"""
import json
import os
import sys
import numpy as np
from stable_baselines3 import SAC
from geo_env import DroneGeoEnv

_BASE = os.path.dirname(os.path.abspath(__file__))
N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
KW = dict(curriculum=False, priv_obs=True, ray_max=4.0, ray_layout="horiz14",
          subgoal_dist=2.5, bump_penalty=2.0, clearance=0.12, max_steps=700)


def run_model(name):
    path = os.path.join(_BASE, name)
    if not os.path.exists(path + ".zip"):
        print(f"[skip] {name}.zip not found")
        return None
    env = DroneGeoEnv(**KW)
    model = SAC.load(path)
    eps = []
    for i in range(N):
        obs, _ = env.reset(seed=7000 + i)
        start = env.pos.copy()
        goal = env.goal.copy()
        phi0 = float(env._phi(env.pos))
        traj = [env.pos.copy()]
        dmin = float(np.linalg.norm(goal - env.pos))
        term = trunc = False
        while not (term or trunc):
            act, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(act)
            traj.append(env.pos.copy())
            dmin = min(dmin, float(np.linalg.norm(goal - env.pos)))
        traj = np.asarray(traj)
        last = traj[-100:]
        bbox = float(np.linalg.norm(last.max(0) - last.min(0)))
        cross = abs(start[2] - goal[2]) > 1.5
        ep = dict(seed=7000 + i, success=bool(info["is_success"]),
                  steps=len(traj) - 1, phi0=round(phi0, 2), cross=bool(cross),
                  bumps=int(info["bumps"]), dmin=round(dmin, 2),
                  bbox=round(bbox, 2), end=[round(float(x), 1) for x in traj[-1]],
                  start_z=round(float(start[2]), 1), goal_z=round(float(goal[2]), 1))
        if not ep["success"]:
            if ep["bumps"] >= 150:
                ep["mode"] = "wedge"
            elif bbox < 1.0:
                ep["mode"] = "freeze"
            else:
                ep["mode"] = "wander"
            ep["near_miss"] = dmin < 1.5
        eps.append(ep)
        if (i + 1) % 25 == 0:
            sr = np.mean([e["success"] for e in eps])
            print(f"  [{name}] {i+1}/{N} running success {sr:.2f}", flush=True)
    return eps


def summarize(name, eps):
    ok = [e for e in eps if e["success"]]
    ng = [e for e in eps if not e["success"]]
    print(f"\n===== {name}: {len(ok)}/{len(eps)} = {len(ok)/len(eps):.3f} =====")
    for label, flt in (("same-floor", lambda e: not e["cross"]),
                       ("cross-floor", lambda e: e["cross"])):
        sub = [e for e in eps if flt(e)]
        if sub:
            s = sum(e["success"] for e in sub)
            print(f"  {label:11s} {s}/{len(sub)} = {s/len(sub):.3f}")
    print("  by distance band:")
    for lo, hi in ((0, 10), (10, 20), (20, 30), (30, 99)):
        sub = [e for e in eps if lo <= e["phi0"] < hi]
        if sub:
            s = sum(e["success"] for e in sub)
            print(f"    {lo:2d}-{hi:2d}m  {s:3d}/{len(sub):3d} = {s/len(sub):.3f}")
    if ng:
        print("  failure modes:")
        for m in ("wedge", "freeze", "wander"):
            sub = [e for e in ng if e["mode"] == m]
            nm = sum(e.get("near_miss", False) for e in sub)
            print(f"    {m:7s} {len(sub):3d} ({len(sub)/len(eps):.1%} of all)"
                  f"  near-miss(<1.5m of goal): {nm}")
        clusters = {}
        for e in ng:
            key = tuple(int(round(v)) for v in e["end"])
            clusters[key] = clusters.get(key, 0) + 1
        top = sorted(clusters.items(), key=lambda kv: -kv[1])[:6]
        print("  stall clusters (end pos, 1m rounded):")
        for k, c in top:
            if c >= 2:
                print(f"    {k}  x{c}")
        print("  failure seeds:", [e["seed"] for e in ng][:40],
              "..." if len(ng) > 40 else "")


results = {}
for name in ("model_geo", "model_geo_best"):
    eps = run_model(name)
    if eps:
        results[name] = eps
        summarize(name, eps)

with open(os.path.join(_BASE, "anatomy_results.json"), "w") as f:
    json.dump(results, f)
print("\nsaved anatomy_results.json")
