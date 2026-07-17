# -*- coding: utf-8 -*-
"""Zero-shot dynamic-obstacle eval: same tasks with/without patrolling people.
People are visible to rays/physics but NOT to the geodesic field -> avoidance
must come from the policy's reactive skill. (ASCII output)

Usage: conda run -n tello python _test_people_eval.py [model] [n_eps] [n_people]
"""
import os
import sys
import numpy as np
from stable_baselines3 import SAC
from geo_env import DroneGeoEnv

_BASE = os.path.dirname(os.path.abspath(__file__))
MODEL = sys.argv[1] if len(sys.argv) > 1 else "model_geo_best"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 100
K = int(sys.argv[3]) if len(sys.argv) > 3 else 2

KW = dict(curriculum=False, priv_obs=True, ray_max=4.0, ray_layout="horiz14",
          subgoal_dist=2.5, bump_penalty=2.0, clearance=0.12, max_steps=700)
model = SAC.load(os.path.join(_BASE, MODEL))


def run(env, seed):
    obs, _ = env.reset(seed=seed)
    term = trunc = False
    info = {}
    while not (term or trunc):
        act, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(act)
    return dict(success=bool(info["is_success"]), steps=env.steps,
                bumps=int(info["bumps"]), phits=int(info.get("person_hits", 0)),
                n_people=len(env._people))


env0 = DroneGeoEnv(**KW)                      # baseline
envP = DroneGeoEnv(people=K, **KW)            # with people
print(f"[people eval] model={MODEL} n={N} people={K}", flush=True)

base, ppl = [], []
for i in range(N):
    s = 8000 + i
    base.append(run(env0, s))
    ppl.append(run(envP, s))
    if (i + 1) % 20 == 0:
        b = np.mean([e["success"] for e in base])
        p = np.mean([e["success"] for e in ppl])
        pc = np.mean([e["success"] and e["phits"] == 0 for e in ppl])
        print(f"  {i+1}/{N}  base {b:.2f} | people {p:.2f} (contact-free {pc:.2f})",
              flush=True)

spawned = [e for e in ppl if e["n_people"] > 0]
print(f"\n===== results ({N} identical tasks) =====")
print(f"  baseline (0 people)    : "
      f"{sum(e['success'] for e in base)}/{N} = "
      f"{np.mean([e['success'] for e in base]):.3f}")
print(f"  with {K} people          : "
      f"{sum(e['success'] for e in ppl)}/{N} = "
      f"{np.mean([e['success'] for e in ppl]):.3f}")
cf = [e["success"] and e["phits"] == 0 for e in ppl]
print(f"  contact-free success   : {sum(cf)}/{N} = {np.mean(cf):.3f}")
touched = [e for e in ppl if e["phits"] > 0]
print(f"  episodes w/ any contact: {len(touched)}/{N}"
      f"  (avg hits when touched {np.mean([e['phits'] for e in touched]):.1f})"
      if touched else "  episodes w/ any contact: 0")
print(f"  people actually spawned: {len(spawned)}/{N}")
# where did it get worse? tasks that flipped success->fail
flip = [8000 + i for i in range(N) if base[i]["success"] and not ppl[i]["success"]]
print(f"  success->fail flips    : {len(flip)}  seeds {flip[:20]}")
st0 = np.mean([e["steps"] for e in base if e["success"]])
st1 = np.mean([e["steps"] for e in ppl if e["success"]])
print(f"  success steps          : base {st0:.0f} -> people {st1:.0f} (우회 비용)")
