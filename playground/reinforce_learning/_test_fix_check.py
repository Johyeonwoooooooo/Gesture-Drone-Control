# -*- coding: utf-8 -*-
"""Post-fix sanity: gymnasium check_env + random rollout + slide/hysteresis smoke.
(ASCII output only)"""
import numpy as np
from gymnasium.utils.env_checker import check_env
from geo_env import DroneGeoEnv

kw = dict(priv_obs=True, ray_max=4.0, ray_layout="horiz14", subgoal_dist=2.5,
          bump_penalty=2.0, clearance=0.12, max_steps=200)

env = DroneGeoEnv(curriculum=False, **kw)
check_env(env, skip_render_check=True)
print("check_env: OK")

# random rollout: no crash, slide actually moves the drone on bumps
rng = np.random.default_rng(0)
moved_on_bump = 0
bumps = 0
for ep in range(5):
    obs, _ = env.reset(seed=100 + ep)
    for _ in range(200):
        p0 = env.pos.copy()
        a = rng.uniform(-1, 1, 3)
        obs, r, term, trunc, info = env.step(a)
        if info["bumps"] > bumps:
            bumps = info["bumps"]
            if np.linalg.norm(env.pos - p0) > 1e-9:
                moved_on_bump += 1
        if term or trunc:
            break
    bumps = 0
print(f"random rollout: OK (slide moved drone on a bump {moved_on_bump} times)")

# curriculum mode still works
env2 = DroneGeoEnv(curriculum=True, **kw)
obs, _ = env2.reset(seed=1)
for _ in range(50):
    obs, r, term, trunc, info = env2.step(rng.uniform(-1, 1, 3))
    if term or trunc:
        obs, _ = env2.reset()
print("curriculum mode: OK, d_max =", info["d_max"])
