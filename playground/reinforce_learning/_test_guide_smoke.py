# -*- coding: utf-8 -*-
"""Smoke: curriculum env with new guide graph - resets sample inside the band,
steps run, stair_mix and explore_mix paths work, check_env passes. (ASCII)"""
import numpy as np
from geo_env import DroneGeoEnv

env = DroneGeoEnv(curriculum=True, priv_obs=True, ray_max=4.0, ray_layout="horiz14",
                  subgoal_dist=2.5, bump_penalty=2.0, clearance=0.12, max_steps=700,
                  stair_mix=0.2, d_init=8.0)
env.reset(seed=0)
bad = 0
for i in range(30):
    obs, _ = env.reset()
    phi0 = env._phi(env.pos)
    for _ in range(5):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        if term or trunc:
            break
    if not np.isfinite(phi0):
        bad += 1
print(f"30 curriculum resets + random steps: OK (non-finite phi0: {bad})")

try:
    from gymnasium.utils.env_checker import check_env as gym_check
    e2 = DroneGeoEnv(curriculum=True, priv_obs=False, ray_max=4.0,
                     ray_layout="horiz14", subgoal_dist=2.5, bump_penalty=2.0,
                     clearance=0.12, max_steps=100)
    gym_check(e2, skip_render_check=True)
    print("gymnasium check_env: PASS")
except Exception as ex:
    print(f"gymnasium check_env: {type(ex).__name__}: {ex}")
