# -*- coding: utf-8 -*-
"""stair_mix smoke: are stair episodes actually cross-floor short tasks? (ASCII)"""
import numpy as np
from geo_env import DroneGeoEnv

env = DroneGeoEnv(curriculum=True, priv_obs=True, ray_max=4.0, ray_layout="horiz14",
                  subgoal_dist=2.5, bump_penalty=2.0, clearance=0.12, max_steps=700,
                  stair_mix=1.0)
env.reset(seed=0)
cross = 0
for i in range(12):
    obs, _ = env.reset()
    dz = abs(env.pos[2] - env.goal[2])
    phi0 = env._phi(env.pos)
    cross += int(dz > 1.5)
    print(f"ep{i:02d} start z={env.pos[2]:5.2f} goal z={env.goal[2]:5.2f} "
          f"|dz|={dz:4.2f} phi0={phi0:5.2f}m explore_ep={env._explore_ep}")
print(f"cross-floor {cross}/12")

# stair_mix=0 regression: normal curriculum episodes unaffected
env2 = DroneGeoEnv(curriculum=True, priv_obs=True, stair_mix=0.0)
env2.reset(seed=1)
for _ in range(3):
    env2.reset()
print("stair_mix=0 resets: OK")
