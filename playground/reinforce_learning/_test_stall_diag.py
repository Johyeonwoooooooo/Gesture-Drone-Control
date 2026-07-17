# -*- coding: utf-8 -*-
"""Stall diagnosis: where do timeout failures get stuck? (ASCII output only)"""
import numpy as np
from stable_baselines3 import SAC
from geo_env import DroneGeoEnv

MODEL = "model_geo_best_0716am"
N = 40

env = DroneGeoEnv(curriculum=False, priv_obs=True, ray_max=4.0, ray_layout="horiz14",
                  subgoal_dist=2.5, bump_penalty=2.0, clearance=0.12, max_steps=600)
model = SAC.load(MODEL)

fails = []
succ = 0
for i in range(N):
    obs, _ = env.reset(seed=7000 + i)
    phi0 = env._phi(env.pos)
    traj = [env.pos.copy()]
    phis = [phi0]
    term = trunc = False
    while not (term or trunc):
        act, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(act)
        traj.append(env.pos.copy())
        phis.append(env._phi(env.pos))
    traj = np.asarray(traj)
    phis = np.asarray(phis)
    if info["is_success"]:
        succ += 1
        print(f"ep{i:02d} OK   phi0={phi0:5.1f}m steps={len(traj)-1:3d} bumps={info['bumps']}")
    else:
        k = int(np.argmin(phis))                       # closest approach (geodesic)
        last100 = traj[-100:]
        drift = float(np.linalg.norm(last100.max(0) - last100.min(0)))  # bbox diag of last 100 steps
        stall = traj[k]
        print(f"ep{i:02d} FAIL phi0={phi0:5.1f}m phi_min={phis[k]:5.1f}m at step {k:3d}"
              f" phi_end={phis[-1]:5.1f}m last100_bbox={drift:4.1f}m bumps={info['bumps']:3d}"
              f" stall=({stall[0]:5.1f},{stall[1]:5.1f},{stall[2]:5.1f})"
              f" goal_z={env.goal[2]:4.1f} start_z={traj[0][2]:4.1f}")
        fails.append((phi0, phis[k], k, drift, info["bumps"], stall, phis[-1]))

print()
print(f"success {succ}/{N}")
if fails:
    ph0 = np.array([f[0] for f in fails]); phm = np.array([f[1] for f in fails])
    drf = np.array([f[3] for f in fails]); bmp = np.array([f[4] for f in fails])
    print(f"fails: phi0 mean {ph0.mean():.1f}m | closest-approach mean {phm.mean():.1f}m")
    print(f"  progress made >70%% of the way: {(phm < 0.3*ph0).sum()}/{len(fails)}")
    print(f"  barely moved (<30%% progress) : {(phm > 0.7*ph0).sum()}/{len(fails)}")
    print(f"  last-100-step bbox <1m (dither in place): {(drf < 1.0).sum()}/{len(fails)}")
    print(f"  bumps mean {bmp.mean():.0f} / max {bmp.max()}")
