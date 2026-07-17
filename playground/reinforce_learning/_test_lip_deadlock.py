# -*- coding: utf-8 -*-
"""Dissect the oracle freeze at the stair lip: which _carrot branch fires,
is the carrot reachable, which single-axis moves are physically open.
(ASCII output only)

Usage: conda run -n tello python _test_lip_deadlock.py [model] [seed ...]
"""
import sys
import numpy as np
from stable_baselines3 import SAC
from geo_env import DroneGeoEnv
import rrt_star_drone as R_mod  # noqa: F401
import geo_env as G

R = G.R
KW = dict(curriculum=False, priv_obs=True, ray_max=4.0, ray_layout="horiz14",
          subgoal_dist=2.5, bump_penalty=2.0, clearance=0.12, max_steps=700)

MODEL = sys.argv[1] if len(sys.argv) > 1 else "model_geo_best"
SEEDS = [int(s) for s in sys.argv[2:]] or [7001, 7012]
model = SAC.load(MODEL)
print(f"[deadlock probe] model = {MODEL}")


def carrot_verbose(env):
    """Re-run _carrot logic and report which branch produced the result."""
    pos = env.pos
    c = np.clip(np.round((pos - env.bounds_lo) / env.grid_v).astype(int),
                0, env.grid_dims - 1)
    n = env.nid[c[0], c[1], c[2]]
    cell_ok = n >= 0 and np.isfinite(env._field[n])
    prev = None if env._carrot_prev is None else env._carrot_prev.copy()
    got = env._carrot()
    branch = "?"
    if np.allclose(got, env.goal):
        branch = "goal"
    elif prev is not None and np.allclose(got, prev):
        branch = "same-as-prev (hysteresis or repeated walk)"
    if branch == "?":
        # distinguish walk-visible / crawl / center-return by checking membership
        if cell_ok and np.allclose(got, env.node_pos[n]):
            branch = "center-return(self cell)"
        else:
            # crawl and walk both return node centers; check visibility from pos
            branch = "walk-or-crawl(node center)"
    vis = R.is_edge_free(pos, got, env.tree,
                         radius=min(env._clr(pos), env._clr(got)))
    return got, branch, bool(vis), cell_ok, (int(n) if n >= 0 else -1)


for seed in SEEDS:
    env = DroneGeoEnv(**KW)
    obs, _ = env.reset(seed=seed)
    print(f"\n===== seed {seed} =====")
    # policy to stall
    best_phi = env._phi(env.pos); since = 0; term = trunc = False; step = 0
    while not (term or trunc) and since < 50 and step < 650:
        act, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(act)
        step += 1
        phi = env._phi(env.pos)
        if phi < best_phi - 0.05:
            best_phi = phi; since = 0
        else:
            since += 1
    if term or trunc:
        print(f"  no stall (success={info['is_success']})"); continue
    # oracle until frozen (pos unchanged 15 steps)
    frozen = 0; last = env.pos.copy()
    for t in range(300):
        cgt = env._carrot()
        v = cgt - env.pos; nn = float(np.linalg.norm(v))
        a = v / nn if nn > 1e-6 else np.zeros(3)
        obs, r, term, trunc, info = env.step(a)
        if term or trunc:
            break
        if float(np.linalg.norm(env.pos - last)) < 1e-6:
            frozen += 1
        else:
            frozen = 0; last = env.pos.copy()
        if frozen >= 15:
            break
    if term or trunc:
        print(f"  oracle finished (success={info['is_success']})"); continue

    p = env.pos
    print(f"  FROZEN at ({p[0]:.2f},{p[1]:.2f},{p[2]:.2f}) phi={env._phi(p):.2f} "
          f"bumps={info['bumps']}")
    got, branch, vis, cell_ok, nid = carrot_verbose(env)
    d = float(np.linalg.norm(got - p))
    print(f"  carrot=({got[0]:.2f},{got[1]:.2f},{got[2]:.2f}) dist={d:.2f} "
          f"branch={branch} visible_from_pos={vis} cell_ok={cell_ok}")
    if env._carrot_prev is not None:
        cp = env._carrot_prev
        cpv = R.is_edge_free(p, cp, env.tree,
                             radius=min(env._clr(p), env._clr(cp)))
        print(f"  carrot_prev=({cp[0]:.2f},{cp[1]:.2f},{cp[2]:.2f}) "
              f"phi={env._phi(cp):.2f} visible={bool(cpv)} "
              f"(phi(pos)={env._phi(p):.2f})")
    # cell center reachability
    if nid >= 0:
        cc = env.node_pos[nid]
        ccv = R.is_edge_free(p, cc, env.tree,
                             radius=min(env._clr(p), env._clr(cc)))
        print(f"  own cell center=({cc[0]:.2f},{cc[1]:.2f},{cc[2]:.2f}) "
              f"dist={float(np.linalg.norm(cc-p)):.2f} reachable={bool(ccv)}")
    # single-axis probes (0.3m, the max step)
    names = ["+x", "-x", "+y", "-y", "+z", "-z"]
    print("  single-axis 0.3m moves:", end=" ")
    for k in range(3):
        for s in (+1, -1):
            q = p.copy(); q[k] += s * 0.3
            ok = (not (np.any(q < env.bounds_lo) or np.any(q > env.bounds_hi))) and \
                R.is_edge_free(p, q, env.tree,
                               radius=min(env._clr(p), env._clr(q)))
            print(f"{names[2*k + (0 if s>0 else 1)]}={'O' if ok else 'X'}", end=" ")
    print()
    # what direction is the oracle pushing?
    v = got - p; nn = float(np.linalg.norm(v))
    if nn > 1e-6:
        a = v / nn
        print(f"  oracle push dir=({a[0]:.2f},{a[1]:.2f},{a[2]:.2f})")
