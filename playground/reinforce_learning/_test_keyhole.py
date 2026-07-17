# -*- coding: utf-8 -*-
"""Is the (4.2,3.1,3.8) passage a real stair or a phantom shortcut?
1) print stair chain endpoints, 2) distance from the keyhole to chains,
3) walk the greedy geodesic descent from the seed-7001 stall through the hole,
4) compare geodesic length via keyhole vs with vertical-off-stair edges banned.
(ASCII output)"""
import numpy as np
from geo_env import DroneGeoEnv
from scipy.sparse.csgraph import dijkstra

KW = dict(curriculum=False, priv_obs=True, ray_max=4.0, ray_layout="horiz14",
          subgoal_dist=2.5, bump_penalty=2.0, clearance=0.12, max_steps=700)
env = DroneGeoEnv(**KW)

print("[chains]")
for i, ch in enumerate(env._chains):
    a, b = ch[0], ch[-1]
    print(f"  chain{i}: ({a[0]:.1f},{a[1]:.1f},{a[2]:.1f}) -> "
          f"({b[0]:.1f},{b[1]:.1f},{b[2]:.1f})  ({len(ch)} pts)")

hole = np.array([4.2, 3.12, 3.81])
dmin = min(float(np.linalg.norm(ch - hole, axis=1).min()) for ch in env._chains)
print(f"[keyhole] min distance to any stair chain = {dmin:.2f} m")
print(f"[keyhole] _clr(hole) = {env._clr(hole):.3f}")

# seed 7001 task: reproduce goal, walk greedy descent from the stall cell
obs, _ = env.reset(seed=7001)
start, goal = env.pos.copy(), env.goal.copy()
print(f"[7001] start=({start[0]:.1f},{start[1]:.1f},{start[2]:.1f}) "
      f"goal=({goal[0]:.1f},{goal[1]:.1f},{goal[2]:.1f}) phi0={env._phi(start):.1f}")

p = np.array([3.6, 3.9, 4.3])
c = np.clip(np.round((p - env.bounds_lo) / env.grid_v).astype(int), 0, env.grid_dims - 1)
n = env.nid[c[0], c[1], c[2]]
print("[greedy descent from stall cell]")
for stp in range(40):
    f = env._field
    best, bf = n, f[n]
    ci = np.round((env.node_pos[n] - env.bounds_lo) / env.grid_v).astype(int)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if not (dx or dy or dz):
                    continue
                x, y, z = ci[0]+dx, ci[1]+dy, ci[2]+dz
                if not (0 <= x < env.grid_dims[0] and 0 <= y < env.grid_dims[1]
                        and 0 <= z < env.grid_dims[2]):
                    continue
                m = env.nid[x, y, z]
                if m >= 0 and f[m] < bf:
                    best, bf = int(m), float(f[m])
    if best == n:
        break
    n = best
    q = env.node_pos[n]
    print(f"  ({q[0]:.2f},{q[1]:.2f},{q[2]:.2f}) phi={bf:.2f}")
    if bf <= env.grid_v:
        break

# geodesic length if vertical edges away from stair chains are banned
pos = env.node_pos
rows, cols = env.csr.nonzero()
wts = np.asarray(env.csr[rows, cols]).ravel()
dz = np.abs(pos[rows][:, 2] - pos[cols][:, 2])
mid = (pos[rows] + pos[cols]) / 2
from scipy.spatial import cKDTree
chain_pts = np.vstack(env._chains)
dd, _ = cKDTree(chain_pts).query(mid, workers=-1)
near_stair = dd < 1.2
keep = (dz < 1e-6) | near_stair
from scipy import sparse
csr2 = sparse.csr_matrix((wts[keep], (rows[keep], cols[keep])), shape=env.csr.shape)

gi = int(np.argmin(np.linalg.norm(pos - goal, axis=1)))
si = int(np.argmin(np.linalg.norm(pos - start, axis=1)))
f1 = dijkstra(env.csr, directed=False, indices=gi)
f2 = dijkstra(csr2, directed=False, indices=gi)
print(f"[7001 geodesic] via anything: {f1[si]:.1f} m | stairs-only vertical: {f2[si]:.1f} m")
