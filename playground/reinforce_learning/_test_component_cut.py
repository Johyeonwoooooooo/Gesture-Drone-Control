# -*- coding: utf-8 -*-
"""Why did the guide graph main component drop to 56%? Component census by
z-band + locate the cut seams (adjacent node pairs in different components).
(ASCII output)"""
import numpy as np
from scipy.sparse.csgraph import connected_components
from geo_env import DroneGeoEnv

KW = dict(curriculum=False, priv_obs=True, ray_max=4.0, ray_layout="horiz14",
          subgoal_dist=2.5, bump_penalty=2.0, clearance=0.12, max_steps=700)
env = DroneGeoEnv(**KW)

ncomp, labels = connected_components(env.csr, directed=False)
sizes = np.bincount(labels)
order = np.argsort(-sizes)
pos = env.node_pos
print(f"[components] total {ncomp}, top 6:")
for k in order[:6]:
    m = labels == k
    zmin, zmax = pos[m][:, 2].min(), pos[m][:, 2].max()
    print(f"  comp{k}: {sizes[k]:6d} cells  z {zmin:5.2f}..{zmax:5.2f}  "
          f"x {pos[m][:,0].min():5.1f}..{pos[m][:,0].max():5.1f}  "
          f"y {pos[m][:,1].min():5.1f}..{pos[m][:,1].max():5.1f}")

main = order[0]
second = order[1]
# find geometric seams between main and 2nd/3rd components: node pairs within
# 0.55m in different components
from scipy.spatial import cKDTree
for other in order[1:4]:
    A = pos[labels == main]
    B = pos[labels == other]
    tb = cKDTree(B)
    dd, jj = tb.query(A, distance_upper_bound=0.55)
    hit = np.isfinite(dd)
    seam = A[hit]
    if len(seam) == 0:
        print(f"[seam main<->comp{other}] none within 0.55m")
        continue
    # cluster seam points to 1m grid
    keys = {}
    for p in seam:
        k = tuple(np.round(p).astype(int))
        keys[k] = keys.get(k, 0) + 1
    top = sorted(keys.items(), key=lambda kv: -kv[1])[:8]
    print(f"[seam main<->comp{other}] {len(seam)} boundary cells, clusters: {top}")
    # clearance stats at a few seam points
    for k, c in top[:3]:
        q = np.array(k, float)
        d, _ = env.tree.query(q)
        ds = float(env.stair_tree.query(q)[0])
        print(f"    at {k}: obstacle clearance {float(d):.3f} m, "
              f"dist to stair chain {ds:.2f} m")

# do the three floors exist in main?
m = labels == main
for zlo, zhi, name in ((-3.2, -0.5, "bottom"), (-0.5, 2.5, "mid"), (2.5, 6.0, "top")):
    n_all = int(np.sum((pos[:, 2] >= zlo) & (pos[:, 2] < zhi)))
    n_main = int(np.sum((pos[m][:, 2] >= zlo) & (pos[m][:, 2] < zhi)))
    print(f"[floor {name:6s}] main {n_main}/{n_all}")
