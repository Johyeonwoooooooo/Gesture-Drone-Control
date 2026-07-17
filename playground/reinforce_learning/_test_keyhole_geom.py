# -*- coding: utf-8 -*-
"""Geometry of the keyhole shaft at (4.2-4.8, 3.1-3.7, z 2.9-4.1):
clearance profile along the descent cells + point-cloud density around the
shaft mouth (scan hole in the floor slab vs real void). (ASCII output)"""
import numpy as np
from geo_env import DroneGeoEnv

KW = dict(curriculum=False, priv_obs=True, ray_max=4.0, ray_layout="horiz14",
          subgoal_dist=2.5, bump_penalty=2.0, clearance=0.12, max_steps=700)
env = DroneGeoEnv(**KW)

cells = [(3.90, 3.72, 4.11), (3.90, 3.42, 3.81), (4.20, 3.12, 3.81),
         (4.50, 3.12, 3.51), (4.50, 3.42, 3.21), (4.80, 3.72, 2.91)]
print("[clearance along keyhole descent]")
for q in cells:
    d, _ = env.tree.query(np.asarray(q))
    print(f"  ({q[0]:.2f},{q[1]:.2f},{q[2]:.2f})  nearest obstacle {float(d):.3f} m")

# slab check: point density in horizontal ring around the shaft mouth at slab
# height. If the surrounding slab has points but the shaft column has none,
# it is a hole in the floor (real void or scan hole).
pts = env.coord
for zlo, zhi in ((3.2, 3.7), (3.7, 4.2)):
    m = (pts[:, 2] >= zlo) & (pts[:, 2] < zhi)
    sl = pts[m]
    print(f"\n[slab slice z {zlo}-{zhi}] {len(sl):,} points; occupancy map "
          f"x 3.0-6.0, y 2.0-5.0 (0.3m bins, count):")
    xs = np.arange(3.0, 6.01, 0.3)
    ys = np.arange(2.0, 5.01, 0.3)
    hdr = "      " + "".join(f"{x:5.1f}" for x in xs[:-1])
    print(hdr)
    for j in range(len(ys) - 1):
        row = []
        for i in range(len(xs) - 1):
            c = int(np.sum((sl[:, 0] >= xs[i]) & (sl[:, 0] < xs[i+1]) &
                           (sl[:, 1] >= ys[j]) & (sl[:, 1] < ys[j+1])))
            row.append(f"{min(c, 9999):5d}")
        print(f"y{ys[j]:4.1f}" + "".join(row))

# how much longer is the legit route? ban a small box around the shaft and
# re-run dijkstra for the 7001 task.
obs, _ = env.reset(seed=7001)
start, goal = env.pos.copy(), env.goal.copy()
pos = env.node_pos
inbox = (np.abs(pos[:, 0] - 4.35) < 1.0) & (np.abs(pos[:, 1] - 3.4) < 1.0) & \
        (pos[:, 2] > 2.8) & (pos[:, 2] < 4.3)
print(f"\n[ban-box] cells removed: {int(inbox.sum())}")
from scipy import sparse
from scipy.sparse.csgraph import dijkstra
rows, cols = env.csr.nonzero()
wts = np.asarray(env.csr[rows, cols]).ravel()
keep = ~(inbox[rows] | inbox[cols])
csr2 = sparse.csr_matrix((wts[keep], (rows[keep], cols[keep])), shape=env.csr.shape)
gi = int(np.argmin(np.linalg.norm(pos - goal, axis=1)))
si = int(np.argmin(np.linalg.norm(pos - start, axis=1)))
f1 = dijkstra(env.csr, directed=False, indices=gi)
f2 = dijkstra(csr2, directed=False, indices=gi)
print(f"[7001 geodesic] with keyhole: {f1[si]:.1f} m | keyhole banned: {f2[si]:.1f} m")
