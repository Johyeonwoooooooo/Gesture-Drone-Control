# -*- coding: utf-8 -*-
"""Per-room component census: for each room bbox, which graph component owns
its cells? Reveals whether comp0 is real rooms (bad cut) or void space.
(ASCII output)"""
import numpy as np
from scipy.sparse.csgraph import connected_components
from geo_env import DroneGeoEnv
from hierarchical_plan import load_graph

KW = dict(curriculum=False, priv_obs=True, ray_max=4.0, ray_layout="horiz14",
          subgoal_dist=2.5, bump_penalty=2.0, clearance=0.12, max_steps=700)
env = DroneGeoEnv(**KW)
ncomp, labels = connected_components(env.csr, directed=False)
sizes = np.bincount(labels)
order = np.argsort(-sizes)
big = order[:2]
pos = env.node_pos

g = load_graph()
print(f"{'room':22s} {'cells':>6s} {'comp'+str(big[0]):>7s} {'comp'+str(big[1]):>7s} {'other':>6s}")
for rid, r in sorted(g['rooms'].items()):
    blo = np.array(r['bbox_min'], float); bhi = np.array(r['bbox_max'], float)
    m = np.all((pos >= blo - 0.1) & (pos <= bhi + 0.1), axis=1)
    n = int(m.sum())
    if n == 0:
        continue
    c0 = int(np.sum(labels[m] == big[0]))
    c1 = int(np.sum(labels[m] == big[1]))
    flag = ""
    if c1 > c0:
        flag = "  <-- majority comp" + str(big[1])
    print(f"{rid:22s} {n:6d} {c0:7d} {c1:7d} {n-c0-c1:6d}{flag}")
