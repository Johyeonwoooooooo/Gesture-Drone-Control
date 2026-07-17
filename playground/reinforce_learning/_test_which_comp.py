# -*- coding: utf-8 -*-
"""Which component is the real indoor network? Classify known-real points
(stair chain waypoints, previous successful goals) by component label.
(ASCII output)"""
import numpy as np
from scipy.sparse.csgraph import connected_components
from geo_env import DroneGeoEnv

KW = dict(curriculum=False, priv_obs=True, ray_max=4.0, ray_layout="horiz14",
          subgoal_dist=2.5, bump_penalty=2.0, clearance=0.12, max_steps=700)
env = DroneGeoEnv(**KW)
ncomp, labels = connected_components(env.csr, directed=False)
sizes = np.bincount(labels)
pos = env.node_pos


def comp_of(p):
    d = np.linalg.norm(pos - np.asarray(p, float), axis=1)
    i = int(np.argmin(d))
    return labels[i], float(d[i])


probes = []
for ci, ch in enumerate(env._chains):
    for tag, q in (("end0", ch[0]), ("mid", ch[len(ch)//2]), ("end1", ch[-1])):
        probes.append((f"chain{ci}.{tag}", q))
# known-good positions from successful episodes / demo flights
probes += [
    ("stair-eval goal A", np.array([7.0, 1.6, -1.5])),
    ("stair-eval goal B", np.array([7.6, 4.0, 4.1])),
    ("stair-eval goal C", np.array([5.0, 6.1, 1.2])),
    ("anatomy 7001 old start", np.array([9.6, 1.9, 4.4])),
    ("anatomy 7001 old goal", np.array([12.6, 4.6, 2.0])),
    ("keyhole shelf", np.array([4.0, 3.6, 4.1])),
    ("keyhole shaft", np.array([4.5, 3.1, 3.5])),
    ("freeze2 point", np.array([1.3, -0.4, 1.8])),
]
for name, q in probes:
    c, d = comp_of(q)
    print(f"  {name:24s} -> comp{c} (size {sizes[c]:6d})  nearest cell {d:.2f} m")

# per-floor split for the two big comps
order = np.argsort(-sizes)
for k in order[:2]:
    m = labels == k
    print(f"\ncomp{k} (size {sizes[k]}):")
    for zlo, zhi, nm in ((-3.2, -0.5, "bottom"), (-0.5, 2.5, "mid"), (2.5, 6.0, "top")):
        n = int(np.sum((pos[m][:, 2] >= zlo) & (pos[m][:, 2] < zhi)))
        print(f"    {nm:6s} {n}")
