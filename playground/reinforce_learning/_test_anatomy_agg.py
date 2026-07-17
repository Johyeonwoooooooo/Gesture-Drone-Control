# -*- coding: utf-8 -*-
"""Aggregate anatomy_results.json: attribute failures to the keyhole region
and other clusters. (ASCII output)"""
import json
import os
import numpy as np

_BASE = os.path.dirname(os.path.abspath(__file__))
res = json.load(open(os.path.join(_BASE, "anatomy_results.json")))

KEYHOLE = np.array([4.0, 3.6, 4.1])   # stall shelf above the shaft (3-4.5, 3-4.3, z~3.8-4.4)

for name, eps in res.items():
    ng = [e for e in eps if not e["success"]]
    print(f"\n===== {name}: fails {len(ng)}/{len(eps)} =====")
    near_key = [e for e in ng if np.linalg.norm(np.array(e["end"]) - KEYHOLE) < 1.6]
    print(f"  ends within 1.6m of keyhole shelf: {len(near_key)} "
          f"({len(near_key)/len(ng):.0%} of fails)")
    top_stuck = [e for e in ng if e["end"][2] > 3.4 and e["goal_z"] < 3.0]
    print(f"  stuck on top floor with goal below: {len(top_stuck)} "
          f"({len(top_stuck)/len(ng):.0%})")
    # of those, how many are the keyhole shelf vs elsewhere on top floor
    ts_key = [e for e in top_stuck if np.linalg.norm(np.array(e["end"]) - KEYHOLE) < 1.6]
    print(f"    of top-stuck, at keyhole shelf: {len(ts_key)}")
    # what if all keyhole-shelf failures became successes?
    fixed = sum(e["success"] for e in eps) + len(near_key)
    print(f"  success if keyhole fixed: {fixed}/200 = {fixed/200:.3f}")
    # remaining big clusters excluding keyhole
    rest = [e for e in ng if np.linalg.norm(np.array(e["end"]) - KEYHOLE) >= 1.6]
    clusters = {}
    for e in rest:
        k = tuple(int(round(v)) for v in e["end"])
        clusters[k] = clusters.get(k, 0) + 1
    top = [kv for kv in sorted(clusters.items(), key=lambda kv: -kv[1]) if kv[1] >= 2]
    print("  non-keyhole clusters:", top[:8])
    # cross-floor failures not at keyhole: mode split
    for lbl, grp in (("keyhole", near_key), ("other", rest)):
        m = {}
        for e in grp:
            m[e["mode"]] = m.get(e["mode"], 0) + 1
        print(f"  modes[{lbl}]: {m}")
