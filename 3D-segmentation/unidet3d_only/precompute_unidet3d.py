"""Offline UniDet3D detection precompute for the webapp cache.

Mirrors the miny-det batch pipeline: load the model ONCE, run UniDet3D over every
region of the requested buildings, and cache each region's detections to
``cache/<building>/unidet3d/<region>.pkl`` in RAW WORLD frame. The webapp's
UniDet3D backend then just LOADS these (no serve-time GPU detection → no OOM).

Run from the `unidet3d` conda env:

    python 3D-segmentation/unidet3d_only/precompute_unidet3d.py \
        --buildings 00800_TEEsavR23oF 00809_Qpor2mEya8F --unidet3d-device cuda:0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Put 3D-segmentation/ on sys.path so the `webapp` + `unidet3d_only` namespace
# packages import the same way the servers set them up.
_SEG_ROOT = Path(__file__).resolve().parents[1]
if str(_SEG_ROOT) not in sys.path:
    sys.path.insert(0, str(_SEG_ROOT))

from webapp import server as webapp_server          # noqa: E402
from webapp import unidet3d_backend as u3d           # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--buildings", nargs="+",
        default=["00800_TEEsavR23oF", "00809_Qpor2mEya8F"],
        help="Building IDs to precompute (default: the two HM3D demo houses).",
    )
    ap.add_argument(
        "--cache-dir", default=str(_SEG_ROOT / "cache"),
        help="Webapp cache dir (contains <building>/feat/<region>/...).",
    )
    ap.add_argument("--score-thr", type=float, default=0.30)
    ap.add_argument("--max-points", type=int, default=u3d.SAFE_MAX_POINTS)
    ap.add_argument("--voxel-size", type=float, default=u3d.SAFE_VOXEL_SIZE)
    ap.add_argument("--force", action="store_true",
                    help="Recompute even if a region's pkl already exists.")
    # UniDet3D model flags (reuse the backend's argument definitions/defaults).
    # This also registers --enable-unidet3d; we force it on below.
    u3d.add_unidet3d_args(ap)
    args = ap.parse_args()
    args.enable_unidet3d = True  # always on for this script
    return args


def main():
    args = parse_args()
    cache_dir = Path(args.cache_dir)

    detector = u3d.make_detector(args)
    if detector is None:
        raise SystemExit(
            "UniDet3D detector failed to load — run this from the `unidet3d` "
            "env and check --unidet3d-ckpt / --unidet3d-cfg."
        )

    device = torch.device("cpu")  # only used by load_region for feat tensors
    total = 0
    for building in args.buildings:
        regions = webapp_server.regions_for_building(building, cache_dir)
        if not regions:
            print(f"[precompute] {building}: no regions under {cache_dir} — skip")
            continue
        print(f"[precompute] {building}: {len(regions)} regions")
        for i, region in enumerate(regions):
            out = u3d.unidet3d_cache_path(region, cache_dir)
            if out.exists() and not args.force:
                print(f"  ({i+1}/{len(regions)}) {region}: exists, skip")
                continue
            asset = webapp_server.load_region(region, cache_dir, device)
            pts = u3d.scene_points6(asset)
            det = detector.detect(
                pts, score_thr=args.score_thr,
                max_points=args.max_points, voxel_size=args.voxel_size)
            # display (centered) frame → RAW WORLD frame for a frame-independent cache
            if len(det.bboxes):
                det.bboxes[:, :3] += asset.center
            u3d.save_detection_world(out, det)
            total += 1
            print(f"  ({i+1}/{len(regions)}) {region}: "
                  f"{len(pts)} pts → {len(det.bboxes)} boxes → {out}")

    print(f"[precompute] done — wrote {total} region pkls.")


if __name__ == "__main__":
    main()
