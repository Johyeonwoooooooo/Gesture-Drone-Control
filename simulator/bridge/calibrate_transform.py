"""
Calibrate the detection-world -> Unity-world transform for one building.

Scores every sign-flip candidate (see coord_transform.candidate_transforms) by
projecting the building's point cloud (data/final_npy/<b>_*/coord.npy) into the
Unity-exported 3D voxel map (ground truth of Unity-world occupancy) and
measuring hit_rate = fraction of transformed points landing in occupied voxels.
The export dilates obstacles by the drone radius, so hit_rate (not IoU) is the
discriminating metric; a wrong mirror scatters surface points into free space.

Run on the GPU server (needs data/final_npy + a voxel map export, no Unity):
  python simulator/bridge/calibrate_transform.py \
      --building 00809_Qpor2mEya8F \
      --voxel-map simulator/bridge/Qpor2mEya8F_voxel_map_3d.json

The voxel map comes from Unity: Tools > Export Voxel Map 3D (ExportVoxelMap3D.cs).
For a new scene placed differently in Unity, pass --scale/--translation with
the values used when placing the glb (see README.md §씬 준비).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent))

from coord_transform import (  # noqa: E402
    DEFAULT_SCALE,
    DEFAULT_TRANSLATION,
    SimTransform,
    candidate_transforms,
    transform_path_for_building,
)

_REPO = _THIS.parents[2]
DEFAULT_COORDS = _REPO / "data" / "final_npy"


def load_building_coords(coords_dir: Path, building: str, max_points: int,
                         seed: int) -> np.ndarray:
    """Point cloud of one building from the LitePT layout: <building>_*/coord.npy."""
    coords = []
    for region in sorted(coords_dir.glob(f"{building}_*")):
        f = region / "coord.npy"
        if f.exists():
            coords.append(np.load(f))
    if not coords:
        raise FileNotFoundError(f"no {building}_*/coord.npy under {coords_dir}")
    pts = np.concatenate(coords).astype(np.float64)
    if len(pts) > max_points:
        rng = np.random.default_rng(seed)
        pts = pts[rng.choice(len(pts), max_points, replace=False)]
    return pts


class VoxelMap:
    def __init__(self, path: Path):
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        self.voxel_size = float(data["voxel_size"])
        o = data["origin"]
        self.origin = np.array([o["x"], o["y"], o["z"]], dtype=np.float64)
        s = data["size"]
        self.size = np.array([s["width"], s["height"], s["depth"]], dtype=np.int64)
        self.grid = np.zeros(self.size, dtype=bool)
        occ = np.array(data["occupied"], dtype=np.int64)
        self.grid[occ[:, 0], occ[:, 1], occ[:, 2]] = True

    def hit_rate(self, pts_unity: np.ndarray) -> tuple[float, float]:
        """(hit_rate, in_bounds_rate) of points vs occupied voxels.

        The dilated export occupies ~43% of the padded box, so hit_rate alone
        saturates for any roughly-aligned candidate; scoring must combine it
        with in_bounds_rate (see score())."""
        cells = np.floor((pts_unity - self.origin) / self.voxel_size).astype(np.int64)
        in_bounds = np.all((cells >= 0) & (cells < self.size), axis=1)
        if not in_bounds.any():
            return 0.0, 0.0
        c = cells[in_bounds]
        hits = self.grid[c[:, 0], c[:, 1], c[:, 2]]
        return float(hits.mean()), float(in_bounds.mean())

    def score(self, pts_unity: np.ndarray) -> float:
        """hit_rate x in_bounds_rate — penalizes candidates that shove most of
        the cloud outside the map (their few in-bounds points hit by chance)."""
        hit, ib = self.hit_rate(pts_unity)
        return hit * ib


def bbox_center_translation(matrix: np.ndarray, pts: np.ndarray, vm: VoxelMap) -> np.ndarray:
    """Translation aligning the point cloud's bbox center with the voxel map's."""
    rotated = pts @ matrix.T
    vm_center = vm.origin + vm.size * vm.voxel_size / 2.0
    r_center = (rotated.min(0) + rotated.max(0)) / 2.0
    return vm_center - r_center


def refine_translation(
    matrix: np.ndarray, t0: np.ndarray, pts: np.ndarray, vm: VoxelMap
) -> tuple[np.ndarray, float]:
    rotated = pts @ matrix.T
    best_t, best_score = t0, -1.0
    offsets = np.arange(-2.0, 2.01, 0.5)
    dys = np.arange(-1.0, 1.01, 0.5)
    for dx in offsets:
        for dy in dys:
            for dz in offsets:
                t = t0 + np.array([dx, dy, dz])
                score = vm.score(rotated + t)
                if score > best_score:
                    best_score, best_t = score, t
    return best_t, best_score


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--coords-dir", type=Path, default=DEFAULT_COORDS,
                    help="LitePT data dir with <building>_*/coord.npy.")
    ap.add_argument("--building", default="00809_Qpor2mEya8F")
    ap.add_argument("--voxel-map", type=Path,
                    default=_THIS.parent / "Qpor2mEya8F_voxel_map_3d.json")
    ap.add_argument("--out", type=Path, default=None,
                    help="default: simulator/bridge/transforms/<building>.json")
    ap.add_argument("--scale", type=float, default=DEFAULT_SCALE)
    ap.add_argument("--translation", type=float, nargs=3, default=list(DEFAULT_TRANSLATION),
                    metavar=("X", "Y", "Z"))
    ap.add_argument("--max-points", type=int, default=300_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--margin", type=float, default=1.2,
                    help="required score ratio best/runner-up")
    args = ap.parse_args()

    pts = load_building_coords(args.coords_dir, args.building, args.max_points,
                               args.seed)
    vm = VoxelMap(args.voxel_map)
    print(f"[calib] {len(pts)} points, voxel map {tuple(vm.size)} @ {vm.voxel_size} u")

    rows = []
    for name, cand in candidate_transforms(args.scale, tuple(args.translation)).items():
        best = None
        for t_init_name, t_init in (
            ("nominal", cand.translation),
            ("bbox", bbox_center_translation(cand.matrix, pts, vm)),
        ):
            t, score = refine_translation(cand.matrix, t_init, pts, vm)
            if best is None or score > best[1]:
                best = (t, score, t_init_name)
        t, score, t_init_name = best
        hit, ib = vm.hit_rate(pts @ cand.matrix.T + t)
        rows.append((name, score, hit, ib, t, t_init_name, cand.matrix))

    rows.sort(key=lambda r: -r[1])
    print(f"\n{'candidate':<10} {'score':>8} {'hit_rate':>8} {'in_bounds':>9}  translation (init)")
    for name, score, hit, ib, t, init, _ in rows:
        print(f"{name:<10} {score:>8.4f} {hit:>8.4f} {ib:>9.4f}  "
              f"({t[0]:+.2f},{t[1]:+.2f},{t[2]:+.2f}) ({init})")

    best_name, best_score, best_hit, best_ib, best_t, best_init, best_m = rows[0]
    runner_score = rows[1][1]
    if runner_score > 0 and best_score / max(runner_score, 1e-9) < args.margin:
        print(f"\n[calib] FAIL: best score ({best_score:.4f}) < {args.margin}x runner-up "
              f"({runner_score:.4f}) — ambiguous, check scale/translation/scene.")
        return 1

    out = args.out or transform_path_for_building(args.building)
    tf = SimTransform(
        matrix=best_m,
        translation=best_t,
        meta={
            "building": args.building,
            "candidate": best_name,
            "scale": args.scale,
            "score": round(best_score, 4),
            "hit_rate": round(best_hit, 4),
            "in_bounds_rate": round(best_ib, 4),
            "runner_up_score": round(runner_score, 4),
            "translation_init": best_init,
            "voxel_map": args.voxel_map.name,
            "n_points": int(len(pts)),
            "seed": args.seed,
        },
    )
    tf.save(out)
    print(f"\n[calib] wrote {out} (candidate {best_name}, score {best_score:.4f}, "
          f"hit_rate {best_hit:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
