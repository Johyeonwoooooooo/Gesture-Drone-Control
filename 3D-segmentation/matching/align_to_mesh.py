"""Match compressed Matterport3D point cloud to the original region mesh.

For each compressed region (`<house>_NN/coord.npy`), find for every original mesh
vertex the nearest compressed point — yielding a per-vertex index that lets us
project per-point features onto the original mesh for visualization.

Output per region:
    cache/match/<house>_NN/
        vertex2point.npy   # (V,) int32 — index into compressed coord.npy
        vertex2dist.npy    # (V,) float32 — distance, for sanity / threshold
        meta.json          # source paths, vertex count, dist stats

Usage:
    python align_to_mesh.py \
        --compressed-dir .../matterport3d_compressed \
        --orig-house-dir .../v1/scans/17DRP5sb8fy/region_segmentations \
        --out-dir .../3D-segmentation/cache/match \
        --house 17DRP5sb8fy
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree


def load_region_mesh(ply_path: Path) -> trimesh.Trimesh:
    """Trimesh load returning a Trimesh (force scene→geometry merge)."""
    m = trimesh.load(ply_path, process=False)
    if isinstance(m, trimesh.Scene):
        # merge all geometries
        m = trimesh.util.concatenate(tuple(m.geometry.values()))
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--compressed-dir",
        required=True,
        help="Path to matterport3d_compressed (contains <house>_NN folders)",
    )
    ap.add_argument(
        "--orig-house-dir",
        required=True,
        help=(
            "Path to <scans>/<house>/region_segmentations dir, "
            "containing regionN.ply files."
        ),
    )
    ap.add_argument("--out-dir", required=True)
    ap.add_argument(
        "--house",
        required=True,
        help="House id, e.g. 17DRP5sb8fy. Matches <house>_NN dirs.",
    )
    ap.add_argument(
        "--max-dist",
        type=float,
        default=0.05,
        help="Warn if median NN distance > this (meters).",
    )
    args = ap.parse_args()

    comp_dir = Path(args.compressed_dir)
    orig_dir = Path(args.orig_house_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # find all <house>_NN
    pat = re.compile(rf"^{re.escape(args.house)}_(\d+)$")
    regions = []
    for p in sorted(comp_dir.iterdir()):
        m = pat.match(p.name)
        if m:
            regions.append((p, int(m.group(1))))
    if not regions:
        raise SystemExit(f"No regions matching {args.house}_NN under {comp_dir}")

    print(f"[match] {len(regions)} regions for house {args.house}")
    summary = []

    for region_path, region_idx in regions:
        ply_path = orig_dir / f"region{region_idx}.ply"
        if not ply_path.exists():
            print(f"[match] SKIP {region_path.name}: missing {ply_path}")
            continue

        coord_comp = np.load(region_path / "coord.npy").astype(np.float32)
        mesh = load_region_mesh(ply_path)
        verts = np.asarray(mesh.vertices, dtype=np.float32)

        # Build KD tree on compressed points, query each mesh vertex
        tree = cKDTree(coord_comp)
        dists, idx = tree.query(verts, k=1)
        idx = idx.astype(np.int32)
        dists = dists.astype(np.float32)

        sub = out_dir / region_path.name
        sub.mkdir(parents=True, exist_ok=True)
        np.save(sub / "vertex2point.npy", idx)
        np.save(sub / "vertex2dist.npy", dists)
        np.save(sub / "vertices.npy", verts)
        if hasattr(mesh, "faces") and mesh.faces is not None and len(mesh.faces) > 0:
            np.save(sub / "faces.npy", np.asarray(mesh.faces, dtype=np.int32))
        if (
            hasattr(mesh.visual, "vertex_colors")
            and mesh.visual.vertex_colors is not None
            and len(mesh.visual.vertex_colors) == len(verts)
        ):
            vc = np.asarray(mesh.visual.vertex_colors, dtype=np.uint8)[:, :3]
            np.save(sub / "vertex_colors.npy", vc)

        med = float(np.median(dists))
        mx = float(dists.max())
        ok = med <= args.max_dist
        marker = "OK" if ok else "WARN"
        print(
            f"[match][{marker}] {region_path.name}: V={len(verts)} P={len(coord_comp)} "
            f"med-dist={med:.4f}m max={mx:.4f}m"
        )

        summary.append(
            {
                "region": region_path.name,
                "ply": str(ply_path),
                "n_vertices": int(len(verts)),
                "n_points": int(len(coord_comp)),
                "median_dist": med,
                "max_dist": mx,
                "ok": ok,
            }
        )

    (out_dir / f"{args.house}_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[match] wrote summary -> {out_dir / f'{args.house}_summary.json'}")


if __name__ == "__main__":
    main()
