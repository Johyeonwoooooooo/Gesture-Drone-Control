"""Convert an HM3D scene .glb to coord/color/normal/segment .npy files.

Bakes UV-textures to per-vertex colors (per sub-mesh, then concatenates).
The output layout matches Pointcept's matterport3d_compressed format so the
existing run_inference.py works unchanged.

Usage:
    python scripts/prepare_hm3d.py \
        --glb 3D-segmentation/cache/hm3d_example/00337-CFVBbU9Rsyb/CFVBbU9Rsyb.glb \
        --scene-id CFVBbU9Rsyb \
        --out-dir 3D-segmentation/cache/hm3d_compressed
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh


def bake_vertex_colors(scene: trimesh.Scene) -> trimesh.Trimesh:
    """Concatenate scene with per-vertex colors baked from each submesh's texture.

    For each submesh: try .visual.to_color() (samples texture at vertex UVs).
    Fallback to gray for submeshes that can't bake.
    """
    parts: list[trimesh.Trimesh] = []
    n_failed = 0
    for k, g in scene.geometry.items():
        if not isinstance(g, trimesh.Trimesh):
            continue
        # Color per vertex
        try:
            cv = g.visual.to_color()
            vc = np.asarray(cv.vertex_colors, dtype=np.uint8)
            if vc.shape[0] != len(g.vertices):
                raise ValueError(f"shape mismatch {vc.shape} vs V={len(g.vertices)}")
        except Exception:
            n_failed += 1
            vc = np.tile([180, 180, 180, 255], (len(g.vertices), 1)).astype(np.uint8)
        gc = g.copy()
        gc.visual = trimesh.visual.ColorVisuals(mesh=gc, vertex_colors=vc)
        parts.append(gc)
    if not parts:
        raise SystemExit("No Trimesh geometry found in glb")
    print(f"  baked {len(parts)} submeshes (failed→gray: {n_failed})")
    cat = trimesh.util.concatenate(parts)
    return cat


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glb", required=True)
    ap.add_argument(
        "--scene-id", required=True, help="Scene name to use as folder (e.g. CFVBbU9Rsyb)"
    )
    ap.add_argument("--out-dir", required=True)
    ap.add_argument(
        "--up-axis-fix",
        action="store_true",
        default=True,
        help="HM3D glb is Y-up; rotate to Z-up to match Pointcept/Mosaic3D conventions.",
    )
    args = ap.parse_args()

    out_root = Path(args.out_dir)
    scene_dir = out_root / args.scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)
    mesh_dir = out_root / f"{args.scene_id}_mesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    print(f"[prepare] loading {args.glb}")
    raw = trimesh.load(args.glb, process=False)
    if isinstance(raw, trimesh.Scene):
        mesh = bake_vertex_colors(raw)
    else:
        mesh = raw

    if args.up_axis_fix:
        # GLB convention is Y-up. Rotate -90deg around X to put +Z up (matching MP3D).
        R = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
        mesh.apply_transform(R)

    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)

    if hasattr(mesh.visual, "vertex_colors") and mesh.visual.vertex_colors is not None:
        rgba = np.asarray(mesh.visual.vertex_colors, dtype=np.uint8)
        rgb = rgba[:, :3]
    else:
        rgb = np.full((len(verts), 3), 180, dtype=np.uint8)

    # Per-vertex normals (trimesh recomputes from faces)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float32)

    # No GT segmentation for HM3D example -> store -1
    seg = np.full(len(verts), -1, dtype=np.int16)

    np.save(scene_dir / "coord.npy", verts)
    np.save(scene_dir / "color.npy", rgb)
    np.save(scene_dir / "normal.npy", normals)
    np.save(scene_dir / "segment.npy", seg)

    np.save(mesh_dir / "vertices.npy", verts)
    np.save(mesh_dir / "faces.npy", faces)
    np.save(mesh_dir / "vertex_colors.npy", rgb)

    bb_min = verts.min(0).tolist()
    bb_max = verts.max(0).tolist()
    meta = {
        "scene_id": args.scene_id,
        "glb": str(args.glb),
        "n_vertices": int(len(verts)),
        "n_faces": int(len(faces)),
        "bbox_min": bb_min,
        "bbox_max": bb_max,
        "extents": [bb_max[i] - bb_min[i] for i in range(3)],
        "up_axis_fixed": bool(args.up_axis_fix),
    }
    (out_root / f"{args.scene_id}.json").write_text(json.dumps(meta, indent=2))

    print(
        f"[prepare] {args.scene_id}: V={len(verts)} F={len(faces)} "
        f"extents={meta['extents']}"
    )
    print(f"[prepare] wrote -> {scene_dir} and {mesh_dir}")


if __name__ == "__main__":
    main()
