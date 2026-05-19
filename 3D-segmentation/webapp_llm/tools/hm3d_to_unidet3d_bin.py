"""Concatenate HM3D region npy files into a (N, 9) float32 .bin for UniDet3D.

Region layout (per dir): coord.npy (N,3 float32 meters), color.npy (N,3 uint8 0-255),
normal.npy (N,3 float32 unit vectors).

Output channels: x, y, z, r, g, b, nx, ny, nz  (float32). RGB normalized to [0,1].

Usage:
    python hm3d_to_unidet3d_bin.py \
        --scene-dir data/hm3d_compressed/val/00800_TEEsavR23oF \
        --out unidet3d/data/my_scene.bin

If --scene-dir doesn't exist as a single dir, the script also accepts a "stem"
that matches multiple region dirs (e.g. ".../val/00800_TEEsavR23oF") and globs
"<stem>*" to merge them.
"""
import argparse, glob, os
import numpy as np


def load_region(d: str) -> np.ndarray:
    coord = np.load(os.path.join(d, "coord.npy")).astype(np.float32)
    color = np.load(os.path.join(d, "color.npy")).astype(np.float32)
    normal = np.load(os.path.join(d, "normal.npy")).astype(np.float32)
    assert coord.shape == normal.shape and coord.shape[0] == color.shape[0], \
        f"shape mismatch in {d}: {coord.shape} {color.shape} {normal.shape}"
    if color.max() > 1.5:
        color = color / 255.0
    return np.concatenate([coord, color, normal], axis=1).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-dir", required=True,
                    help="Region dir OR stem; if stem, all '<stem>*' dirs merged.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if os.path.isdir(args.scene_dir) and os.path.exists(
        os.path.join(args.scene_dir, "coord.npy")
    ):
        regions = [args.scene_dir]
    else:
        regions = sorted(d for d in glob.glob(args.scene_dir + "*")
                         if os.path.isdir(d)
                         and os.path.exists(os.path.join(d, "coord.npy")))
    if not regions:
        raise SystemExit(f"No region dirs found for {args.scene_dir}")
    print(f"[hm3d->bin] merging {len(regions)} region(s):")
    for r in regions:
        print(f"  {r}")

    parts = [load_region(d) for d in regions]
    pts = np.concatenate(parts, axis=0)
    print(f"[hm3d->bin] total points: {pts.shape}, dtype={pts.dtype}")
    print(f"  xyz range: {pts[:,:3].min(0)} .. {pts[:,:3].max(0)}")
    print(f"  rgb range: {pts[:,3:6].min():.3f} .. {pts[:,3:6].max():.3f}")
    print(f"  |n| mean: {np.linalg.norm(pts[:,6:9], axis=1).mean():.3f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    pts.tofile(args.out)
    print(f"[hm3d->bin] wrote {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
