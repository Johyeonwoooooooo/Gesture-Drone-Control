"""Per-point CLIP-aligned feature extraction with Mosaic3D SpUNet101.

Reads compressed Matterport3D regions (coord/color/normal/segment.npy) and
saves per-point feat (N, text_dim) as float16 .npy plus origin coords.

Usage:
    python run_inference.py \
        --ckpt /data1/workspaces/jgshin22/Gesture-Drone-Control/data/spunet101.ckpt \
        --data-dir /data1/workspaces/jgshin22/Gesture-Drone-Control/data/matterport3d_compressed \
        --out-dir /data1/workspaces/jgshin22/Gesture-Drone-Control/3D-segmentation/cache/feat \
        --regions 17DRP5sb8fy_00,17DRP5sb8fy_01
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
MOSAIC_SRC = REPO_ROOT / "Mosaic3D"
sys.path.insert(0, str(MOSAIC_SRC))

from src.models.networks.ppt.model import PPT  # noqa: E402
from src.models.networks.spunet.spconv_unet_v1m1_base import SpUNetBottleneck  # noqa: E402

# spunet101 architecture (from configs/model/spunet101+ppt.yaml)
SPUNET101_KW = dict(
    in_channels=3,
    out_channels=768,  # CLIP text_dim (ViT-L-16-HTxt-Recap-CLIP)
    base_channels=32,
    channels=[32, 64, 128, 256, 256, 128, 96, 96],
    layers=[2, 3, 4, 23, 2, 2, 2, 2],
)
PPT_CONDITIONS = ["ScanNet", "ARKitScenes", "ScanNetPP"]
PPT_CONTEXT_CHANNELS = 256
DEFAULT_GRID_SIZE = 0.02
DEFAULT_CONDITION = "ScanNet"  # closest indoor RGB-D to Matterport


def build_model() -> torch.nn.Module:
    """Build PPT(SpUNetBottleneck) matching the spunet101+ppt config."""
    backbone_factory = lambda: SpUNetBottleneck(**SPUNET101_KW)  # noqa: E731
    model = PPT(
        backbone=backbone_factory,
        conditions=PPT_CONDITIONS,
        context_channels=PPT_CONTEXT_CHANNELS,
    )
    return model


def load_checkpoint(model: torch.nn.Module, ckpt_path: str) -> None:
    """Load Lightning ckpt into our standalone PPT module.

    Lightning ckpt keys look like `net.backbone.<...>` and `net.embedding_table.weight`.
    We strip the leading `net.` prefix, and keep only `backbone.*` and `embedding_table.*`.
    """
    print(f"[ckpt] loading {ckpt_path}")
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = raw.get("state_dict", raw)

    new_sd = {}
    for k, v in sd.items():
        if k.startswith("net."):
            nk = k[len("net.") :]
        else:
            nk = k
        if nk.startswith("backbone.") or nk.startswith("embedding_table."):
            new_sd[nk] = v

    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    if missing:
        print(f"[ckpt] missing {len(missing)} keys (sample: {missing[:3]})")
    if unexpected:
        print(f"[ckpt] unexpected {len(unexpected)} keys (sample: {unexpected[:3]})")
    if len(missing) > 5:
        raise RuntimeError(f"Too many missing keys; ckpt may not match. Missing: {missing[:10]}")
    print(f"[ckpt] loaded ({len(new_sd)} tensors).")


@torch.no_grad()
def infer_region(
    model: torch.nn.Module,
    coord_np: np.ndarray,
    color_np: np.ndarray,
    *,
    device: torch.device,
    grid_size: float = DEFAULT_GRID_SIZE,
    condition: str = DEFAULT_CONDITION,
) -> np.ndarray:
    """Run the model on one region's point cloud. Returns (N, D) feat as fp16."""
    coord = coord_np.astype(np.float32, copy=True)
    color = color_np.astype(np.float32, copy=True)

    # CenterShift(apply_z=True): subtract centroid using min/max midpoints, mirroring the eval transform.
    # The actual CenterShift uses (x_min+x_max)/2 for x,y and z_min for z (when apply_z=True it shifts z by z_min as well).
    # See src/data/utils/transform.py:CenterShift. We use the center-of-bbox formula for parity.
    x_min, y_min, z_min = coord.min(axis=0)
    x_max, y_max, _ = coord.max(axis=0)
    coord[:, 0] -= (x_min + x_max) / 2
    coord[:, 1] -= (y_min + y_max) / 2
    coord[:, 2] -= z_min

    # NormalizeColor: color/127.5 - 1 (uint8 -> [-1, 1])
    color = color / 127.5 - 1.0

    coord_t = torch.from_numpy(coord).to(device)
    feat_t = torch.from_numpy(color).to(device)
    n = coord_t.shape[0]
    offset_t = torch.tensor([n], dtype=torch.long, device=device)
    batch_t = torch.zeros(n, dtype=torch.long, device=device)

    batch_dict = dict(
        coord=coord_t,
        feat=feat_t,
        offset=offset_t,
        batch=batch_t,
        grid_size=torch.tensor(grid_size, device=device),
        condition=[condition],
    )

    point = model(batch_dict)
    # Point.sparse_conv_feat.features is per-voxel; v2p_map maps voxel->point
    feat_per_point = point.sparse_conv_feat.features[point.v2p_map]
    return feat_per_point.detach().cpu().to(torch.float16).numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", required=True, help="matterport3d_compressed dir")
    ap.add_argument("--out-dir", required=True, help="cache output dir")
    ap.add_argument(
        "--regions",
        default=None,
        help="Comma-separated region names. If omitted, uses --house to auto-discover.",
    )
    ap.add_argument(
        "--house",
        default=None,
        help="If set and --regions omitted, runs all <house>_NN regions found.",
    )
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--grid-size", type=float, default=DEFAULT_GRID_SIZE)
    ap.add_argument("--condition", default=DEFAULT_CONDITION, choices=PPT_CONDITIONS)
    ap.add_argument("--limit", type=int, default=None, help="Optional cap on #regions.")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # resolve regions
    if args.regions:
        regions = [r.strip() for r in args.regions.split(",") if r.strip()]
    elif args.house:
        regions = sorted(p.name for p in data_dir.iterdir() if p.name.startswith(args.house + "_"))
    else:
        raise SystemExit("Provide --regions or --house")

    if args.limit:
        regions = regions[: args.limit]
    print(f"[run] {len(regions)} regions on {args.device}")

    device = torch.device(args.device)
    model = build_model()
    load_checkpoint(model, args.ckpt)
    model = model.to(device).eval()

    for i, name in enumerate(regions):
        rdir = data_dir / name
        out_sub = out_dir / name
        if (out_sub / "feat.npy").exists():
            print(f"[{i+1}/{len(regions)}] skip {name} (cached)")
            continue
        if not rdir.exists():
            print(f"[{i+1}/{len(regions)}] MISSING {rdir}, skipping")
            continue

        coord = np.load(rdir / "coord.npy")
        color = np.load(rdir / "color.npy")
        t0 = time.time()
        feat = infer_region(
            model,
            coord,
            color,
            device=device,
            grid_size=args.grid_size,
            condition=args.condition,
        )
        dt = time.time() - t0

        out_sub.mkdir(parents=True, exist_ok=True)
        np.save(out_sub / "feat.npy", feat)
        # Also persist origin coord (used for matching to original mesh)
        np.save(out_sub / "coord.npy", coord.astype(np.float32))
        print(
            f"[{i+1}/{len(regions)}] {name}: N={feat.shape[0]} D={feat.shape[1]} "
            f"({dt:.1f}s, {feat.nbytes/1e6:.1f} MB)"
        )


if __name__ == "__main__":
    main()
