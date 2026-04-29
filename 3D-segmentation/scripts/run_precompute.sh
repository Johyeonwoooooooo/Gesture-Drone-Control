#!/usr/bin/env bash
# Precompute per-point CLIP features for an entire house.
# Usage:
#   bash scripts/run_precompute.sh 17DRP5sb8fy [cuda:0]
set -euo pipefail

HOUSE=${1:-17DRP5sb8fy}
DEVICE=${2:-cuda:0}

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
SEG_ROOT="$REPO_ROOT/3D-segmentation"

source /data1/workspaces/jgshin22/miniconda3/etc/profile.d/conda.sh
conda activate mosaic3d

cd "$SEG_ROOT"

python inference/run_inference.py \
    --ckpt "$REPO_ROOT/data/spunet101.ckpt" \
    --data-dir "$REPO_ROOT/data/hm3d/train" \
    --out-dir "$SEG_ROOT/cache/feat" \
    --house "$HOUSE" \
    --device "$DEVICE"

echo "[done] precompute for $HOUSE finished"
