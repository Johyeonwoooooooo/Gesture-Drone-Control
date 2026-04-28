#!/usr/bin/env bash
# Create conda env `mosaic3d` and install inference-only deps.
# Run inside tmux:
#   tmux new -s mosaic-env -d 'bash 3D-segmentation/env/setup_env.sh 2>&1 | tee 3D-segmentation/env/setup.log'

set -euo pipefail

ENV_NAME=${ENV_NAME:-mosaic3d}
REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
REQ_FILE="$REPO_ROOT/3D-segmentation/env/requirements-inference.txt"

# locate conda
if ! command -v conda >/dev/null; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || \
    source "/data1/workspaces/jgshin22/miniconda3/etc/profile.d/conda.sh"
fi
source "$(conda info --base)/etc/profile.d/conda.sh"

echo "[setup] env: $ENV_NAME"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "[setup] env already exists, reusing"
else
    conda create -n "$ENV_NAME" python=3.10 -y
fi

conda activate "$ENV_NAME"

python -m pip install -U pip wheel setuptools

# Install torch first (cu121) so other wheels resolve against it
python -m pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121

# Install torch-scatter/cluster prebuilt wheels for torch-2.2.2+cu121
python -m pip install torch-scatter torch-cluster -f https://data.pyg.org/whl/torch-2.2.2+cu121.html

# Rest of deps
python -m pip install -r "$REQ_FILE"

# Sanity check
python - <<'PY'
import torch, spconv, transformers, open_clip, hydra, lightning, viser, trimesh
print("torch", torch.__version__, "cuda:", torch.cuda.is_available(), torch.version.cuda)
print("spconv", spconv.__version__)
print("transformers", transformers.__version__)
print("open_clip", open_clip.__version__)
print("hydra", hydra.__version__)
print("lightning", lightning.__version__)
print("viser ok, trimesh ok")
PY

echo "[setup] DONE"
