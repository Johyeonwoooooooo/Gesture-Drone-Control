#!/bin/bash
# LitePT 저장소 클론 및 의존성 설치

set -e

PYTHON=/opt/conda/bin/python
PIP=/opt/conda/bin/pip

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── 1. LitePT 클론 ────────────────────────────────────────
if [ ! -d "$PROJECT_DIR/LitePT" ]; then
    git clone https://github.com/prs-eth/LitePT.git "$PROJECT_DIR/LitePT"
else
    echo "LitePT already cloned, skipping."
fi

# ── 2. CUDA 라이브러리 빌드 (RTX 4090 = sm_89) ───────────
export TORCH_CUDA_ARCH_LIST="8.9"

cd "$PROJECT_DIR/LitePT/libs/pointrope"
$PYTHON setup.py install --user

cd "$PROJECT_DIR/LitePT/libs/pointops"
$PYTHON setup.py install --user

cd "$PROJECT_DIR"

# ── 3. pip 패키지 설치 ────────────────────────────────────
$PIP install --user -r "$PROJECT_DIR/requirements.txt"

echo ""
echo "Setup complete. Run inference with:"
echo "  $PYTHON infer_centers.py --npy_dir <room_dir> --out <output.pkl>"
