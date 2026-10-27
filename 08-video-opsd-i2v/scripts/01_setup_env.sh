#!/usr/bin/env bash
# One-shot env setup for 08-video-opsd-i2v on macOS (M-series) / Linux.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

echo "[1/4] creating venv..."
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -U pip wheel

echo "[2/4] installing torch (MPS build for macOS, CUDA 12.1 wheel for Linux)..."
UNAME="$(uname -s)"
if [[ "$UNAME" == "Darwin" ]]; then
  pip install torch==2.4.1 torchvision==0.19.1
else
  pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
fi

echo "[3/4] installing project requirements..."
pip install -r requirements.txt

echo "[4/4] enabling MPS fallback env var in shell..."
cat > .envrc <<'ENV'
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
ENV
echo "done. activate with: source .venv/bin/activate && source .envrc"
