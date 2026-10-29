#!/usr/bin/env bash
# ------------------------------------------------------------------------------
# H20 one-shot setup for OPSD-I2V Tier-2 training.
#
# Usage (on the H20 container):
#     bash 08-video-opsd-i2v/scripts/h20_setup.sh
#
# What it does:
#   1. Confirms GPU + CUDA + Python environment
#   2. Creates .venv, installs CUDA-matched PyTorch + project deps
#   3. Clones DynamiCrafter under third_party/
#   4. Downloads DynamiCrafter 512 checkpoint (~10GB)
#   5. Pre-caches OpenCLIP ViT-H-14 and ViT-B-32 weights
#   6. Runs the end-to-end smoke test to prove the trainer is alive on H20
#
# Expected total runtime: 20-40 min depending on download speed.
# All stages are idempotent — safe to re-run.
# ------------------------------------------------------------------------------
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

LOG=logs/h20_setup.log
mkdir -p logs weights third_party outputs
step() { echo -e "\n[h20-setup] $(date '+%H:%M:%S')  $*" | tee -a "$LOG"; }

# ------------------------------------------------------------- 0. environment
step "=== STAGE 0: environment probe ==="
echo "uname -a: $(uname -a)" | tee -a "$LOG"
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi | tee -a "$LOG"
else
    echo "WARN: nvidia-smi not found; are we actually on a GPU box?" | tee -a "$LOG"
fi
echo "python: $(python3 --version)" | tee -a "$LOG"
echo "free RAM: $(free -h | grep Mem || true)" | tee -a "$LOG"
echo "disk free: $(df -h . | tail -1)" | tee -a "$LOG"

# ------------------------------------------------------------- 1. python env
step "=== STAGE 1: venv + torch (CUDA build) ==="
if [[ ! -d .venv ]]; then
    python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools 2>&1 | tee -a "$LOG"

# Detect CUDA major from nvidia-smi; default to cu121 (safe for H20 / Hopper).
CUDA_MAJOR="121"
if nvidia-smi &>/dev/null; then
    CU=$(nvidia-smi | grep -oE "CUDA Version: [0-9]+" | head -1 | grep -oE "[0-9]+")
    case "$CU" in
        11) CUDA_MAJOR="118";;
        12) CUDA_MAJOR="121";;
        *)  CUDA_MAJOR="121";;
    esac
fi
step "installing torch (cu${CUDA_MAJOR}) ..."
pip install --quiet torch==2.3.1 torchvision==0.18.1 \
    --index-url "https://download.pytorch.org/whl/cu${CUDA_MAJOR}" 2>&1 | tee -a "$LOG"

step "installing project deps ..."
pip install --quiet \
    diffusers==0.30.0 \
    transformers==4.44.0 \
    peft==0.12.0 \
    open_clip_torch==2.26.1 \
    omegaconf==2.3.0 \
    imageio==2.35.1 imageio-ffmpeg==0.5.1 \
    einops==0.8.0 \
    pyyaml \
    scipy \
    tqdm \
    pytorch-lightning==2.3.3 \
    huggingface_hub==0.24.5 \
    timm==1.0.9 \
    kornia==0.7.3 \
    av==12.3.0 2>&1 | tee -a "$LOG"

python -c "import torch; print(f'[torch] {torch.__version__}  cuda={torch.cuda.is_available()}  dev={torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NONE\"}')" | tee -a "$LOG"

# ------------------------------------------------------------- 2. DynamiCrafter code
step "=== STAGE 2: DynamiCrafter source ==="
if [[ ! -d third_party/DynamiCrafter/lvdm ]]; then
    git clone --depth 1 https://github.com/Doubiiu/DynamiCrafter third_party/DynamiCrafter 2>&1 | tee -a "$LOG"
else
    step "DynamiCrafter already present, skip clone"
fi

# ------------------------------------------------------------- 3. DynamiCrafter 512 ckpt
step "=== STAGE 3: DynamiCrafter 512 checkpoint (~10 GB) ==="
mkdir -p weights/dynamicrafter_512
if [[ ! -f weights/dynamicrafter_512/model.ckpt ]]; then
    step "downloading via huggingface_hub ..."
    python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='Doubiiu/DynamiCrafter_512',
    filename='model.ckpt',
    local_dir='weights/dynamicrafter_512',
    local_dir_use_symlinks=False,
)
print('[*] 512 ckpt downloaded')
" 2>&1 | tee -a "$LOG"
else
    step "ckpt already present ($(du -h weights/dynamicrafter_512/model.ckpt | cut -f1)), skip"
fi

# ------------------------------------------------------------- 4. OpenCLIP weights
step "=== STAGE 4: OpenCLIP ViT-H-14 + ViT-B-32 pre-cache ==="
python - <<'PY' 2>&1 | tee -a "$LOG"
import open_clip
print("[*] pre-caching ViT-H-14 (laion2b_s32b_b79k) ...")
open_clip.create_model_and_transforms(
    "ViT-H-14", pretrained="laion2b_s32b_b79k", device="cpu",
)
print("[*] pre-caching ViT-B-32 (openai) ...")
open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai", device="cpu",
)
print("[*] openclip cached.")
PY

# ------------------------------------------------------------- 5. smoke test
step "=== STAGE 5: end-to-end smoke test on H20 (production shape) ==="
export PYTHONPATH="$HERE"
# On H20 we have the VRAM for the real shape — this is a much more meaningful
# verification than the mini shape used on M3 Pro.
SMOKE_SHAPE=production python scripts/smoke_trainer_e2e.py 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}

echo
if [[ "$rc" -eq 0 ]]; then
    step "=== ✅ H20 SETUP DONE — trainer is alive on CUDA. ==="
    echo "Next: bash scripts/h20_run_tier2.sh" | tee -a "$LOG"
else
    step "=== ❌ smoke failed; see $LOG for details ==="
    exit 1
fi
