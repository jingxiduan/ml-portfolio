#!/usr/bin/env bash
# Fetch DynamiCrafter code + weights + VBench + aesthetic predictor.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

mkdir -p third_party weights

# ---- 1. DynamiCrafter code ----
if [[ ! -d third_party/DynamiCrafter ]]; then
  echo "[*] cloning DynamiCrafter..."
  git clone --depth 1 https://github.com/Doubiiu/DynamiCrafter third_party/DynamiCrafter
fi

# ---- 2. DynamiCrafter 320 checkpoint ----
mkdir -p weights/dynamicrafter_320
if [[ ! -f weights/dynamicrafter_320/model.ckpt ]]; then
  echo "[*] downloading DynamiCrafter-320 checkpoint from HuggingFace..."
  # Use huggingface-cli if available, else wget direct.
  if command -v huggingface-cli >/dev/null; then
    huggingface-cli download Doubiiu/DynamiCrafter_320 model.ckpt \
      --local-dir weights/dynamicrafter_320 --local-dir-use-symlinks False
  else
    pip install huggingface_hub >/dev/null
    huggingface-cli download Doubiiu/DynamiCrafter_320 model.ckpt \
      --local-dir weights/dynamicrafter_320 --local-dir-use-symlinks False
  fi
fi

# ---- 3. VBench (eval only) ----
if [[ ! -d third_party/VBench ]]; then
  echo "[*] cloning VBench..."
  git clone --depth 1 https://github.com/Vchitect/VBench third_party/VBench
  pip install -e third_party/VBench
fi

# ---- 4. Aesthetic predictor weights ----
mkdir -p weights
if [[ ! -f weights/aesthetic_predictor_v2.pth ]]; then
  echo "[*] downloading aesthetic predictor v2 weights..."
  curl -L -o weights/aesthetic_predictor_v2.pth \
    "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac%2Blogos%2Bava1-l14-linearMSE.pth"
fi

echo "[done] all weights and code are in place."
echo "       dynamicrafter: weights/dynamicrafter_320/model.ckpt"
echo "       vbench: third_party/VBench"
echo "       aesthetic: weights/aesthetic_predictor_v2.pth"
