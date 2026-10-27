#!/usr/bin/env bash
# Fetch all required weights: DynamiCrafter + VBench + aesthetic predictor + SD3.5-M + OPSD LoRA.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"
mkdir -p third_party weights

# Load HF token if present (needed for gated SD3.5-M)
if [[ -f .env ]]; then
  set -o allexport; source .env; set +o allexport
fi

# ---- 1. DynamiCrafter code ----
if [[ ! -d third_party/DynamiCrafter ]]; then
  echo "[*] cloning DynamiCrafter..."
  git clone --depth 1 https://github.com/Doubiiu/DynamiCrafter third_party/DynamiCrafter
fi

# ---- 2. DynamiCrafter 320 checkpoint ----
mkdir -p weights/dynamicrafter_320
if [[ ! -f weights/dynamicrafter_320/model.ckpt ]]; then
  echo "[*] downloading DynamiCrafter-320..."
  pip install -q huggingface_hub
  huggingface-cli download Doubiiu/DynamiCrafter_320 model.ckpt \
    --local-dir weights/dynamicrafter_320 --local-dir-use-symlinks False
fi

# ---- 3. VBench ----
if [[ ! -d third_party/VBench ]]; then
  echo "[*] cloning + installing VBench..."
  git clone --depth 1 https://github.com/Vchitect/VBench third_party/VBench
  pip install -e third_party/VBench
fi

# ---- 4. Aesthetic predictor ----
if [[ ! -f weights/aesthetic_predictor_v2.pth ]]; then
  echo "[*] downloading aesthetic predictor..."
  curl -L -o weights/aesthetic_predictor_v2.pth \
    "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac%2Blogos%2Bava1-l14-linearMSE.pth"
fi

# ---- 5. SD3.5-M (GATED — accept license at https://huggingface.co/stabilityai/stable-diffusion-3.5-medium first!) ----
if [[ ! -d weights/sd35m_cache ]]; then
  echo "[*] pre-downloading SD3.5-M (gated model, needs HF login + accepted license)..."
  if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "    HF_TOKEN not in .env; invoking interactive 'huggingface-cli login'..."
    huggingface-cli login
  else
    huggingface-cli login --token "$HF_TOKEN"
  fi
  python3 - <<'PY'
from diffusers import StableDiffusion3Pipeline
import torch
_ = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-medium",
    torch_dtype=torch.float16,
    cache_dir="weights/sd35m_cache",
)
print("[*] SD3.5-M cached.")
PY
fi

# ---- 6. OPSD LoRA (public, Apache-2.0) ----
if [[ ! -d weights/opsd_lora_sd35m_hpsv3 ]] || [[ -z "$(ls -A weights/opsd_lora_sd35m_hpsv3 2>/dev/null)" ]]; then
  echo "[*] downloading OPSD LoRA (sd35-m-hpsv3 variant)..."
  mkdir -p weights/opsd_lora_sd35m_hpsv3
  huggingface-cli download WeiChow/DiffusionOPSD \
    --include "sd35-m-hpsv3/*" \
    --local-dir weights/opsd_lora_sd35m_hpsv3 \
    --local-dir-use-symlinks False
  # move content up one level if the download nested
  if [[ -d weights/opsd_lora_sd35m_hpsv3/sd35-m-hpsv3 ]]; then
    mv weights/opsd_lora_sd35m_hpsv3/sd35-m-hpsv3/* weights/opsd_lora_sd35m_hpsv3/
    rmdir weights/opsd_lora_sd35m_hpsv3/sd35-m-hpsv3 || true
  fi
fi

echo
echo "[done] all weights ready:"
echo "       dynamicrafter : weights/dynamicrafter_320/model.ckpt"
echo "       vbench        : third_party/VBench"
echo "       aesthetic     : weights/aesthetic_predictor_v2.pth"
echo "       SD3.5-M       : weights/sd35m_cache/"
echo "       OPSD LoRA     : weights/opsd_lora_sd35m_hpsv3/"
