#!/usr/bin/env bash
# ------------------------------------------------------------------------------
# H20 one-shot setup for 07 · Tier-2 BoN rendering.
#
# Usage (on the H20 container):
#     bash 07-talking-head-motion-opsd/scripts/h20_render_setup.sh
#
# What it does:
#   1. GPU + Python env probe
#   2. Creates .venv, installs CUDA-matched PyTorch + project deps
#   3. Clones JoyVASA under third_party/
#   4. Downloads MediaPipe face_landmarker.task
#   5. Downloads pretrained weights (JoyVASA, LivePortrait, HuBERT-chinese, InsightFace)
#      — total ~2.5 GB, placed inside third_party/JoyVASA/pretrained_weights/
#   6. Applies the small compatibility patches to JoyVASA
#   7. Loads selected_coefs.pt + eye_probe + MP landmarker to confirm render
#      pipeline is alive (no actual render yet; keeps smoke < 1 min).
#
# Expected total runtime: 10-20 min depending on download speed.
# All stages are idempotent — safe to re-run.
# ------------------------------------------------------------------------------
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

LOG=logs/h20_render_setup.log
mkdir -p logs third_party
step() { echo -e "\n[07-render-setup] $(date '+%H:%M:%S')  $*" | tee -a "$LOG"; }

# -------------------------------------------------------- 0. environment probe
step "=== STAGE 0: environment probe ==="
echo "uname -a: $(uname -a)" | tee -a "$LOG"
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi | tee -a "$LOG"
else
    echo "WARN: nvidia-smi not found; render will fall back to CPU (slower but works)" | tee -a "$LOG"
fi
echo "python: $(python3 --version)" | tee -a "$LOG"
echo "disk free: $(df -h . | tail -1)" | tee -a "$LOG"

# ------------------------------------------------------------- 1. venv + torch
step "=== STAGE 1: venv + python deps ==="
if [[ ! -d .venv ]]; then
    python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools 2>&1 | tee -a "$LOG"

# Detect CUDA major from nvidia-smi; default to cu121.
CUDA_MAJOR=""
if nvidia-smi &>/dev/null; then
    CU=$(nvidia-smi | grep -oE "CUDA Version: [0-9]+" | head -1 | grep -oE "[0-9]+")
    case "$CU" in
        11) CUDA_MAJOR="118";;
        12) CUDA_MAJOR="121";;
    esac
fi
if [[ -n "$CUDA_MAJOR" ]]; then
    step "installing torch (cu${CUDA_MAJOR}) ..."
    pip install --quiet torch torchvision torchaudio \
        --index-url "https://download.pytorch.org/whl/cu${CUDA_MAJOR}" 2>&1 | tee -a "$LOG"
    ORT_PKG="onnxruntime-gpu"
else
    step "installing torch (cpu) ..."
    pip install --quiet torch torchvision torchaudio 2>&1 | tee -a "$LOG"
    ORT_PKG="onnxruntime"
fi

step "installing project deps ..."
pip install --quiet \
    "numpy<2" opencv-python librosa pyyaml tyro rich \
    imageio imageio-ffmpeg scikit-image onnx "transformers<4.46" pykalman \
    matplotlib scipy tqdm huggingface_hub mediapipe soundfile diffusers \
    "$ORT_PKG" 2>&1 | tee -a "$LOG"

python -c "import torch; print(f'[torch] {torch.__version__}  cuda={torch.cuda.is_available()}')" | tee -a "$LOG"

# ------------------------------------------------------------- 2. JoyVASA code
step "=== STAGE 2: JoyVASA source ==="
if [[ ! -d third_party/JoyVASA/src ]]; then
    git clone --depth 1 https://github.com/jdh-algo/JoyVASA third_party/JoyVASA 2>&1 | tee -a "$LOG"
else
    step "JoyVASA already present, skip clone"
fi

# ------------------------------------------------------------- 3. MediaPipe landmarker
step "=== STAGE 3: MediaPipe face_landmarker.task ==="
mkdir -p third_party/mediapipe
MP_TASK=third_party/mediapipe/face_landmarker.task
if [[ ! -f "$MP_TASK" ]]; then
    curl -L -o "$MP_TASK" \
        https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task \
        2>&1 | tee -a "$LOG"
else
    step "MP landmarker already present, skip"
fi

# ------------------------------------------------------------- 4. pretrained weights
step "=== STAGE 4: pretrained weights (~2.5 GB) ==="
WROOT=third_party/JoyVASA/pretrained_weights
mkdir -p "$WROOT"

# 4a. JoyVASA motion generator + template
if [[ ! -f "$WROOT/JoyVASA/motion_generator/motion_generator_hubert_chinese.pt" ]]; then
    step "downloading JoyVASA (motion_generator + template) ..."
    python - <<PY 2>&1 | tee -a "$LOG"
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="jdh-algo/JoyVASA",
    local_dir="$WROOT/JoyVASA",
    local_dir_use_symlinks=False,
    allow_patterns=["motion_generator/*", "motion_template/*"],
)
print("[*] JoyVASA weights downloaded")
PY
else
    step "JoyVASA weights already present"
fi

# 4b. HuBERT-chinese audio encoder
if [[ ! -f "$WROOT/chinese-hubert-base/pytorch_model.bin" ]] && [[ ! -f "$WROOT/chinese-hubert-base/model.safetensors" ]]; then
    step "downloading HuBERT-chinese ..."
    python - <<PY 2>&1 | tee -a "$LOG"
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="TencentGameMate/chinese-hubert-base",
    local_dir="$WROOT/chinese-hubert-base",
    local_dir_use_symlinks=False,
)
print("[*] HuBERT-chinese downloaded")
PY
    # JoyVASA's default config points at a dir named "hubert-base-ls960" --- symlink for safety.
    if [[ ! -e "$WROOT/hubert-base-ls960" ]]; then
        ln -s chinese-hubert-base "$WROOT/hubert-base-ls960"
    fi
else
    step "HuBERT-chinese already present"
fi

# 4c. LivePortrait (human) base_models + landmark + stitching
if [[ ! -f "$WROOT/liveportrait/base_models/warping_module.pth" ]]; then
    step "downloading LivePortrait human weights ..."
    python - <<PY 2>&1 | tee -a "$LOG"
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="KwaiVGI/LivePortrait",
    local_dir="$WROOT/liveportrait_raw",
    local_dir_use_symlinks=False,
    allow_patterns=["liveportrait/*"],
)
import os, shutil
# flatten KwaiVGI/LivePortrait -> expected path layout
src = "$WROOT/liveportrait_raw/liveportrait"
dst = "$WROOT/liveportrait"
if os.path.isdir(src):
    os.makedirs(dst, exist_ok=True)
    for name in os.listdir(src):
        shutil.move(os.path.join(src, name), os.path.join(dst, name))
    shutil.rmtree("$WROOT/liveportrait_raw", ignore_errors=True)
print("[*] LivePortrait human weights downloaded")
PY
else
    step "LivePortrait weights already present"
fi

# 4d. InsightFace buffalo_l (used by LivePortrait cropper)
if [[ ! -f "$WROOT/insightface/models/buffalo_l/det_10g.onnx" ]]; then
    step "downloading InsightFace buffalo_l ..."
    mkdir -p "$WROOT/insightface/models/buffalo_l"
    curl -L -o "/tmp/buffalo_l.zip" \
        "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip" \
        2>&1 | tee -a "$LOG"
    unzip -o "/tmp/buffalo_l.zip" -d "$WROOT/insightface/models/buffalo_l/" 2>&1 | tee -a "$LOG"
    rm "/tmp/buffalo_l.zip"
else
    step "InsightFace already present"
fi

# ------------------------------------------------------------- 5. patch JoyVASA
step "=== STAGE 5: patch JoyVASA for compatibility ==="
python src/patch_joyvasa.py 2>&1 | tee -a "$LOG"

# ------------------------------------------------------------- 6. smoke test
step "=== STAGE 6: smoke test (no render yet; load pipeline + probe + coefs) ==="
python - <<PY 2>&1 | tee -a "$LOG"
import os, sys, torch
sys.path.insert(0, os.path.abspath("src"))
# load engine with renderer=on; this proves all weights are discoverable
from engine import TalkingHeadEngine
eng = TalkingHeadEngine(load_renderer=True)
print(f"[*] engine loaded, device={eng.device}")

from eye_probe import EyeProbe
p = EyeProbe()
p.load_state_dict(torch.load("outputs/eye_probe/eye_probe.pt", map_location="cpu"))
print(f"[*] eye_probe loaded, params={sum(x.numel() for x in p.parameters())}")

bundle = torch.load("outputs/tier2/selected_coefs.pt", map_location="cpu")
print(f"[*] selected_coefs loaded, {len(bundle['selected_tuples'])} entries, {len(bundle['clips'])} clips")

# touch MP landmarker (lazy init happens inside video_metrics)
from video_metrics import _landmarker
lm = _landmarker()
print(f"[*] MediaPipe landmarker loaded")
print("[*] all pieces in place for rendering.")
PY
rc=${PIPESTATUS[0]}

echo
if [[ "$rc" -eq 0 ]]; then
    step "=== ✅ 07 RENDER SETUP DONE — pipeline is ready. ==="
    cat <<'TXT' | tee -a "$LOG"
Next:
    # render all 8 clips × 4 variants (~10 min on H20 CUDA)
    python src/tier2_bon.py --load-coefs outputs/tier2/selected_coefs.pt --n-eval-clips 8

    # 8 comparison GIFs
    python src/make_tier2_gifs.py --max-side 240 --fps 12

    # commit + push
    git add outputs/tier2/bon_metrics.{json,md} outputs/tier2/tier2_compare_*.gif
    git commit -m "07: Tier-2 BoN+thin · 8 clips rendered on H20"
    git push
TXT
else
    step "=== ❌ setup failed; see $LOG for details ==="
    exit 1
fi
