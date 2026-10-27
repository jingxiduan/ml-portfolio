#!/usr/bin/env bash
# Run VBench-I2V on 4 ckpts: base, opsd_i2v, instructvideo, dpo_video.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"
source .venv/bin/activate
source .envrc

STEP="${1:-500}"
EVAL_PROMPTS="data/prompts/eval.json"
FRAMES="data/first_frames"

for TAG in base opsd_i2v instructvideo dpo_video; do
  OUT="outputs/vbench_eval/${TAG}"
  mkdir -p "$OUT"
  if [[ "$TAG" == "base" ]]; then
    CKPT="weights/dynamicrafter_320"   # no LoRA => evaluator falls back to base
  else
    CKPT="outputs/${TAG}/lora_step$(printf '%05d' "$STEP")"
  fi
  echo "[eval] $TAG  ckpt=$CKPT  out=$OUT"
  python -m src.eval.vbench_i2v_eval \
    --ckpt "$CKPT" \
    --tag "$TAG" \
    --out_dir "$OUT" \
    --prompts "$EVAL_PROMPTS" \
    --first_frames "$FRAMES"
done

python -m src.eval.collect_metrics \
  --methods \
    base:outputs/vbench_eval/base/summary.json \
    opsd_i2v:outputs/vbench_eval/opsd_i2v/summary.json \
    instructvideo:outputs/vbench_eval/instructvideo/summary.json \
    dpo_video:outputs/vbench_eval/dpo_video/summary.json \
  --out_dir outputs/tables
