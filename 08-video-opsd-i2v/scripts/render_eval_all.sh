#!/usr/bin/env bash
# Render evaluation videos for all 4 configs (base, v1, v2, no_nat)
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONPATH="$PWD"

LIMIT="${1:-40}"  # default 40 prompts
echo "[$(date)] Rendering eval videos (limit=$LIMIT)"

COMMON="--frames_dir data/first_frames_eval --prompts data/prompts/tier2_eval_40.json --seeds 0 --num_frames 16 --fps 8 --limit $LIMIT"

echo "=== BASE (no LoRA) ==="
python -m src.render_i2v $COMMON --out_dir outputs/eval_cvpr/videos_base

echo "=== V1 (nat_clamp) ==="
python -m src.render_i2v $COMMON --out_dir outputs/eval_cvpr/videos_v1_nat_clamp --i2v_lora_dir outputs/opsd_tier2_cvpr/lora_step00500

echo "=== V2 (nat_sigmoid) ==="
python -m src.render_i2v $COMMON --out_dir outputs/eval_cvpr/videos_v2_nat_sigmoid --i2v_lora_dir outputs/opsd_tier2_cvpr_v2/lora_step00500

echo "=== NoNat (ablation) ==="
python -m src.render_i2v $COMMON --out_dir outputs/eval_cvpr/videos_ablation_no_nat --i2v_lora_dir outputs/opsd_tier2_cvpr_no_nat/lora_step00500

echo "[$(date)] All renders complete!"
