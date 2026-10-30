#!/usr/bin/env bash
# Train OPSD-I2V LoRA — quality (500 steps, 200 prompts)
set -euo pipefail
cd "$(dirname "$0")/.."

source .venv/bin/activate
export PYTHONPATH="$PWD"

CONFIG="${1:-configs/tier2.yaml}"

echo "[$(date)] Starting Tier-2 OPSD-I2V training"
echo "  Config: $CONFIG"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

python -m src.train_entry --method opsd_i2v --config "$CONFIG"

echo "[$(date)] Training complete"
echo "  Metrics: outputs/opsd_tier2/train_metrics.jsonl"
echo "  LoRA checkpoints: outputs/opsd_tier2/lora_step*/"
