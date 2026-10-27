#!/usr/bin/env bash
# OPSD-I2V main training. Takes ~2-3 h on M3 Pro for 500 steps.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"
source .venv/bin/activate
source .envrc

SEED="${1:-0}"
RUN_NAME="opsd-i2v-seed${SEED}"
export WANDB_RUN_NAME="$RUN_NAME"

python -m src.train_entry \
  --method opsd_i2v \
  --config configs/opsd_i2v_default.yaml
