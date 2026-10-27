#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"
source .venv/bin/activate
source .envrc

python -m src.train_entry \
  --method dpo_video \
  --config configs/baseline_dpovideo.yaml
