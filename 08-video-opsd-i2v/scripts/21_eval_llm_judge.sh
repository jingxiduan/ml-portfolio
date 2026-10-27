#!/usr/bin/env bash
# Qwen2-VL pairwise comparison: OPSD-I2V vs {base, instructvideo, dpo_video}.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"
source .venv/bin/activate
source .envrc

N_PAIRS="${1:-200}"

for OPP in base instructvideo dpo_video; do
  echo "[judge] OPSD-I2V vs $OPP"
  python -m src.eval.llm_judge \
    --a outputs/vbench_eval/opsd_i2v \
    --b outputs/vbench_eval/${OPP} \
    --out outputs/judge/opsd_vs_${OPP}.json \
    --n_pairs "$N_PAIRS"
done
