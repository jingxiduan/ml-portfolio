#!/usr/bin/env bash
# Tier 1 end-to-end: zero video training, keyframe carryover study.
#
# Pipeline:
#   1. generate first frames from base SD3.5-M        (outputs/tier1/frames_base/)
#   2. generate first frames from OPSD-SD3.5-M        (outputs/tier1/frames_opsd/)
#   3. render videos with frozen DynamiCrafter (both) (outputs/tier1/videos_{base,opsd}/)
#   4. VBench-I2V on both                              (outputs/tier1/vbench_{base,opsd}/)
#   5. Qwen2-VL pairwise judge                         (outputs/tier1/judge.json)
#   6. collect to tier1_main_table.{csv, md}           (outputs/tier1/)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"
source .venv/bin/activate 2>/dev/null || true
[[ -f .envrc ]] && source .envrc || true

CONFIG="${1:-configs/tier1_carryover.yaml}"
echo "[tier1] using config: $CONFIG"

python -m src.tier1_keyframe_pipeline \
  --config "$CONFIG" \
  "${@:2}"

echo
echo "[tier1] DONE."
echo "        headline table: outputs/tier1/tier1_main_table.md"
echo "        stats:          outputs/tier1/stats.json"
echo "        judge:          outputs/tier1/judge.json"
