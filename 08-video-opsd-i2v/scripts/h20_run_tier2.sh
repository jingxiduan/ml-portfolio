#!/usr/bin/env bash
# ------------------------------------------------------------------------------
# H20 Tier-2 training launcher.
#
# Usage (after h20_setup.sh passes):
#     bash 08-video-opsd-i2v/scripts/h20_run_tier2.sh
#
# What it does:
#   - Loads .venv (assumes h20_setup.sh already ran)
#   - Starts tier2_natural.yaml training with 50 steps, saves LoRA every 10 steps
#   - Live-streams per-step metrics to logs/h20_tier2.log
#   - On success, prints the path of the final LoRA checkpoint
#
# Expected runtime: 4-6 h on H20 single card (production shape).
# Checkpoints: outputs/opsd_tier2_h20/lora_stepXXXXX/  (small, ~15 MB each)
# ------------------------------------------------------------------------------
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

if [[ ! -f .venv/bin/activate ]]; then
    echo "[err] .venv missing. Run bash scripts/h20_setup.sh first."
    exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate

mkdir -p logs outputs/opsd_tier2_h20
LOG=logs/h20_tier2.log

# -------- override config at launch time to persist checkpoints
cat > /tmp/_tier2_override.yaml <<'YAML'
# Minimal overrides on top of configs/tier2_natural.yaml
train:
  ckpt_every: 10            # save LoRA every 10 steps (5 checkpoints total)
  out_dir: "outputs/opsd_tier2_h20"
YAML

# Merge: load tier2_natural.yaml, deep-merge our overrides.
python - <<'PY' > /tmp/_tier2_merged.yaml
import yaml
base = yaml.safe_load(open("configs/tier2_natural.yaml"))
over = yaml.safe_load(open("/tmp/_tier2_override.yaml"))
def merge(a, b):
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            merge(a[k], v)
        else:
            a[k] = v
    return a
print(yaml.safe_dump(merge(base, over), sort_keys=False))
PY

echo "[h20-tier2] $(date '+%H:%M:%S')  starting training ..." | tee -a "$LOG"
echo "[h20-tier2] device:" | tee -a "$LOG"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv | tee -a "$LOG"

export PYTHONPATH="$HERE"
set +e
python -m src.train_entry --method opsd_i2v --config /tmp/_tier2_merged.yaml 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
set -e

if [[ "$rc" -ne 0 ]]; then
    echo "[h20-tier2] ❌ training failed (rc=$rc); see $LOG"
    exit "$rc"
fi

LAST=$(ls -1d outputs/opsd_tier2_h20/lora_step*/ 2>/dev/null | sort | tail -1)
echo
echo "[h20-tier2] ✅ training done at $(date '+%H:%M:%S')"
echo "[h20-tier2] final LoRA: ${LAST:-<none found>}"
echo "[h20-tier2] Next: bash scripts/h20_infer_compare.sh ${LAST:-outputs/opsd_tier2_h20/lora_step00050}"
