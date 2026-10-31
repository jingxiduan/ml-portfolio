#!/bin/bash
# Train one OPSD LoRA per industry persona (finance / real-estate / short-video).
#
# Design rule: a persona is defined ONLY by the motion band measured from that
# industry's real footage (outputs/industry_bands.json). The reward function,
# its weights, the LoRA config, lr, rounds and seed are byte-identical to v4, so
# any difference between the three checkpoints is attributable to the target band
# and nothing else.
#
# Usage:
#   ROUNDS=3000 GPU=0 bash scripts/train_industries.sh
#   ROUNDS=3000 GPU=0 bash scripts/train_industries.sh finance   # one only
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

DATA=${DATA:-data/audio_public/train}
ROUNDS=${ROUNDS:-3000}
GPU=${GPU:-0}
BANDS=${BANDS:-outputs/industry_bands.json}
ONLY=${1:-}

if [[ ! -f "$BANDS" ]]; then
    echo "[err] $BANDS missing; run scripts/calibrate_industry_bands.py first" >&2
    exit 1
fi

# Emit "name<TAB>amp_target<TAB>amp_tol<TAB>jitter_cap<TAB>blink" per industry.
#   amp_target  = measured median low-band pose std (deg)
#   amp_tol     = v4's verified 1.0, widened only if the industry's own spread is
#                 wider. Kept at >= 1.0 because a narrower Gaussian makes the reward
#                 gradient vanish once the policy drifts off-target; 1.0 is the
#                 value v4 was validated with.
#   jitter_cap  = 2x the measured median high-band std (v4 used 0.20 ~ 2x 0.107)
#   blink       = measured median blinks/min / 100, clipped to the human 10-22 band
python - "$BANDS" <<'PY' > /tmp/industry_args.tsv
import json, sys
d = json.load(open(sys.argv[1]))
for name, v in d["industries"].items():
    lo = v["low_std_deg"]; hi = v["high_std_deg"]; bl = v["blink_per_min"]
    amp = lo["median"]
    tol = max(1.0, 0.5 * (lo["p75"] - lo["p25"]))
    cap = min(0.50, max(0.10, 2.0 * hi["median"]))
    blink = min(0.22, max(0.10, bl["median"] / 100.0))
    print(f"{name}\t{amp:.4f}\t{tol:.4f}\t{cap:.4f}\t{blink:.4f}")
PY

while IFS=$'\t' read -r name amp tol cap blink; do
    [[ -n "$ONLY" && "$name" != "$ONLY" ]] && continue
    out="outputs/opsd_ind_${name}"
    echo "=== industry: ${name}  amp_target=${amp} tol=${tol} jitter_cap=${cap} blink=${blink} ==="
    mkdir -p "$out"
    CUDA_VISIBLE_DEVICES="$GPU" python src/opsd_train_motion_v4.py \
        --out "$out" \
        --data-dir "$DATA" \
        --rounds "$ROUNDS" \
        --amp-target-deg "$amp" \
        --amp-tol-deg "$tol" \
        --jitter-cap-deg "$cap" \
        --blink-target-mean "$blink" \
        --seed 0
done < /tmp/industry_args.tsv

echo "[ok] industry LoRAs trained"
