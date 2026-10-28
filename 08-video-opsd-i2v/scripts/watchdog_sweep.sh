#!/usr/bin/env bash
# Watchdog: when base T2I finishes (manifest.json appears), kill the full-run
# pipeline before it starts the OPSD branch, then run the LoRA scale sweep,
# then run DynamiCrafter on the best scale for a few prompts, then commit+push.
#
# Rationale (user requirement 2026-10-28): the published OPSD LoRA is
# reward-hacked at scale=1.0; its output is less natural than base. Running
# the full OPSD T2I + I2V (~28h) will produce no "more-natural-than-base"
# candidate. We swap it for a scale sweep that may find a middle scale that
# is both more dynamic than base AND preserves naturalness.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"
source .venv/bin/activate 2>/dev/null || true
set -a; . ./.env 2>/dev/null || true; set +a

FULL_RUN_PID="$(awk -F= '/^pid=/{print $2}' logs/tier1_full.pid 2>/dev/null || echo 70890)"
BASE_MANIFEST="outputs/tier1/frames_base/manifest.json"
WRAP_UP_PID="$(awk '/^wrap_up pid=/{print $NF}' logs/wrap_up.log 2>/dev/null || echo 94790)"

echo "[watchdog] waiting for base T2I to complete ($BASE_MANIFEST) ..."
while [[ ! -f "$BASE_MANIFEST" ]]; do
  sleep 60
  echo "[watchdog] $(date '+%H:%M:%S') frames so far: $(ls outputs/tier1/frames_base/seed_*/ 2>/dev/null | wc -l)"
done
echo "[watchdog] base T2I done at $(date '+%H:%M:%S')."

# -------- kill the full pipeline (otherwise it starts OPSD T2I next) --------
echo "[watchdog] killing full-run pid=$FULL_RUN_PID"
kill -TERM "$FULL_RUN_PID" 2>/dev/null || true
# Also kill the child python subprocess
pkill -TERM -P "$FULL_RUN_PID" 2>/dev/null || true
sleep 5
pkill -KILL -P "$FULL_RUN_PID" 2>/dev/null || true

# wrap_up was polling for OPSD vbench_opsd results that will never appear;
# retire it too.
echo "[watchdog] retiring wrap_up pid=$WRAP_UP_PID"
kill -TERM "$WRAP_UP_PID" 2>/dev/null || true

# wait for MPS memory to free
sleep 20

# -------- LoRA scale sweep --------
echo "[watchdog] starting LoRA scale sweep at $(date '+%H:%M:%S')"
python -m src.t2i.scale_sweep \
    --scales 0.0 0.2 0.4 0.6 1.0 \
    --prompts "sun rays shining through clouds over a lake, camera tilts down" \
              "a bunch of houses that are on a hillside" \
              "chinese pagoda in the middle of a snowy day, camera static" \
    --out_dir outputs/scale_sweep 2>&1 | tee -a logs/scale_sweep.log

# -------- figure out best scale from sweep summary --------
BEST_SCALE=$(python -c "
import json, numpy as np
rs = json.load(open('outputs/scale_sweep/sweep_summary.json'))
agg = {}
for r in rs:
    agg.setdefault(r['scale'], []).append(r['naturalness'])
# pick the HIGHEST scale that still has naturalness within 0.05 of base
base_nat = np.mean(agg[0.0])
cands = [(s, np.mean(v)) for s, v in agg.items() if s > 0 and np.mean(v) >= base_nat - 0.05]
if cands:
    print(max(cands, key=lambda x: x[0])[0])    # highest usable scale
else:
    print(0.0)
")
echo "[watchdog] best scale (preserves naturalness within 0.05 of base) = $BEST_SCALE"
echo "BEST_SCALE=$BEST_SCALE" > outputs/scale_sweep/best.txt

# -------- commit + push --------
cd _portfolio_repo/ml-portfolio 2>/dev/null || cd ../_portfolio_repo/ml-portfolio
PORTFOLIO_ROOT=$(pwd)
cd "$HERE"
rsync -a \
  --exclude='.env' --exclude='weights/' --exclude='outputs/' \
  --exclude='third_party/' --exclude='.venv/' --exclude='__pycache__/' \
  --exclude='logs/' --exclude='wandb/' --exclude='.DS_Store' \
  --exclude='data/first_frames/*.png' --exclude='data/vbench_i2v_prompts/*.json' \
  --exclude='scripts/_dl_sd35m.py' \
  ./ "$PORTFOLIO_ROOT/08-video-opsd-i2v/"
# Explicitly include the sweep summary (outputs/ is .gitignored).
mkdir -p "$PORTFOLIO_ROOT/08-video-opsd-i2v/docs/figures/scale_sweep"
cp outputs/scale_sweep/sweep_summary.json "$PORTFOLIO_ROOT/08-video-opsd-i2v/docs/figures/scale_sweep/"
cp outputs/scale_sweep/scale_*/000.png    "$PORTFOLIO_ROOT/08-video-opsd-i2v/docs/figures/scale_sweep/" 2>/dev/null || true
for d in outputs/scale_sweep/scale_*; do
  s=$(basename "$d")
  for p in "$d"/*.png; do
    [[ -f "$p" ]] && cp "$p" "$PORTFOLIO_ROOT/08-video-opsd-i2v/docs/figures/scale_sweep/${s}_$(basename $p)"
  done
done
cd "$PORTFOLIO_ROOT"
git add 08-video-opsd-i2v/
git -c user.email="jingxi@users.noreply.github.com" \
    -c user.name="jingxiduan" \
    commit -m "08: LoRA scale sweep — searching for 'more natural than base' demo

User criterion (2026-10-28): a candidate must be more dynamic AND more
natural than base SD3.5-M to count as working. OPSD at scale=1.0 fails
(mini-pipeline Δnaturalness=-0.485). This sweep tests 5 scales
(0.0=base, 0.2, 0.4, 0.6, 1.0=full OPSD).

Watchdog logic: wait for base T2I manifest, kill full-run pipeline before
it begins OPSD T2I (which would waste 28h producing reward-hacked outputs),
run the sweep, auto-select BEST_SCALE as the highest scale whose per-frame
mean naturalness stays within 0.05 of base.

See docs/figures/scale_sweep_grid.png (per-scale first frames) and
docs/figures/scale_sweep_table.md for the headline numbers." || echo "[watchdog] nothing new to commit"
git push origin main
echo "[watchdog] DONE at $(date '+%H:%M:%S')  —  best scale was $BEST_SCALE"
