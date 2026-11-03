#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PROJECT="$REPO/08-video-opsd-i2v"
RUN="${RUN_DIR:-$PROJECT/outputs/opsd_long_v4_gradclip_rerun}"
RUN_REL="${RUN#$REPO/}"
RUN_NAME="$(basename "$RUN")"
EVAL="$PROJECT/outputs/eval_v4"
DEV="$PROJECT/outputs/v4_dev_selection"
LOG="$RUN/finalize.log"

exec > >(tee -a "$LOG") 2>&1
cd "$PROJECT"

TRAIN_PID="$(cat "$RUN/train.pid")"
echo "[finalize] waiting for train pid=$TRAIN_PID"
while kill -0 "$TRAIN_PID" 2>/dev/null; do
    sleep 60
done
if [[ -f "$RUN/guard.pid" ]]; then
    GUARD_PID="$(cat "$RUN/guard.pid")"
    while kill -0 "$GUARD_PID" 2>/dev/null; do
        sleep 5
    done
fi

source .venv/bin/activate
export PYTHONPATH="$PROJECT"
python scripts/plot_v4_summary.py "$RUN/train_metrics.jsonl" --out "$RUN/fig_v4_summary.png"

LAST_STEP="$(python - "$RUN/train_metrics.jsonl" <<'PY'
import json, sys
from pathlib import Path
rows=[json.loads(x) for x in Path(sys.argv[1]).read_text().splitlines() if x.strip()]
print(rows[-1]['step'])
PY
)"
CANDIDATE_TAG="v4_incomplete"

if [[ "$LAST_STEP" == "2000" && -d "$RUN/lora_step02000" && ! -f "$RUN/GUARD_STOP.json" ]]; then
    echo "[finalize] training complete; selecting on six-pair development split"
    python scripts/select_v4_candidate.py --run-dir "$RUN"
    readarray -t SELECTED < <(python - <<'PY'
import json
x=json.load(open('outputs/v4_dev_selection/selection.json'))['selected']
print(x['tag'])
print(x['adapter'])
PY
)
    CANDIDATE_TAG="${SELECTED[0]}"
    CANDIDATE_ADAPTER="${SELECTED[1]}"
    echo "[finalize] selected $CANDIDATE_TAG ($CANDIDATE_ADAPTER)"
    python scripts/eval_candidate.py \
        --candidate-tag "$CANDIDATE_TAG" \
        --candidate-lora "$CANDIDATE_ADAPTER" \
        --frames-dir data/first_frames_eval_coco40 \
        --prompts data/prompts/coco_eval_40.json \
        --limit 40 \
        --out "$EVAL"
    python scripts/make_candidate_gifs.py \
        --base-dir "$EVAL/videos_base/seed_00" \
        --candidate-dir "$EVAL/videos_${CANDIDATE_TAG}/seed_00" \
        --candidate-label "$CANDIDATE_TAG" \
        --prompts data/prompts/coco_eval_40.json \
        --out-dir docs/figures/eval_v4 \
        --indices 0 1 2 3 4 5
fi

python scripts/publish_v4_result.py \
    --repo-root "$REPO" \
    --run-dir "$RUN" \
    --eval-dir "$EVAL" \
    --candidate-tag "$CANDIDATE_TAG"

cd "$REPO"
git add README.md \
    08-video-opsd-i2v/README.md \
    08-video-opsd-i2v/docs/RESULTS_LONG.md \
    08-video-opsd-i2v/operations/v4/README.md \
    08-video-opsd-i2v/scripts

git add -f \
    "$RUN_REL/train_metrics.jsonl" \
    "$RUN_REL/train.log" \
    "$RUN_REL/guard.log" \
    "$RUN_REL/finalize.log" \
    "$RUN_REL/fig_v4_summary.png" \
    "$RUN_REL/FINAL_REPORT.md"
[[ -f "$RUN/GUARD_STOP.json" ]] && git add -f "$RUN_REL/GUARD_STOP.json"

if [[ "$LAST_STEP" == "2000" ]]; then
    git add -f \
        08-video-opsd-i2v/outputs/v4_dev_selection/selection.json \
        "08-video-opsd-i2v/outputs/eval_v4/metrics_${CANDIDATE_TAG}.json" \
        08-video-opsd-i2v/docs/figures/eval_v4
    for index in 000 001 002 003 004 005; do
        git add -f \
            "08-video-opsd-i2v/outputs/eval_v4/videos_base/seed_00/${index}.mp4" \
            "08-video-opsd-i2v/outputs/eval_v4/videos_${CANDIDATE_TAG}/seed_00/${index}.mp4"
    done
fi

git diff --cached --check
git commit -m "Publish final V4 training, held-out metrics and demos"
git push https://github.com/jingxiduan/ml-portfolio.git main

BACKUP="/root/ml-portfolio-backup-worktree"
if [[ -d "$BACKUP/.git" || -f "$BACKUP/.git" ]]; then
    mkdir -p "$BACKUP/08-video-opsd-i2v/outputs"
    rm -rf "$BACKUP/08-video-opsd-i2v/outputs/$RUN_NAME"
    cp -a "$RUN" "$BACKUP/08-video-opsd-i2v/outputs/"
    [[ -d "$DEV" ]] && { rm -rf "$BACKUP/08-video-opsd-i2v/outputs/$(basename "$DEV")"; cp -a "$DEV" "$BACKUP/08-video-opsd-i2v/outputs/"; }
    [[ -d "$EVAL" ]] && { rm -rf "$BACKUP/08-video-opsd-i2v/outputs/$(basename "$EVAL")"; cp -a "$EVAL" "$BACKUP/08-video-opsd-i2v/outputs/"; }
    cd "$BACKUP"
    git add -f "08-video-opsd-i2v/outputs/$RUN_NAME"
    [[ -d 08-video-opsd-i2v/outputs/v4_dev_selection ]] && git add -f 08-video-opsd-i2v/outputs/v4_dev_selection
    [[ -d 08-video-opsd-i2v/outputs/eval_v4 ]] && git add -f 08-video-opsd-i2v/outputs/eval_v4
    git commit -m "Backup full V4 run and evaluation artifacts"
    git push https://github.com/jingxiduan/ml-portfolio.git HEAD:backup/2026-11-03-r3-v3-preflight
fi

rm -f /root/.mlp-github-token /tmp/mlp_git_askpass.sh
echo "[finalize] complete and pushed"
