#!/usr/bin/env bash
# Push 08-video-opsd-i2v to jingxiduan/ml-portfolio (main branch, sibling of 07).
#
# PREREQ: git clone of ml-portfolio exists somewhere, and you have write access.
# Usage:
#    PORTFOLIO_PATH=/path/to/ml-portfolio bash scripts/99_push_to_github.sh
#
# Set PORTFOLIO_PATH to the local checkout. The script will:
#   1. Copy the 08-video-opsd-i2v/ folder into ${PORTFOLIO_PATH}/
#   2. git add + commit on main
#   3. Push.
#
# IF YOU DO NOT HAVE ml-portfolio CLONED YET:
#   gh repo clone jingxiduan/ml-portfolio
#   # OR:
#   git clone git@github.com:jingxiduan/ml-portfolio.git
#   export PORTFOLIO_PATH=$PWD/ml-portfolio
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$HERE"
DST_ROOT="${PORTFOLIO_PATH:?PORTFOLIO_PATH env var must be set to your ml-portfolio checkout}"
DST="${DST_ROOT}/08-video-opsd-i2v"

if [[ ! -d "$DST_ROOT/.git" ]]; then
  echo "ERROR: $DST_ROOT is not a git repository."
  echo "       Clone it first:  git clone git@github.com:jingxiduan/ml-portfolio.git"
  exit 1
fi

echo "[push] syncing $SRC  →  $DST"
mkdir -p "$DST"

# rsync excludes mirror what .gitignore excludes (weights, outputs, third_party, data blobs).
rsync -av --delete \
  --exclude ".venv" \
  --exclude ".envrc" \
  --exclude "weights/" \
  --exclude "third_party/" \
  --exclude "outputs/" \
  --exclude "wandb/" \
  --exclude "data/first_frames/*.png" \
  --exclude "data/first_frames/*.jpg" \
  --exclude "data/vbench_i2v_prompts/*.json" \
  --exclude "__pycache__" \
  --exclude ".DS_Store" \
  "$SRC/" "$DST/"

cd "$DST_ROOT"
git checkout main
git pull --rebase origin main
git add 08-video-opsd-i2v
if git diff --cached --quiet; then
  echo "[push] nothing to commit."
  exit 0
fi
MSG="08: OPSD-I2V scaffold — code + docs + configs"
git commit -m "$MSG"
git push origin main
echo "[push] done."
