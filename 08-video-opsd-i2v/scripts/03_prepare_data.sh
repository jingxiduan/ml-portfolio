#!/usr/bin/env bash
# Prepare the VBench-I2V prompt lists and split them into train/eval.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

mkdir -p data/vbench_i2v_prompts data/prompts

VBENCH_PROMPT_JSON="third_party/VBench/vbench_i2v/vbench2_i2v_full_info.json"
if [[ ! -f "$VBENCH_PROMPT_JSON" ]]; then
  echo "[!] VBench-I2V prompts not found at $VBENCH_PROMPT_JSON"
  echo "    Make sure you ran scripts/02_download_weights.sh first."
  exit 1
fi

python3 - <<'PY'
import json, random
from pathlib import Path

src = Path("third_party/VBench/vbench_i2v/vbench2_i2v_full_info.json")
with src.open() as f:
    raw = json.load(f)

# Normalize entries to {id, prompt, category}
entries = []
for i, item in enumerate(raw):
    if isinstance(item, str):
        entries.append({"id": f"vb_{i:04d}", "prompt": item, "category": "unknown"})
    elif isinstance(item, dict):
        entries.append({
            "id": item.get("id", f"vb_{i:04d}"),
            "prompt": item.get("prompt_en") or item.get("prompt") or item.get("caption", ""),
            "category": item.get("dimension", [""])[0] if item.get("dimension") else "unknown",
        })

Path("data/vbench_i2v_prompts/vbench_i2v_full.json").write_text(json.dumps(entries, indent=2))

random.seed(0)
random.shuffle(entries)
cut = int(0.8 * len(entries))
train = entries[:cut]
eval_ = entries[cut:]

Path("data/vbench_i2v_prompts/train_split.json").write_text(json.dumps(train, indent=2))
Path("data/vbench_i2v_prompts/eval_split.json").write_text(json.dumps(eval_, indent=2))

# Build the training prompt list in the (image, prompt) format used by I2VDataset.
# Each entry refers to a user-provided first-frame image by index.
pairs = [{"image": f"{i:03d}.png", "prompt": e["prompt"]} for i, e in enumerate(train)]
Path("data/prompts/train.json").write_text(json.dumps(pairs, indent=2))

pairs_eval = [{"image": f"{i:03d}.png", "prompt": e["prompt"]} for i, e in enumerate(eval_)]
Path("data/prompts/eval.json").write_text(json.dumps(pairs_eval, indent=2))

print(f"train: {len(train)}  eval: {len(eval_)}  total: {len(entries)}")
PY

echo "[done] data lists written to data/vbench_i2v_prompts/ and data/prompts/"
echo
echo "NEXT: drop first-frame images into data/first_frames/000.png, 001.png, ..."
echo "      (one per training prompt). See data/first_frames/README.md."
