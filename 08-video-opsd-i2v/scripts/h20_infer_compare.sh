#!/usr/bin/env bash
# ------------------------------------------------------------------------------
# H20 inference & comparison: render 6 I2V clips with the newly-trained Tier-2 LoRA,
# place side-by-side with base + scale=0.2 for the final 3-way (or 4-way with lightness).
#
# Usage:
#     bash 08-video-opsd-i2v/scripts/h20_infer_compare.sh [lora_dir]
#
# If lora_dir is omitted, picks the latest outputs/opsd_tier2_h20/lora_step*/
#
# Expected runtime: 30-60 min on H20 (6 prompts × 1 seed × 16 frames × 320×512).
# ------------------------------------------------------------------------------
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

LORA_DIR="${1:-$(ls -1d outputs/opsd_tier2_h20/lora_step*/ 2>/dev/null | sort | tail -1)}"
if [[ -z "$LORA_DIR" || ! -d "$LORA_DIR" ]]; then
    echo "[err] no LoRA dir found. Train first: bash scripts/h20_run_tier2.sh"
    exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate
export PYTHONPATH="$HERE"
mkdir -p logs
LOG=logs/h20_infer.log

echo "[h20-infer] $(date '+%H:%M:%S')  using LoRA: $LORA_DIR" | tee -a "$LOG"

# Render Tier-2-LoRA I2V for all 6 prompts, re-using scale=0.2 first frames
for round in duel duel2; do
    FRAMES="outputs/$round/frames_sweet"
    PROMPTS="outputs/$round/prompts.json"
    OUT="outputs/tier2_lora_rendered/$round"
    if [[ ! -d "$FRAMES" ]]; then
        echo "[h20-infer] missing frames dir $FRAMES — did you forget to pull outputs/ from the portfolio? skip $round" | tee -a "$LOG"
        continue
    fi
    mkdir -p "$OUT"
    echo "[h20-infer] rendering $round with Tier-2 LoRA ..." | tee -a "$LOG"
    python -m src.render_i2v \
        --frames_dir "$FRAMES" \
        --prompts   "$PROMPTS" \
        --out_dir   "$OUT" \
        --seeds 0 --num_frames 16 --fps 8 --limit 3 \
        --i2v_lora_dir "$LORA_DIR" \
        --lora_rank 16 --lora_alpha 32 2>&1 | tee -a "$LOG"
done

# Build 3-way comparison GIFs: base | scale=0.2 | tier2_lora
python - <<'PY' 2>&1 | tee -a "$LOG"
import imageio.v3 as iio, numpy as np
from pathlib import Path
FIG = Path("docs/figures/duel"); FIG.mkdir(parents=True, exist_ok=True)
PAIRS = [
    ("sun_rays",        "duel",  0),
    ("houses_hillside", "duel",  1),
    ("pagoda_snowy",    "duel",  2),
    ("candle_dark",     "duel2", 0),
    ("fog_pine_dawn",   "duel2", 1),
    ("waterfall_zoom",  "duel2", 2),
]
for title, rd, idx in PAIRS:
    b = Path(f"outputs/{rd}/videos_base/seed_00/{idx:03d}.mp4")
    s = Path(f"outputs/{rd}/videos_sweet/seed_00/{idx:03d}.mp4")
    t = Path(f"outputs/tier2_lora_rendered/{rd}/seed_00/{idx:03d}.mp4")
    if not all(p.is_file() for p in (b, s, t)):
        print(f"[skip] {title}: missing one of {b.is_file()}/{s.is_file()}/{t.is_file()}")
        continue
    def _r(p): return np.stack(list(iio.imiter(str(p))))
    B, S, T = _r(b), _r(s), _r(t)
    L = min(len(B), len(S), len(T))
    gap = np.full((B.shape[1], 16, 3), 255, dtype=np.uint8)
    frames = [np.concatenate([B[k], gap, S[k], gap, T[k]], axis=1) for k in range(L)]
    out_stem = FIG / f"tier2_lora_{title}"
    iio.imwrite(str(out_stem.with_suffix(".gif")), frames, duration=125, loop=0)
    iio.imwrite(str(out_stem.with_suffix(".mp4")), frames, fps=8, codec="libx264")
    print(f"[ok] {out_stem.name} (base | scale=0.2 | tier2_lora)")
PY

# Score Tier-2 LoRA renders with naturalness_v2 and emit a metrics table
python - <<'PY' 2>&1 | tee -a "$LOG"
import json
import imageio.v3 as iio, numpy as np
from pathlib import Path
from src.eval.naturalness_v2 import naturalness_v2_video

PAIRS = [
    ("sun_rays",        "duel",  0),
    ("houses_hillside", "duel",  1),
    ("pagoda_snowy",    "duel",  2),
    ("candle_dark",     "duel2", 0),
    ("fog_pine_dawn",   "duel2", 1),
    ("waterfall_zoom",  "duel2", 2),
]
rows = []
for title, rd, idx in PAIRS:
    paths = {
        "base":   f"outputs/{rd}/videos_base/seed_00/{idx:03d}.mp4",
        "s=0.2":  f"outputs/{rd}/videos_sweet/seed_00/{idx:03d}.mp4",
        "tier2":  f"outputs/tier2_lora_rendered/{rd}/seed_00/{idx:03d}.mp4",
    }
    row = {"title": title}
    for k, p in paths.items():
        if not Path(p).is_file():
            row[f"{k}_nat"] = None; row[f"{k}_appeal"] = None
            continue
        frames = np.stack(list(iio.imiter(p)))
        m = naturalness_v2_video(frames)
        row[f"{k}_nat"]    = float(m["nat"])
        row[f"{k}_appeal"] = float(m["appeal"])
    rows.append(row)

out_dir = Path("docs/tier2"); out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "tier2_lora_metrics.json").write_text(json.dumps(rows, indent=2))

md = ["# Tier-2 trained LoRA — H20-rendered comparison (n=6)", "",
      "| prompt | base nat | s=0.2 nat | **tier2 LoRA nat** | Δvs base | Δvs s=0.2 |",
      "|---|---|---|---|---|---|"]
dn_b = dn_s = n = 0
for r in rows:
    if r["tier2_nat"] is None: continue
    db = r["tier2_nat"] - r["base_nat"]
    ds = r["tier2_nat"] - r["s=0.2_nat"]
    dn_b += db; dn_s += ds; n += 1
    md.append(f"| {r['title']} | {r['base_nat']:.3f} | {r['s=0.2_nat']:.3f} | **{r['tier2_nat']:.3f}** | {db:+.3f} | {ds:+.3f} |")
if n:
    md.append(f"| **mean (n={n})** |  |  |  | **{dn_b/n:+.3f}** | **{dn_s/n:+.3f}** |")
md += ["", "Companion GIFs: `docs/figures/duel/tier2_lora_<title>.gif`",
       "  (base | scale=0.2 | tier2_lora, three-way, 16 frames @ 8 fps)."]
(out_dir / "tier2_lora_metrics.md").write_text("\n".join(md) + "\n")
print("[done] wrote docs/tier2/tier2_lora_metrics.{md,json}")
for r in rows:
    print(f"  {r['title']:20s}  tier2_nat={r.get('tier2_nat')}")
PY

echo
echo "[h20-infer] $(date '+%H:%M:%S')  ✅ done"
echo "[h20-infer] metrics: docs/tier2/tier2_lora_metrics.md"
echo "[h20-infer] GIFs:    docs/figures/duel/tier2_lora_*.gif"
