"""Pick BoN winners for tier2_bon_r2.

Scores every (prompt, seed) MP4 under outputs/tier2_bon_r2/videos/seed_*/00X.mp4
with naturalness_v2_video, picks argmax `appeal` per prompt, writes:

  outputs/tier2_bon_r2/bon_scores.json
  outputs/tier2_bon_r2/winners.json
  docs/tier2/bon_r2_table.md
"""
from __future__ import annotations
import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from src.eval.naturalness_v2 import naturalness_v2_video


def _score(path: Path) -> dict:
    frames = iio.imread(str(path), plugin="FFMPEG")  # (T,H,W,3) uint8
    return naturalness_v2_video(np.asarray(frames))

ROOT = Path(__file__).resolve().parents[1]
VID = ROOT / "outputs" / "tier2_bon_r2" / "videos"
PROMPTS = [
    (0, "a candle flickering in the dark, close-up"),
    (1, "fog drifting through a pine forest at dawn"),
    (2, "a view of a waterfall from a distance, camera zooms out"),
]
SEEDS = [1, 2, 3]


def main() -> None:
    all_scores: list[dict] = []
    winners: list[dict] = []
    for idx, prompt in PROMPTS:
        row = {"index": idx, "prompt": prompt, "candidates": []}
        best = None
        for s in SEEDS:
            fp = VID / f"seed_{s:02d}" / f"{idx:03d}.mp4"
            if not fp.exists():
                continue
            m = _score(fp)
            m_f = {k: float(v) for k, v in m.items()}
            cand = {"seed": s, "file": str(fp.relative_to(ROOT)), **m_f}
            row["candidates"].append(cand)
            if best is None or m_f["appeal"] > best["appeal"]:
                best = cand
        row["winner"] = best
        all_scores.append(row)
        if best is not None:
            winners.append({"index": idx, "prompt": prompt, **best})

    out_dir = ROOT / "outputs" / "tier2_bon_r2"
    (out_dir / "bon_scores.json").write_text(json.dumps(all_scores, indent=2))
    (out_dir / "winners.json").write_text(json.dumps(winners, indent=2))

    md_lines = [
        "# Tier 2 — Best-of-N (R2) results",
        "",
        "3 prompts × 3 seeds (seed=1,2,3), scored with `naturalness_v2_video`.",
        "Winner per prompt = argmax `appeal`.",
        "",
        "| prompt | seed=1 appeal | seed=2 appeal | seed=3 appeal | winner |",
        "|---|---|---|---|---|",
    ]
    for row in all_scores:
        cells = {c["seed"]: c["appeal"] for c in row["candidates"]}
        w = row["winner"]
        md_lines.append(
            f"| {row['prompt'][:48]} | "
            f"{cells.get(1, float('nan')):.3f} | "
            f"{cells.get(2, float('nan')):.3f} | "
            f"{cells.get(3, float('nan')):.3f} | "
            f"**seed={w['seed']}** ({w['appeal']:.3f}) |"
        )

    md_lines += [
        "",
        "## Winner metrics (full)",
        "",
        "| prompt | seed | nat | motion | bonus_lift | appeal |",
        "|---|---|---|---|---|---|",
    ]
    for w in winners:
        md_lines.append(
            f"| {w['prompt'][:48]} | {w['seed']} | "
            f"{w['nat']:.3f} | {w['motion_score']:.3f} | "
            f"{w['bonus_lift']:.3f} | {w['appeal']:.3f} |"
        )

    docs = ROOT / "docs" / "tier2"
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "bon_r2_table.md").write_text("\n".join(md_lines) + "\n")
    print("[ok] wrote bon_scores.json / winners.json / docs/tier2/bon_r2_table.md")
    for w in winners:
        print(f"  prompt {w['index']}: seed={w['seed']}  appeal={w['appeal']:.3f}")


if __name__ == "__main__":
    main()
