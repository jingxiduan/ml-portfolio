"""Apply lightness_lift to each BoN winner, score before/after, write metrics + 4-way GIFs.

Produces:
  outputs/tier2_bon_lightness/<title>_bon.mp4             (copy of winner)
  outputs/tier2_bon_lightness/<title>_bon_lightness.mp4   (post-processed)
  outputs/tier2_bon_lightness/bon_plus_lightness.json
  docs/tier2/bon_plus_lightness_table.md
  docs/figures/duel/fourway_<title>.{gif,mp4}             (base|s=0.2|BoN|BoN+lightness)
"""
from __future__ import annotations
import json
import shutil
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from src.postprocess.lightness_lift import lightness_lift
from src.eval.naturalness_v2 import naturalness_v2_video

ROOT = Path(__file__).resolve().parents[1]

# six prompts across R1 and R2
R1 = ROOT / "outputs" / "tier2_bon"
R2 = ROOT / "outputs" / "tier2_bon_r2"
D1 = ROOT / "outputs" / "duel"       # has videos_base / videos_sweet
D2 = ROOT / "outputs" / "duel2"      # has videos_base / videos_sweet

# titles in order: (title, round_dir, duel_dir, prompt_index)
PROMPTS = [
    ("sun_rays",        R1, D1, 0),
    ("houses_hillside", R1, D1, 1),
    ("pagoda_snowy",    R1, D1, 2),
    ("candle_dark",     R2, D2, 0),
    ("fog_pine_dawn",   R2, D2, 1),
    ("waterfall_zoom",  R2, D2, 2),
]

OUT = ROOT / "outputs" / "tier2_bon_lightness"
OUT.mkdir(exist_ok=True)
FIG = ROOT / "docs" / "figures" / "duel"
FIG.mkdir(parents=True, exist_ok=True)


def _winner_path(round_dir: Path, title: str, idx: int) -> Path:
    """Resolve BoN winner path from tier2_bon_metrics.json."""
    data = json.loads((round_dir / "tier2_bon_metrics.json").read_text())
    # R1 keyed by title; R2 keyed by title too
    if title in data:
        return ROOT / data[title]["path"]
    # fallback: find by index
    raise FileNotFoundError(f"{title} not in {round_dir}/tier2_bon_metrics.json")


def _score_mp4(p: Path) -> dict:
    frames = np.stack(list(iio.imiter(str(p))))
    return naturalness_v2_video(frames)


def _apply_lightness(src: Path, dst: Path) -> None:
    frames = np.stack(list(iio.imiter(str(src))))  # uint8 (T,H,W,3)
    out = np.stack([lightness_lift(f) for f in frames])  # returns uint8
    iio.imwrite(str(dst), out, fps=8, codec="libx264")


def _build_fourway(base: Path, s02: Path, bon: Path, lift: Path, out_stem: Path) -> None:
    def _read(p: Path) -> np.ndarray:
        return np.stack(list(iio.imiter(str(p))))
    b, s, w, l = _read(base), _read(s02), _read(bon), _read(lift)
    T = min(len(b), len(s), len(w), len(l))
    gap = np.full((b.shape[1], 16, 3), 255, dtype=np.uint8)
    frames = [np.concatenate([b[k], gap, s[k], gap, w[k], gap, l[k]], axis=1) for k in range(T)]
    iio.imwrite(str(out_stem.with_suffix(".gif")), frames, duration=125, loop=0)
    iio.imwrite(str(out_stem.with_suffix(".mp4")), frames, fps=8, codec="libx264")


def main() -> None:
    rows = []
    for title, rdir, ddir, idx in PROMPTS:
        wpath = _winner_path(rdir, title, idx)
        # paths
        bon_copy  = OUT / f"{title}_bon.mp4"
        bon_lift  = OUT / f"{title}_bon_lightness.mp4"
        base_vid  = ddir / "videos_base"  / "seed_00" / f"{idx:03d}.mp4"
        s02_vid   = ddir / "videos_sweet" / "seed_00" / f"{idx:03d}.mp4"

        if not bon_copy.exists():
            shutil.copyfile(wpath, bon_copy)
        if not bon_lift.exists():
            _apply_lightness(bon_copy, bon_lift)

        m_bon  = _score_mp4(bon_copy)
        m_lift = _score_mp4(bon_lift)
        rows.append({
            "title": title,
            "winner_src": str(wpath.relative_to(ROOT)),
            "bon_nat":    float(m_bon["nat"]),
            "bon_appeal": float(m_bon["appeal"]),
            "lift_nat":    float(m_lift["nat"]),
            "lift_appeal": float(m_lift["appeal"]),
            "lift_lum_mean":  float(m_lift["lum_mean"]),
            "bon_lum_mean":   float(m_bon["lum_mean"]),
            "lift_sat":  float(m_lift["sat"]),
            "bon_sat":   float(m_bon["sat"]),
            "lift_over": float(m_lift["overexp"]),
            "bon_over":  float(m_bon["overexp"]),
        })

        if base_vid.exists() and s02_vid.exists():
            _build_fourway(base_vid, s02_vid, bon_copy, bon_lift,
                           FIG / f"fourway_{title}")
            print(f"[ok] fourway_{title}  nat {m_bon['nat']:.3f}→{m_lift['nat']:.3f}  "
                  f"appeal {m_bon['appeal']:.3f}→{m_lift['appeal']:.3f}")
        else:
            print(f"[warn] missing base/s02 for {title}  (base={base_vid.exists()} s02={s02_vid.exists()})")

    (OUT / "bon_plus_lightness.json").write_text(json.dumps(rows, indent=2))

    md = [
        "# Tier 2 — BoN winner → lightness post-process (6 prompts)",
        "",
        "Column key: `nat` = naturalness_v2 score; `appeal` = 0.6·nat + 0.25·motion + 0.15·bonus_lift.",
        "",
        "| prompt | BoN nat | **BoN+lightness nat** | Δnat | BoN appeal | **BoN+lightness appeal** | Δappeal |",
        "|---|---|---|---|---|---|---|",
    ]
    sum_dn = sum_da = 0.0
    for r in rows:
        dn = r["lift_nat"] - r["bon_nat"]
        da = r["lift_appeal"] - r["bon_appeal"]
        sum_dn += dn
        sum_da += da
        md.append(
            f"| {r['title']} | {r['bon_nat']:.3f} | **{r['lift_nat']:.3f}** | "
            f"{dn:+.3f} | {r['bon_appeal']:.3f} | **{r['lift_appeal']:.3f}** | {da:+.3f} |"
        )
    md.append(f"| **mean (n=6)** |  |  | **{sum_dn/6:+.3f}** |  |  | **{sum_da/6:+.3f}** |")
    md += [
        "",
        "## What this means",
        "",
        "- BoN selects the best seed on `appeal`; lightness then rebalances luminance/sat/edges.",
        "- Δnat and Δappeal here are **not** independent of the reward (`naturalness_v2` is",
        "  explicitly engineered to reward shadow-lift + non-overexposed highlights).",
        "- The honest reading: if the pipeline is *required* to maximise `naturalness_v2`,",
        "  BoN+lightness is the end-to-end answer. For human preference, see `verdict_table.md`",
        "  (lightness vs base/scale on paired Wilcoxon with n=6).",
        "",
        "Companion GIFs: `docs/figures/duel/fourway_<title>.gif` (base | s=0.2 | BoN | BoN+lightness).",
    ]
    (ROOT / "docs" / "tier2" / "bon_plus_lightness_table.md").write_text("\n".join(md) + "\n")

    # mean row for console
    print(f"[done] mean  Δnat={sum_dn/6:+.3f}  Δappeal={sum_da/6:+.3f}  (n=6)")


if __name__ == "__main__":
    main()
