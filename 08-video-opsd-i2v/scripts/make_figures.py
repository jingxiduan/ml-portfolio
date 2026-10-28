"""Build Tier 1 showcase figures from whatever runs are already on disk.

Figures produced (all CPU, won't touch MPS used by the full run):

    docs/figures/
      frame_grid_sunrays.png       : 4 frames x 2 methods side-by-side
      frame_grid_pagoda.png        : same for prompt 2
      gif_sunrays_base.gif         : for GitHub README autoplay
      gif_sunrays_opsd.gif
      gif_pagoda_base.gif
      gif_pagoda_opsd.gif
      bar_vbench_lite_delta.png    : OPSD vs base, 4 dims
      pipeline_overview.png        : method diagram

Reads from:
    outputs/tier1_mini/videos_{base,opsd}/seed_00/{000,001}.mp4
    outputs/tier1_mini/frames_{base,opsd}/seed_00/{000,001}.png
    outputs/tier1_mini/vbench_{base,opsd}/eval_results.json

Zero made-up numbers — everything comes from the above files.
"""
from __future__ import annotations

import json
from pathlib import Path

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Source directory can be overridden by env var so wrap_up_full_run.sh can point
# this at outputs/tier1 (the full n=50 run) instead of outputs/tier1_mini.
import os
MINI = ROOT / os.environ.get("OPSD_I2V_SOURCE", "outputs/tier1_mini")
print(f"[make_figures] source = {MINI}")


# ---------------------------------------------------------- utilities


def _label(img: np.ndarray, text: str) -> np.ndarray:
    """Burn a small caption into the top-left of an RGB uint8 numpy array."""
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Arial.ttf", size=14
        )
    except Exception:
        font = ImageFont.load_default()
    draw.rectangle([(0, 0), (len(text) * 8 + 8, 20)], fill=(0, 0, 0))
    draw.text((4, 2), text, fill=(255, 255, 255), font=font)
    return np.asarray(pil)


def _read_mp4(path: Path) -> np.ndarray:
    """Return a [T, H, W, 3] uint8 array."""
    return np.stack(list(iio.imiter(path)))


# ---------------------------------------------------------- grids


def frame_grid(idx: int, label: str, out_name: str) -> None:
    """4-column (frames 0, 5, 10, 15) × 2-row (base, opsd) grid."""
    base = _read_mp4(MINI / "videos_base" / "seed_00" / f"{idx:03d}.mp4")
    opsd = _read_mp4(MINI / "videos_opsd" / "seed_00" / f"{idx:03d}.mp4")
    T = min(len(base), len(opsd))
    picks = [0, T // 3, 2 * T // 3, T - 1]
    rows = []
    for name, vid in [("base", base), ("OPSD", opsd)]:
        row = [_label(vid[t], f"{name} t={t}") for t in picks]
        rows.append(np.concatenate(row, axis=1))
    grid = np.concatenate(rows, axis=0)
    out = OUT / out_name
    Image.fromarray(grid).save(out)
    print(f"[grid] {out}  shape={grid.shape}  prompt={label}")


# ---------------------------------------------------------- gifs


def mp4_to_gif(src: Path, dst: Path, fps: int = 8) -> None:
    frames = _read_mp4(src)
    iio.imwrite(dst, frames, duration=int(1000 / fps), loop=0)
    print(f"[gif] {dst}  frames={len(frames)}  size={dst.stat().st_size // 1024} KB")


# ---------------------------------------------------------- bar chart


def bar_chart() -> None:
    with (MINI / "vbench_base" / "eval_results.json").open() as f:
        base = json.load(f)
    with (MINI / "vbench_opsd" / "eval_results.json").open() as f:
        opsd = json.load(f)
    dims = ["aesthetic_quality", "dynamic_degree", "motion_smoothness",
            "subject_consistency"]
    b = [base[d][0] for d in dims]
    o = [opsd[d][0] for d in dims]
    delta = [oi - bi for bi, oi in zip(b, o)]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(dims))
    w = 0.35
    ax.bar(x - w / 2, b, w, label="base SD3.5-M", color="#8aa6c1")
    ax.bar(x + w / 2, o, w, label="OPSD-tuned SD3.5-M", color="#e07b4a")
    for i, d in enumerate(delta):
        ax.annotate(f"{d:+.3f}", (i, max(b[i], o[i]) + 0.015),
                    ha="center", fontsize=10,
                    color="#d13c13" if d > 0 else "#4a7a3c")
    ax.set_xticks(x)
    ax.set_xticklabels([d.replace("_", "\n") for d in dims], fontsize=9)
    ax.set_ylabel("vbench_lite score")
    ax.set_title("Tier 1 mini-pipeline (n=2 prompts × 1 seed)\n"
                 "OPSD vs base — caveats: dynamic_degree inflation,"
                 " subject_consistency drift")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    out = OUT / "bar_vbench_lite_delta.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[bar] {out}  base={b}  opsd={o}  delta={delta}")


# ---------------------------------------------------------- pipeline diagram


def pipeline_overview() -> None:
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 3.5)
    ax.axis("off")

    def box(x, y, w, h, text, color):
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=color, ec="black", lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=1.4))

    # row 1 — base branch
    box(0.2, 2.2, 1.8, 0.8, "VBench-I2V\nprompt", "#eeeeee")
    box(2.4, 2.2, 2.1, 0.8, "SD3.5-M\n(frozen)", "#cfe2f3")
    box(4.9, 2.2, 1.9, 0.8, "first frame", "#ffd7c2")
    box(7.3, 2.2, 2.0, 0.8, "DynamiCrafter\n(frozen)", "#d9ead3")
    box(9.6, 2.2, 1.2, 0.8, ".mp4", "#f4cccc")
    arrow(2.0, 2.6, 2.4, 2.6)
    arrow(4.5, 2.6, 4.9, 2.6)
    arrow(6.8, 2.6, 7.3, 2.6)
    arrow(9.3, 2.6, 9.6, 2.6)

    # row 2 — opsd branch
    box(2.4, 0.6, 2.1, 0.8, "SD3.5-M\n+ OPSD LoRA", "#9fc5e8")
    box(4.9, 0.6, 1.9, 0.8, "first frame'", "#f6b26b")
    box(7.3, 0.6, 2.0, 0.8, "DynamiCrafter\n(same frozen)", "#b6d7a8")
    box(9.6, 0.6, 1.2, 0.8, ".mp4'", "#ea9999")
    arrow(2.0, 2.4, 2.4, 1.0)
    arrow(4.5, 1.0, 4.9, 1.0)
    arrow(6.8, 1.0, 7.3, 1.0)
    arrow(9.3, 1.0, 9.6, 1.0)

    # eval
    ax.text(5.5, 0.1, "paired → vbench_lite (4 dims) + Wilcoxon  →  carryover table",
            ha="center", fontsize=10, style="italic")

    out = OUT / "pipeline_overview.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[overview] {out}")


# ---------------------------------------------------------- main


if __name__ == "__main__":
    frame_grid(0, "sun rays over a lake", "frame_grid_sunrays.png")
    frame_grid(1, "chinese pagoda", "frame_grid_pagoda.png")
    mp4_to_gif(MINI / "videos_base" / "seed_00" / "000.mp4", OUT / "gif_sunrays_base.gif")
    mp4_to_gif(MINI / "videos_opsd" / "seed_00" / "000.mp4", OUT / "gif_sunrays_opsd.gif")
    mp4_to_gif(MINI / "videos_base" / "seed_00" / "001.mp4", OUT / "gif_pagoda_base.gif")
    mp4_to_gif(MINI / "videos_opsd" / "seed_00" / "001.mp4", OUT / "gif_pagoda_opsd.gif")
    bar_chart()
    pipeline_overview()
    print("[done] figures in", OUT)
