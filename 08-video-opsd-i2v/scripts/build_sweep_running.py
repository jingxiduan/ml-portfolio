"""Build the sweep comparison figure from whatever scales are done so far.

Idempotent: can be re-run every few minutes while the sweep is in progress.
Produces docs/figures/sweep_running.png with one row per scale, prompts as
columns, metrics burned into each caption.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "outputs" / "scale_sweep"
OUT = ROOT / "docs" / "figures" / "sweep_running.png"


def _metrics(img: Image.Image) -> dict:
    a = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    sat = float(np.where(mx == 0, 0.0, (mx - mn) / np.maximum(mx, 1e-8)).mean())
    over = float((mx >= 0.95).mean())
    gy = np.diff(a.mean(-1), axis=0, prepend=a.mean(-1)[:1])
    gx = np.diff(a.mean(-1), axis=1, prepend=a.mean(-1)[:, :1])
    sob = float(np.percentile(np.sqrt(gx * gx + gy * gy) * 255.0, 95))
    pen = max(0.0, sat - 0.40) + over + max(0.0, (sob - 70.0) / 80.0)
    return {"sat": sat, "over": over, "sob": sob, "nat": max(0.0, 1.0 - pen)}


def main() -> None:
    scales = sorted(d for d in SRC.iterdir() if d.is_dir() and d.name.startswith("scale_"))
    if not scales:
        print("no scales done yet"); return
    pngs_per_scale = {d: sorted(d.glob("*.png")) for d in scales}
    max_prompts = max((len(v) for v in pngs_per_scale.values()), default=0)
    if max_prompts == 0:
        print("no pngs yet"); return

    # Load one sample to get image size
    sample = next(p for v in pngs_per_scale.values() for p in v)
    sample_img = Image.open(sample).convert("RGB")
    iw, ih = sample_img.size

    label = 32
    header = 50
    pad = 10
    cols = max_prompts
    rows = len(scales)
    cw = iw + pad
    ch = ih + label + pad
    canvas = Image.new("RGB",
                       (pad + cols * cw, header + rows * ch + pad),
                       "white")
    draw = ImageDraw.Draw(canvas)
    try:
        f_hdr = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 22)
        f_cap = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 13)
    except Exception:
        f_hdr = ImageFont.load_default(); f_cap = ImageFont.load_default()

    draw.text((pad, 10),
              "LoRA scale sweep (prompts: sun rays over lake | houses hillside | pagoda snowy)",
              fill="black", font=f_hdr)

    for ri, sdir in enumerate(scales):
        scale = float(sdir.name.replace("scale_", ""))
        for ci in range(cols):
            x = pad + ci * cw
            y = header + ri * ch
            pngs = pngs_per_scale[sdir]
            if ci >= len(pngs):
                draw.rectangle([x, y, x + iw, y + ih], fill="#eeeeee")
                draw.text((x + iw // 2 - 30, y + ih // 2), "pending", fill="#888", font=f_cap)
                continue
            img = Image.open(pngs[ci]).convert("RGB").resize((iw, ih))
            canvas.paste(img, (x, y))
            m = _metrics(img)
            # color-code naturalness: green >=0.85, orange 0.5..0.85, red <0.5
            if m["nat"] >= 0.85: col = "darkgreen"
            elif m["nat"] >= 0.5: col = "darkorange"
            else: col = "darkred"
            cap = f"scale={scale:.2f}  sat={m['sat']:.2f}  over={m['over']*100:.1f}%  nat={m['nat']:.2f}"
            draw.text((x, y + ih + 4), cap, fill=col, font=f_cap)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT)
    print(f"wrote {OUT}  ({canvas.size[0]}x{canvas.size[1]}, {rows} scales x {cols} prompts)")


if __name__ == "__main__":
    main()
