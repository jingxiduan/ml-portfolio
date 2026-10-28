"""LoRA scale sweep: generate the same prompt+seed at several OPSD scales,
then score each clip (first frame only — fast) with naturalness-aware metrics.

Produces:
    outputs/scale_sweep/scale_{s}/000.png
    outputs/scale_sweep/sweep_summary.json
    docs/figures/scale_sweep_grid.png
    docs/figures/scale_sweep_table.md

Only computes T2I. The best-scoring scales can then be fed into DynamiCrafter
separately (see scripts/scale_sweep_followup.sh).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image

from .lora_scale import prepare_scaled_lora
from .sd35_loader import SD35Config, generate_first_frame, load_sd35_base, load_sd35_with_opsd_lora
from ..utils.mps_patch import apply_mps_patches, get_device


def _naturalness_frame(img: Image.Image) -> dict:
    """Pixel-level naturalness penalties on a single PIL image (frame)."""
    a = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    sat = float(np.where(mx == 0, 0.0, (mx - mn) / np.maximum(mx, 1e-8)).mean())
    over = float((mx >= 0.95).mean())
    gy = np.diff(a.mean(-1), axis=0, prepend=a.mean(-1)[:1])
    gx = np.diff(a.mean(-1), axis=1, prepend=a.mean(-1)[:, :1])
    sobel_p95 = float(np.percentile(np.sqrt(gx * gx + gy * gy) * 255.0, 95))
    # Same thresholds as src/eval/naturalness_score.py
    pen = (
        max(0.0, sat - 0.40)
        + max(0.0, over - 0.0)
        + max(0.0, (sobel_p95 - 70.0) / 80.0)
    )
    nat = max(0.0, 1.0 - pen)
    return {"sat": sat, "overexposed": over, "sobel_p95": sobel_p95, "naturalness": nat}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", nargs="+", default=[
        "sun rays shining through clouds over a lake, camera tilts down",
        "a bunch of houses that are on a hillside",
    ])
    ap.add_argument("--scales", type=float, nargs="+",
                    default=[0.0, 0.2, 0.4, 0.6, 1.0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="outputs/scale_sweep")
    ap.add_argument("--height", type=int, default=320)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--cfg", type=float, default=4.5)
    args = ap.parse_args()

    apply_mps_patches()
    device = get_device()
    out_root = Path(args.out_dir); out_root.mkdir(parents=True, exist_ok=True)

    results: List[dict] = []

    for scale in args.scales:
        tag = f"scale_{scale:.2f}"
        print(f"\n[sweep] ========== {tag} ==========")
        t0 = time.time()

        # scale == 0 is literally base SD3.5-M; skip LoRA plumbing
        if scale == 0.0:
            cfg = SD35Config(dtype="fp32")
            pipe = load_sd35_base(cfg, device)
        else:
            scaled_dir = prepare_scaled_lora("weights/opsd_lora_sd35m_hpsv3", scale)
            cfg = SD35Config(dtype="fp32", opsd_lora_dir=str(scaled_dir))
            pipe = load_sd35_with_opsd_lora(cfg, device)
        print(f"[sweep] loaded in {time.time()-t0:.1f}s")

        scale_dir = out_root / tag
        scale_dir.mkdir(parents=True, exist_ok=True)
        for i, prompt in enumerate(args.prompts):
            fname = scale_dir / f"{i:03d}.png"
            if not fname.is_file():
                img = generate_first_frame(
                    pipe, prompt=prompt, seed=args.seed,
                    height=args.height, width=args.width,
                    num_inference_steps=args.steps, guidance_scale=args.cfg,
                    device=str(device),
                )
                img.save(fname)
            else:
                img = Image.open(fname)
            m = _naturalness_frame(img)
            m.update({"scale": scale, "prompt_idx": i, "prompt": prompt,
                      "file": str(fname.relative_to(out_root))})
            results.append(m)
            print(f"  prompt {i}: sat={m['sat']:.3f}  overexp={m['overexposed']:.3f}  "
                  f"sobel_p95={m['sobel_p95']:.1f}  naturalness={m['naturalness']:.3f}")

        # Free before next load
        del pipe
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()

    (out_root / "sweep_summary.json").write_text(json.dumps(results, indent=2))
    print(f"\n[sweep] summary at {out_root/'sweep_summary.json'}")

    # ---- figures ----
    try:
        import matplotlib.pyplot as plt
        OUT_FIG = Path("docs/figures"); OUT_FIG.mkdir(parents=True, exist_ok=True)

        # grid: n_prompts rows × n_scales cols
        prompts = sorted({r["prompt_idx"] for r in results})
        scales = sorted({r["scale"] for r in results})
        rows = []
        for pi in prompts:
            row = []
            for s in scales:
                hits = [r for r in results if r["prompt_idx"] == pi and r["scale"] == s]
                if hits:
                    img = np.asarray(Image.open(out_root / hits[0]["file"]).convert("RGB"))
                    # burn caption
                    cap = f"scale={s:.2f}  nat={hits[0]['naturalness']:.2f}"
                    row.append((img, cap))
            rows.append(row)
        fig, axes = plt.subplots(len(prompts), len(scales),
                                 figsize=(3 * len(scales), 2.3 * len(prompts)))
        if len(prompts) == 1:
            axes = axes[None, :]
        for ri, row in enumerate(rows):
            for ci, (img, cap) in enumerate(row):
                ax = axes[ri][ci]
                ax.imshow(img); ax.axis("off")
                ax.set_title(cap, fontsize=9)
        fig.tight_layout()
        fig.savefig(OUT_FIG / "scale_sweep_grid.png", dpi=140)
        plt.close(fig)

        # Markdown table (avg over prompts)
        agg = {s: [] for s in scales}
        for r in results:
            agg[r["scale"]].append(r)
        lines = ["| LoRA scale | sat | overexp | Sobel-p95 | naturalness | note |",
                 "|---|---|---|---|---|---|"]
        best_scale = max(scales, key=lambda s: np.mean([x["naturalness"] for x in agg[s]]))
        for s in scales:
            rs = agg[s]
            sat = np.mean([x["sat"] for x in rs])
            over = np.mean([x["overexposed"] for x in rs])
            sob = np.mean([x["sobel_p95"] for x in rs])
            nat = np.mean([x["naturalness"] for x in rs])
            tag = "**↑ highest naturalness**" if s == best_scale else ""
            if s == 0.0:
                tag = (tag + "  (= base SD3.5-M)").strip()
            lines.append(f"| {s:.2f} | {sat:.3f} | {over:.3f} | {sob:.1f} | **{nat:.3f}** | {tag} |")
        (OUT_FIG / "scale_sweep_table.md").write_text("\n".join(lines))

        print(f"[sweep] figures at {OUT_FIG}")
        print(f"[sweep] best_scale (highest mean naturalness) = {best_scale}")
    except Exception as e:
        print(f"[sweep] figure stage failed: {e}")


if __name__ == "__main__":
    main()
