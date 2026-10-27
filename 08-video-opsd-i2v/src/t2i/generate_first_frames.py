"""Batch generate first frames for a prompt list, with and without OPSD LoRA.

Usage:
    python -m src.t2i.generate_first_frames \
        --prompts data/vbench_i2v_prompts/eval_split.json \
        --out_dir outputs/tier1/frames_base \
        --mode base \
        --seeds 0 1 2

    python -m src.t2i.generate_first_frames \
        --prompts data/vbench_i2v_prompts/eval_split.json \
        --out_dir outputs/tier1/frames_opsd \
        --mode opsd \
        --seeds 0 1 2

Output layout:
    out_dir/
      seed_00/
        000.png                  # frame for prompt index 0 at seed 0
        001.png
        ...
      seed_01/
        ...
      manifest.json              # [{index, prompt, seed, file}, ...]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List

import torch

from ..utils.mps_patch import apply_mps_patches, get_device
from .sd35_loader import SD35Config, generate_first_frame, load_sd35_base, load_sd35_with_opsd_lora


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True, help="JSON file with [{id, prompt, ...}] list")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--mode", choices=["base", "opsd"], required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--limit", type=int, default=None,
                    help="process only the first N prompts (smoke test)")
    ap.add_argument("--height", type=int, default=320)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--cfg", type=float, default=4.5)
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "fp32", "bf16"])
    ap.add_argument("--cpu_offload", action="store_true")
    args = ap.parse_args()

    apply_mps_patches()
    device = get_device()

    sd_cfg = SD35Config(dtype=args.dtype, enable_cpu_offload=args.cpu_offload)
    print(f"[t2i] loading SD3.5-M ({args.mode}) on {device}...")
    if args.mode == "base":
        pipe = load_sd35_base(sd_cfg, device)
    else:
        pipe = load_sd35_with_opsd_lora(sd_cfg, device)
    print("[t2i] loaded.")

    with open(args.prompts) as f:
        prompts: List[dict] = json.load(f)
    if args.limit:
        prompts = prompts[: args.limit]

    out_root = Path(args.out_dir)
    manifest = []
    t0 = time.time()

    for seed in args.seeds:
        seed_dir = out_root / f"seed_{seed:02d}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        for i, item in enumerate(prompts):
            fname = f"{i:03d}.png"
            fpath = seed_dir / fname
            if fpath.exists():
                manifest.append({
                    "index": i, "prompt": item["prompt"], "seed": seed,
                    "file": str(fpath.relative_to(out_root)),
                })
                continue
            img = generate_first_frame(
                pipe,
                prompt=item["prompt"],
                seed=seed,
                height=args.height,
                width=args.width,
                num_inference_steps=args.steps,
                guidance_scale=args.cfg,
                device=str(device),
            )
            img.save(fpath)
            manifest.append({
                "index": i, "prompt": item["prompt"], "seed": seed,
                "file": str(fpath.relative_to(out_root)),
            })
            if (i + 1) % 20 == 0:
                elapsed = time.time() - t0
                total_done = len(manifest)
                total_to_do = len(prompts) * len(args.seeds)
                eta = elapsed * (total_to_do - total_done) / max(total_done, 1)
                print(f"[t2i] seed={seed} {i+1}/{len(prompts)}  "
                      f"({total_done}/{total_to_do} overall, ETA {eta/60:.1f} min)")

    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[t2i] done. {len(manifest)} frames in {out_root}. "
          f"Total {time.time()-t0:.1f}s.")


if __name__ == "__main__":
    main()
