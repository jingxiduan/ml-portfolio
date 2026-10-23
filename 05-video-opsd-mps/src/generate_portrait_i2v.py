"""
Image-to-video portrait demo using Stable Video Diffusion (SVD-img2vid).

This is a separate capability demo, NOT part of the OPSD-trained pipeline:
AnimateDiff (the model OPSD was trained on in this project) is text-to-video
only, it has no image-conditioning path, so there is nothing for an
OPSD-trained LoRA to apply to here. SVD is a different, frozen, pretrained
image-to-video model, included to answer "does image-to-video work on this
machine" and to produce portrait motion from a still photo. No OPSD training
or LoRA is involved in this script.

Source portrait stills are synthesized with the frozen SD-Turbo pipeline
(from `04-diffusion-opsd-mps`) rather than photos of real people, so there is
no real-person likeness involved.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from diffusers import AutoPipelineForText2Image, StableVideoDiffusionPipeline
from diffusers.utils import export_to_gif

ROOT = Path(__file__).resolve().parent.parent
SD_TURBO_ID = "stabilityai/sd-turbo"
SVD_ID = "stabilityai/stable-video-diffusion-img2vid"

PORTRAIT_PROMPTS = [
    ("studio portrait photo of a young woman smiling, neutral gray background, "
     "soft lighting, photorealistic, shoulders up", 42),
    ("studio portrait photo of a middle-aged man with short gray beard, smiling, "
     "neutral dark background, soft lighting, photorealistic, shoulders up", 7),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--resolution", type=int, default=384)
    ap.add_argument("--num-frames", type=int, default=14)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--fps", type=int, default=5)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "samples"))
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[info] generating portrait stills with SD-Turbo...")
    t2i = AutoPipelineForText2Image.from_pretrained(SD_TURBO_ID, torch_dtype=torch.float32)
    t2i = t2i.to(args.device)
    t2i.set_progress_bar_config(disable=True)

    sources = []
    for i, (prompt, seed) in enumerate(PORTRAIT_PROMPTS):
        gen = torch.Generator(device="cpu").manual_seed(seed)
        img = t2i(prompt=prompt, num_inference_steps=4, guidance_scale=0.0, generator=gen).images[0]
        img = img.resize((args.resolution, args.resolution))
        img.save(out_dir / f"portrait_source_{i}.png")
        sources.append(img)
    del t2i

    print("[info] loading SVD img2vid pipeline...")
    i2v = StableVideoDiffusionPipeline.from_pretrained(SVD_ID, torch_dtype=torch.float32)
    i2v = i2v.to(args.device)
    i2v.set_progress_bar_config(disable=True)

    for i, img in enumerate(sources):
        print(f"[info] animating portrait {i}...")
        gen = torch.Generator(device="cpu").manual_seed(i)
        out = i2v(
            image=img, height=args.resolution, width=args.resolution, num_frames=args.num_frames,
            num_inference_steps=args.steps, decode_chunk_size=2, fps=6,
            motion_bucket_id=40, noise_aug_strength=0.02, generator=gen,
        )
        export_to_gif(out.frames[0], str(out_dir / f"portrait_i2v_{i}.gif"), fps=args.fps)

    print(f"[ok] saved portrait_source_*.png and portrait_i2v_*.gif to {out_dir}")


if __name__ == "__main__":
    main()
