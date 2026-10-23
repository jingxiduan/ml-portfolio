"""
Generate side-by-side before/after video clips: the frozen AnimateDiff base
(SD1.5 UNet + frozen motion adapter) vs. the same pipeline with the
OPSD-trained LoRA adapter applied, on held-out demo prompts, using a fixed
seed per prompt so the only difference between "before" and "after" is the
LoRA weights. Mirrors `04-diffusion-opsd-mps/src/generate_samples.py`, with
GIFs standing in for video since GIF is the simplest format both this README
and GitHub natively render.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler
from diffusers.utils import export_to_gif
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
BASE_MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
MOTION_ADAPTER_ID = "guoyww/animatediff-motion-adapter-v1-5-2"


def load_prompts(path: Path) -> list[str]:
    return [l.strip() for l in path.read_text().splitlines() if l.strip()]


def generate(pipe, prompts: list[str], seed: int, resolution: int, num_frames: int, steps: int):
    clips = []
    for i, prompt in enumerate(prompts):
        gen = torch.Generator(device="cpu").manual_seed(seed + i)
        out = pipe(
            prompt=prompt, num_frames=num_frames, height=resolution, width=resolution,
            num_inference_steps=steps, guidance_scale=7.5, generator=gen,
        )
        clips.append(out.frames[0])
    return clips


def make_side_by_side_gif(before_frames: list[Image.Image], after_frames: list[Image.Image],
                           prompt: str, out_path: Path, fps: int):
    w, h = before_frames[0].size
    pad = 6
    label_h = 24
    combined = []
    for b, a in zip(before_frames, after_frames):
        canvas = Image.new("RGB", (w * 2 + pad * 3, h + label_h + pad * 2), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((pad, pad), "before (base AnimateDiff)", fill="black")
        draw.text((pad * 2 + w, pad), "after (OPSD LoRA)", fill="black")
        canvas.paste(b, (pad, label_h + pad))
        canvas.paste(a, (pad * 2 + w, label_h + pad))
        combined.append(canvas)
    export_to_gif(combined, str(out_path), fps=fps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--prompts", default=str(ROOT / "data" / "demo_prompts.txt"))
    ap.add_argument("--lora-dir", default=str(ROOT / "outputs" / "lora_trainable"))
    ap.add_argument("--resolution", type=int, default=320)
    ap.add_argument("--num-frames", type=int, default=16)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--fps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "samples"))
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts = load_prompts(Path(args.prompts))
    print(f"[info] {len(prompts)} demo prompts")

    print("[info] loading base AnimateDiff pipeline...")
    adapter = MotionAdapter.from_pretrained(MOTION_ADAPTER_ID, torch_dtype=torch.float32)
    pipe = AnimateDiffPipeline.from_pretrained(BASE_MODEL_ID, motion_adapter=adapter, torch_dtype=torch.float32)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(args.device)
    pipe.set_progress_bar_config(disable=True)

    print("[info] generating BEFORE clips (base model)...")
    before_clips = generate(pipe, prompts, args.seed, args.resolution, args.num_frames, args.steps)
    for i, frames in enumerate(before_clips):
        export_to_gif(frames, str(out_dir / f"before_{i}.gif"), fps=args.fps)

    print(f"[info] loading OPSD LoRA adapter from {args.lora_dir} ...")
    lora_file = str(Path(args.lora_dir) / "pytorch_lora_weights.safetensors")
    pipe.unet.load_lora_adapter(lora_file, prefix=None, adapter_name="trained")
    pipe.unet.set_adapter("trained")

    print("[info] generating AFTER clips (OPSD LoRA applied)...")
    after_clips = generate(pipe, prompts, args.seed, args.resolution, args.num_frames, args.steps)
    for i, frames in enumerate(after_clips):
        export_to_gif(frames, str(out_dir / f"after_{i}.gif"), fps=args.fps)

    for i, (b, a, p) in enumerate(zip(before_clips, after_clips, prompts)):
        make_side_by_side_gif(b, a, p, out_dir / f"comparison_{i}.gif", args.fps)

    (out_dir / "prompts.txt").write_text("\n".join(prompts) + "\n")
    print(f"[ok] saved before/after/comparison GIFs to {out_dir}")


if __name__ == "__main__":
    main()
