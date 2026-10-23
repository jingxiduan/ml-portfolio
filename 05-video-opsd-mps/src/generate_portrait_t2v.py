"""
Text-to-video portrait demo: same OPSD-trained LoRA as `generate_samples_video.py`,
applied to `data/portrait_prompts.txt` instead of the object/scene demo prompts.
Kept as a separate script (rather than just pointing `generate_samples_video.py`
at a different prompt file) because portrait faces at this resolution/step-count
are a harder case for the base model: most seeds produce venetian-blind-style
striping artifacts on faces, so each prompt below uses a seed hand-picked for a
recognizable result, not just `base_seed + index` like the object demos.
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

# (prompt, seed) - seeds hand-picked per prompt, see module docstring.
PORTRAIT_PROMPTS = [
    ("a young woman with curly hair laughing and tossing her head back", 456),
    ("a man in a suit smiling and nodding his head, upper body portrait, office background", 55),
]


def make_side_by_side_gif(before_frames, after_frames, out_path: Path, fps: int):
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
    ap.add_argument("--lora-dir", default=str(ROOT / "outputs" / "lora_trainable"))
    ap.add_argument("--resolution", type=int, default=320)
    ap.add_argument("--num-frames", type=int, default=16)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--fps", type=int, default=4)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "samples"))
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[info] loading base AnimateDiff pipeline...")
    adapter = MotionAdapter.from_pretrained(MOTION_ADAPTER_ID, torch_dtype=torch.float32)
    pipe = AnimateDiffPipeline.from_pretrained(BASE_MODEL_ID, motion_adapter=adapter, torch_dtype=torch.float32)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(args.device)
    pipe.set_progress_bar_config(disable=True)

    before_clips = []
    for prompt, seed in PORTRAIT_PROMPTS:
        gen = torch.Generator(device="cpu").manual_seed(seed)
        out = pipe(prompt=prompt, num_frames=args.num_frames, height=args.resolution, width=args.resolution,
                    num_inference_steps=args.steps, guidance_scale=7.5, generator=gen)
        before_clips.append(out.frames[0])

    print(f"[info] loading OPSD LoRA adapter from {args.lora_dir} ...")
    lora_file = str(Path(args.lora_dir) / "pytorch_lora_weights.safetensors")
    pipe.unet.load_lora_adapter(lora_file, prefix=None, adapter_name="trained")
    pipe.unet.set_adapter("trained")

    after_clips = []
    for prompt, seed in PORTRAIT_PROMPTS:
        gen = torch.Generator(device="cpu").manual_seed(seed)
        out = pipe(prompt=prompt, num_frames=args.num_frames, height=args.resolution, width=args.resolution,
                    num_inference_steps=args.steps, guidance_scale=7.5, generator=gen)
        after_clips.append(out.frames[0])

    for i, (b, a) in enumerate(zip(before_clips, after_clips)):
        export_to_gif(b, str(out_dir / f"portrait_t2v_before_{i}.gif"), fps=args.fps)
        export_to_gif(a, str(out_dir / f"portrait_t2v_after_{i}.gif"), fps=args.fps)
        make_side_by_side_gif(b, a, out_dir / f"portrait_t2v_comparison_{i}.gif", args.fps)

    print(f"[ok] saved portrait_t2v_*.gif to {out_dir}")


if __name__ == "__main__":
    main()
