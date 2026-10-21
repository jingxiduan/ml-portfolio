"""
Generate side-by-side before/after samples: the frozen SD-Turbo base model vs.
the same model with the OPSD-trained LoRA adapter applied, for a handful of
held-out demo prompts, using a fixed seed so the only difference is the LoRA
weights.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from diffusers import AutoPipelineForText2Image
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = "stabilityai/sd-turbo"


def load_prompts(path: Path) -> list[str]:
    return [l.strip() for l in path.read_text().splitlines() if l.strip()]


def generate(pipe, prompts: list[str], seed: int, resolution: int) -> list[Image.Image]:
    images = []
    for i, prompt in enumerate(prompts):
        gen = torch.Generator(device="cpu").manual_seed(seed + i)
        image = pipe(
            prompt=prompt, num_inference_steps=2, guidance_scale=0.0,
            height=resolution, width=resolution, generator=gen,
        ).images[0]
        images.append(image)
    return images


def make_grid(before: list[Image.Image], after: list[Image.Image], prompts: list[str]) -> Image.Image:
    n = len(prompts)
    w, h = before[0].size
    pad = 8
    label_h = 28
    grid = Image.new("RGB", (w * 2 + pad * 3, (h + label_h) * n + pad * (n + 1)), "white")
    from PIL import ImageDraw
    draw = ImageDraw.Draw(grid)
    for i in range(n):
        y = pad + i * (h + label_h + pad)
        draw.text((pad, y), "before (base SD-Turbo)", fill="black")
        draw.text((pad * 2 + w, y), "after (OPSD LoRA)", fill="black")
        grid.paste(before[i], (pad, y + label_h))
        grid.paste(after[i], (pad * 2 + w, y + label_h))
    return grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--prompts", default=str(ROOT / "data" / "demo_prompts.txt"))
    ap.add_argument("--lora-dir", default=str(ROOT / "outputs" / "lora_trainable"))
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "samples"))
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts = load_prompts(Path(args.prompts))
    print(f"[info] {len(prompts)} demo prompts")

    print("[info] loading base SD-Turbo pipeline...")
    pipe = AutoPipelineForText2Image.from_pretrained(MODEL_ID, torch_dtype=torch.float32).to(args.device)
    pipe.set_progress_bar_config(disable=True)

    print("[info] generating BEFORE images (base model)...")
    before_images = generate(pipe, prompts, args.seed, args.resolution)
    for i, img in enumerate(before_images):
        img.save(out_dir / f"before_{i}.png")

    print(f"[info] loading OPSD LoRA adapter from {args.lora_dir} ...")
    lora_file = str(Path(args.lora_dir) / "pytorch_lora_weights.safetensors")
    pipe.unet.load_lora_adapter(lora_file, prefix=None, adapter_name="trained")
    pipe.unet.set_adapter("trained")

    print("[info] generating AFTER images (OPSD LoRA applied)...")
    after_images = generate(pipe, prompts, args.seed, args.resolution)
    for i, img in enumerate(after_images):
        img.save(out_dir / f"after_{i}.png")

    grid = make_grid(before_images, after_images, prompts)
    grid.save(out_dir / "comparison_grid.png")
    (out_dir / "prompts.txt").write_text("\n".join(prompts) + "\n")
    print(f"[ok] saved before/after images and comparison_grid.png to {out_dir}")


if __name__ == "__main__":
    main()
