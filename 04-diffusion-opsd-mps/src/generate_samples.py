"""Render fixed, non-cherry-picked Base vs OPSD panels for scenes and portraits."""
from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import torch
from diffusers import AutoPipelineForText2Image, EulerDiscreteScheduler
from peft import LoraConfig, set_peft_model_state_dict
from PIL import Image, ImageDraw, ImageFont
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = "stabilityai/sdxl-turbo"


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def generate(pipe, prompts: list[str], seed: int, resolution: int, steps: int) -> list[Image.Image]:
    images = []
    for index, prompt in enumerate(prompts):
        generator = torch.Generator(device="cpu").manual_seed(seed + index)
        images.append(pipe(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=0.0,
            height=resolution,
            width=resolution,
            generator=generator,
        ).images[0])
    return images


def make_grid(base: list[Image.Image], candidate: list[Image.Image], prompts: list[str]) -> Image.Image:
    width, height = base[0].size
    pad = 16
    header = 34
    caption = 64
    row_height = header + height + caption
    grid = Image.new("RGB", (2 * width + 3 * pad, len(prompts) * row_height + (len(prompts) + 1) * pad), "white")
    draw = ImageDraw.Draw(grid)
    font = ImageFont.load_default()
    for index, prompt in enumerate(prompts):
        top = pad + index * (row_height + pad)
        draw.text((pad, top), "BASE · SDXL-Turbo", fill="black", font=font)
        draw.text((2 * pad + width, top), "OPSD · EMA ADAPTER", fill="black", font=font)
        image_top = top + header
        grid.paste(base[index], (pad, image_top))
        grid.paste(candidate[index], (2 * pad + width, image_top))
        wrapped = "\n".join(textwrap.wrap(prompt, width=105)[:3])
        draw.multiline_text((pad, image_top + height + 8), wrapped, fill="black", font=font, spacing=2)
    return grid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=("scene", "portrait"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--inference-steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--lora-dir", default=None)
    parser.add_argument("--adapter-scale", type=float, default=1.0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    rows = load_rows(ROOT / "data" / f"showcase_{args.domain}.jsonl")
    prompts = [item["prompt"] for item in rows]
    lora_dir = Path(args.lora_dir or ROOT / "outputs" / args.domain / "lora_ema")
    out_dir = Path(args.out or ROOT / "outputs" / args.domain / "samples")
    out_dir.mkdir(parents=True, exist_ok=True)

    pipe = AutoPipelineForText2Image.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, variant="fp16", use_safetensors=True
    ).to(args.device)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.set_progress_bar_config(disable=True)

    base = generate(pipe, prompts, args.seed, args.resolution, args.inference_steps)
    for index, image in enumerate(base):
        image.save(out_dir / f"base_{index:02d}.png")

    run_config = json.loads((lora_dir.parent / "run_config.json").read_text())
    pipe.unet.add_adapter(
        LoraConfig(
            r=int(run_config["lora_r"]),
            lora_alpha=int(run_config["lora_alpha"]),
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        ),
        adapter_name="candidate",
    )
    state = load_file(str(lora_dir / "adapter.safetensors"), device=args.device)
    load_result = set_peft_model_state_dict(pipe.unet, state, adapter_name="candidate")
    if load_result.unexpected_keys:
        raise RuntimeError(f"unexpected LoRA keys: {load_result.unexpected_keys}")
    pipe.unet.set_adapters("candidate", weights=args.adapter_scale)
    candidate = generate(pipe, prompts, args.seed, args.resolution, args.inference_steps)
    for index, image in enumerate(candidate):
        image.save(out_dir / f"candidate_{index:02d}.png")

    make_grid(base, candidate, prompts).save(out_dir / f"{args.domain}_comparison.png")
    metadata = {
        "domain": args.domain,
        "seed_policy": f"seed={args.seed}+fixed_manifest_index",
        "selection_policy": "four portfolio-authored prompts frozen before generation; no output-based selection",
        "adapter_scale": args.adapter_scale,
        "manifest_rows": rows,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"saved fixed {args.domain} comparison to {out_dir}")


if __name__ == "__main__":
    main()
