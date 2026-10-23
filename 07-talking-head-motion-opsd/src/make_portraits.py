"""Synthesize fictional, photoreal 1024x1024 portraits with SDXL (no real people, no likeness issues)."""
import argparse
import os

import torch
from diffusers import AutoencoderKL, StableDiffusionXLPipeline

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PORTRAITS = [
    ("host_f", 11, "professional studio headshot photo of a 30 year old east asian woman news presenter, "
                   "looking directly at camera, neutral relaxed expression, mouth closed, shoulder-length black hair, "
                   "navy blazer, soft key light, plain light gray backdrop, 85mm lens, sharp focus, high detail skin texture"),
    ("host_m", 23, "professional studio headshot photo of a 40 year old man corporate trainer, looking directly at "
                   "camera, calm friendly expression, mouth closed, short brown hair, light stubble, white shirt and "
                   "gray suit jacket, soft key light, plain warm beige backdrop, 85mm lens, sharp focus, detailed skin"),
    ("host_f2", 37, "professional studio headshot photo of a 25 year old woman e-commerce livestream host with curly "
                    "dark hair, looking directly at camera, gentle expression, mouth closed, casual cream sweater, "
                    "soft ring light, plain pastel blue backdrop, 85mm lens, sharp focus, detailed skin texture"),
]
NEG = ("cartoon, illustration, painting, 3d render, blurry, deformed, extra fingers, hands, open mouth, teeth, "
       "sunglasses, hat, text, watermark, logo, profile view, looking away")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "portraits"))
    ap.add_argument("--steps", type=int, default=30)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", vae=vae, torch_dtype=torch.float16, variant="fp16",
        use_safetensors=True).to("mps")
    pipe.set_progress_bar_config(disable=True)
    for name, seed, prompt in PORTRAITS:
        g = torch.Generator("cpu").manual_seed(seed)
        img = pipe(prompt=prompt, negative_prompt=NEG, width=1024, height=1024, num_inference_steps=a.steps,
                   guidance_scale=6.0, generator=g).images[0]
        img.save(os.path.join(a.out, f"{name}.png"))
        print("saved", name, flush=True)


if __name__ == "__main__":
    main()
