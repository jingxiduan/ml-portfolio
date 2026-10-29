"""Generate first frames for I2V training using SDXL-Turbo (4-step, no classifier-free guidance).

Saves 512×320 PNG images matching the DynamiCrafter training resolution.
Uses batch processing for speed.
"""
import json, os, sys, time
from pathlib import Path

import torch
from diffusers import AutoPipelineForText2Image

TRAIN_PROMPTS = "data/prompts/tier2_train_200.json"
EVAL_PROMPTS  = "data/prompts/tier2_eval_40.json"
OUT_TRAIN     = "data/first_frames_200"
OUT_EVAL      = "data/first_frames_eval"
WIDTH, HEIGHT = 512, 320  # DynamiCrafter 512 resolution
BATCH_SIZE    = 8
NUM_STEPS     = 4
SEED          = 42

def generate_frames(prompts_file: str, out_dir: str, device: str = "cuda:0"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    prompts = json.loads(Path(prompts_file).read_text())
    
    # Check which frames already exist
    existing = set()
    for f in Path(out_dir).glob("*.png"):
        existing.add(f.stem)
    
    todo = [(i, p) for i, p in enumerate(prompts) if f"{i:03d}" not in existing]
    if not todo:
        print(f"All {len(prompts)} frames already exist in {out_dir}")
        return
    
    print(f"Generating {len(todo)}/{len(prompts)} frames → {out_dir}")
    
    # Load SDXL-Turbo (small, fast, 4-step)
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float16,
        variant="fp16",
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    
    gen = torch.Generator(device=device).manual_seed(SEED)
    
    t0 = time.time()
    for batch_start in range(0, len(todo), BATCH_SIZE):
        batch = todo[batch_start:batch_start + BATCH_SIZE]
        batch_prompts = [p["prompt"] for _, p in batch]
        batch_ids = [i for i, _ in batch]
        
        images = pipe(
            prompt=batch_prompts,
            num_inference_steps=NUM_STEPS,
            guidance_scale=0.0,  # Turbo doesn't need CFG
            width=WIDTH,
            height=HEIGHT,
            generator=gen,
        ).images
        
        for idx, img in zip(batch_ids, images):
            img.save(f"{out_dir}/{idx:03d}.png")
        
        elapsed = time.time() - t0
        done = min(batch_start + BATCH_SIZE, len(todo))
        eta = elapsed / done * (len(todo) - done)
        print(f"  [{done}/{len(todo)}] {elapsed:.0f}s elapsed, ETA {eta:.0f}s")
    
    print(f"Done: {len(todo)} frames in {time.time() - t0:.0f}s")
    
    # Cleanup GPU
    del pipe
    torch.cuda.empty_cache()


if __name__ == "__main__":
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
    generate_frames(TRAIN_PROMPTS, OUT_TRAIN, device)
    generate_frames(EVAL_PROMPTS, OUT_EVAL, device)
