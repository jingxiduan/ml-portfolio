"""
Quantitative evaluation: mean CLIPScore on held-out DrawBench prompts
(`data/eval_prompts.txt`, disjoint from the OPSD training prompts), base
SD-Turbo vs. the same model with the OPSD-trained LoRA adapter applied.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from diffusers import AutoPipelineForText2Image

from reward_model import ClipRewardModel

ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = "stabilityai/sd-turbo"


def load_prompts(path: Path) -> list[str]:
    return [l.strip() for l in path.read_text().splitlines() if l.strip()]


@torch.no_grad()
def eval_pipe(pipe, reward_model, prompts: list[str], seed: int, resolution: int) -> list[float]:
    scores = []
    for i, prompt in enumerate(prompts):
        gen = torch.Generator(device="cpu").manual_seed(seed + i)
        image = pipe(
            prompt=prompt, num_inference_steps=2, guidance_scale=0.0,
            height=resolution, width=resolution, generator=gen,
        ).images[0]
        import numpy as np
        img_t = torch.from_numpy(np.array(image)).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        img_t = img_t.to(pipe.device)
        text_feat = reward_model.encode_text([prompt])
        score = reward_model.score(img_t, text_feat).item()
        scores.append(score)
        print(f"  [{i+1}/{len(prompts)}] {score:.2f}  {prompt[:60]}")
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--prompts", default=str(ROOT / "data" / "eval_prompts.txt"))
    ap.add_argument("--lora-dir", default=str(ROOT / "outputs" / "lora_trainable"))
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "eval_results.json"))
    args = ap.parse_args()

    prompts = load_prompts(Path(args.prompts))
    print(f"[info] {len(prompts)} held-out eval prompts (disjoint from training prompts)")

    reward_model = ClipRewardModel(device=args.device)

    print("[info] evaluating BASE model...")
    pipe = AutoPipelineForText2Image.from_pretrained(MODEL_ID, torch_dtype=torch.float32).to(args.device)
    pipe.set_progress_bar_config(disable=True)
    base_scores = eval_pipe(pipe, reward_model, prompts, args.seed, args.resolution)

    print(f"[info] loading OPSD LoRA adapter from {args.lora_dir} ...")
    lora_file = str(Path(args.lora_dir) / "pytorch_lora_weights.safetensors")
    pipe.unet.load_lora_adapter(lora_file, prefix=None, adapter_name="trained")
    pipe.unet.set_adapter("trained")

    print("[info] evaluating OPSD-trained model...")
    trained_scores = eval_pipe(pipe, reward_model, prompts, args.seed, args.resolution)

    base_mean = sum(base_scores) / len(base_scores)
    trained_mean = sum(trained_scores) / len(trained_scores)
    win_rate = sum(1 for b, t in zip(base_scores, trained_scores) if t > b) / len(base_scores)

    result = {
        "n_prompts": len(prompts),
        "base_clipscore_mean": base_mean,
        "trained_clipscore_mean": trained_mean,
        "delta": trained_mean - base_mean,
        "win_rate_trained_over_base": win_rate,
        "base_scores": base_scores,
        "trained_scores": trained_scores,
    }
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"\n[result] base CLIPScore mean  = {base_mean:.3f}")
    print(f"[result] trained CLIPScore mean = {trained_mean:.3f}")
    print(f"[result] delta = {result['delta']:+.3f}  win_rate(trained>base) = {win_rate:.1%}")
    print(f"[ok] saved {args.out}")


if __name__ == "__main__":
    main()
