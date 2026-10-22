"""
Quantitative evaluation: mean video reward (frame-averaged CLIPScore +
temporal consistency, see reward_model.py) on held-out prompts
(`data/eval_prompts.txt`, disjoint from the OPSD training prompts), base
AnimateDiff vs. the same pipeline with the OPSD-trained LoRA adapter applied.
Mirrors `04-diffusion-opsd-mps/src/evaluate.py`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler

from reward_model import ClipVideoRewardModel

ROOT = Path(__file__).resolve().parent.parent
BASE_MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
MOTION_ADAPTER_ID = "guoyww/animatediff-motion-adapter-v1-5-2"


def load_prompts(path: Path) -> list[str]:
    return [l.strip() for l in path.read_text().splitlines() if l.strip()]


def frames_to_tensor(frames, device) -> torch.Tensor:
    """list[PIL.Image] -> (1, n_frames, 3, H, W) float tensor in [0, 1]."""
    arrs = [np.array(f).astype(np.float32) / 255.0 for f in frames]
    t = torch.from_numpy(np.stack(arrs)).permute(0, 3, 1, 2).unsqueeze(0)
    return t.to(device)


@torch.no_grad()
def eval_pipe(pipe, reward_model, prompts: list[str], seed: int, resolution: int,
              num_frames: int, steps: int):
    results = []
    for i, prompt in enumerate(prompts):
        gen = torch.Generator(device="cpu").manual_seed(seed + i)
        out = pipe(
            prompt=prompt, num_frames=num_frames, height=resolution, width=resolution,
            num_inference_steps=steps, guidance_scale=7.5, generator=gen,
        )
        frames_t = frames_to_tensor(out.frames[0], pipe.device)
        text_feat = reward_model.encode_text([prompt])
        reward, align, consistency = reward_model.score_video(frames_t, text_feat)
        rec = {"reward": reward.item(), "align": align.item(), "consistency": consistency.item()}
        results.append(rec)
        print(f"  [{i+1}/{len(prompts)}] reward={rec['reward']:.2f} align={rec['align']:.2f} "
              f"cons={rec['consistency']:.2f}  {prompt[:50]}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--prompts", default=str(ROOT / "data" / "eval_prompts.txt"))
    ap.add_argument("--lora-dir", default=str(ROOT / "outputs" / "lora_trainable"))
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--num-frames", type=int, default=8)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "eval_results.json"))
    args = ap.parse_args()

    prompts = load_prompts(Path(args.prompts))
    print(f"[info] {len(prompts)} held-out eval prompts (disjoint from training prompts)")

    reward_model = ClipVideoRewardModel(device=args.device)

    print("[info] loading base AnimateDiff pipeline...")
    adapter = MotionAdapter.from_pretrained(MOTION_ADAPTER_ID, torch_dtype=torch.float32)
    pipe = AnimateDiffPipeline.from_pretrained(BASE_MODEL_ID, motion_adapter=adapter, torch_dtype=torch.float32)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(args.device)
    pipe.set_progress_bar_config(disable=True)

    print("[info] evaluating BASE model...")
    base_results = eval_pipe(pipe, reward_model, prompts, args.seed, args.resolution, args.num_frames, args.steps)

    print(f"[info] loading OPSD LoRA adapter from {args.lora_dir} ...")
    lora_file = str(Path(args.lora_dir) / "pytorch_lora_weights.safetensors")
    pipe.unet.load_lora_adapter(lora_file, prefix=None, adapter_name="trained")
    pipe.unet.set_adapter("trained")

    print("[info] evaluating OPSD-trained model...")
    trained_results = eval_pipe(pipe, reward_model, prompts, args.seed, args.resolution, args.num_frames, args.steps)

    base_rewards = [r["reward"] for r in base_results]
    trained_rewards = [r["reward"] for r in trained_results]
    base_mean = sum(base_rewards) / len(base_rewards)
    trained_mean = sum(trained_rewards) / len(trained_rewards)
    win_rate = sum(1 for b, t in zip(base_rewards, trained_rewards) if t > b) / len(base_rewards)

    base_align_mean = sum(r["align"] for r in base_results) / len(base_results)
    trained_align_mean = sum(r["align"] for r in trained_results) / len(trained_results)
    base_cons_mean = sum(r["consistency"] for r in base_results) / len(base_results)
    trained_cons_mean = sum(r["consistency"] for r in trained_results) / len(trained_results)

    result = {
        "n_prompts": len(prompts),
        "base_reward_mean": base_mean,
        "trained_reward_mean": trained_mean,
        "delta": trained_mean - base_mean,
        "win_rate_trained_over_base": win_rate,
        "base_align_mean": base_align_mean,
        "trained_align_mean": trained_align_mean,
        "base_consistency_mean": base_cons_mean,
        "trained_consistency_mean": trained_cons_mean,
        "base_results": base_results,
        "trained_results": trained_results,
    }
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"\n[result] base reward mean    = {base_mean:.3f}  (align={base_align_mean:.2f} cons={base_cons_mean:.2f})")
    print(f"[result] trained reward mean = {trained_mean:.3f}  (align={trained_align_mean:.2f} cons={trained_cons_mean:.2f})")
    print(f"[result] delta = {result['delta']:+.3f}  win_rate(trained>base) = {win_rate:.1%}")
    print(f"[ok] saved {args.out}")


if __name__ == "__main__":
    main()
