"""Render videos with frozen DynamiCrafter from a directory of first frames.

Expected input layout (produced by src/t2i/generate_first_frames.py):
    frames_dir/
      seed_00/000.png, 001.png, ...
      seed_01/...
      manifest.json

Output layout:
    out_dir/
      seed_00/000.mp4, 001.mp4, ...
      seed_01/...
      manifest.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import torch
from PIL import Image
from torchvision import transforms
from torchvision.io import write_video

from .dynamicrafter_wrapper import I2VConfig, build_opsd_policies
from .utils.mps_patch import apply_mps_patches, get_device


def _load_scheduler(num_steps: int, device):
    """Stand-in scheduler. Replace with DynamiCrafter's own sampler after vendoring."""
    from diffusers import DDIMScheduler
    sched = DDIMScheduler.from_pretrained(
        "stabilityai/stable-diffusion-2-1", subfolder="scheduler"
    )
    sched.set_timesteps(num_steps, device=device)
    return sched


@torch.no_grad()
def render_one(
    policies,
    scheduler,
    first_frame_px: torch.Tensor,       # [1, 3, H, W] in [-1,1]
    prompt: str,
    num_frames: int = 16,
    fps: int = 8,
    seed: int = 0,
) -> torch.Tensor:
    """Return a video tensor [T, 3, H, W] uint8."""
    device = policies.device
    torch.manual_seed(seed)
    # VAE-encode first frame as conditioning
    x0_lat = policies.vae.encode(first_frame_px.to(device)).latent_dist.sample() * 0.18215
    cond = {
        "first_frame_latent": x0_lat,
        "text": prompt,
        "video_length": num_frames,
        "fps": fps,
    }
    B = 1
    T = num_frames
    C = x0_lat.shape[1]
    H_, W_ = x0_lat.shape[-2], x0_lat.shape[-1]
    z = torch.randn(B, C, T, H_, W_, device=device, dtype=x0_lat.dtype)
    for t in scheduler.timesteps:
        pred = policies.v_theta(z, t, cond=cond)
        z = scheduler.step(pred, t, z).prev_sample
    vid = policies.vae.decode(z / 0.18215).sample            # [B,3,T,H,W] or [B,T,3,H,W]
    if vid.dim() == 5 and vid.shape[1] == 3:
        vid = vid.permute(0, 2, 1, 3, 4)
    vid = (vid.clamp(-1, 1) + 1.0) * 127.5
    return vid[0].to(torch.uint8).cpu()                      # [T, 3, H, W]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_dir", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    apply_mps_patches()
    device = get_device()

    model_cfg = I2VConfig(num_frames=args.num_frames, fps=args.fps)
    policies = build_opsd_policies(model_cfg, device=device)
    # Tier 1: no LoRA, v_theta == v_old. Both frozen.
    for p in policies.v_theta.parameters():
        p.requires_grad_(False)
    policies.v_theta.eval()
    scheduler = _load_scheduler(model_cfg.num_inference_steps, device)

    with open(args.prompts) as f:
        prompts: List[dict] = json.load(f)
    if args.limit:
        prompts = prompts[: args.limit]

    H, W = model_cfg.resolution
    tf = transforms.Compose([
        transforms.Resize((H, W), antialias=True),
        transforms.CenterCrop((H, W)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    frames_root = Path(args.frames_dir)
    out_root = Path(args.out_dir)
    manifest = []

    for seed in args.seeds:
        in_dir = frames_root / f"seed_{seed:02d}"
        out_dir = out_root / f"seed_{seed:02d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, item in enumerate(prompts):
            frame_path = in_dir / f"{i:03d}.png"
            if not frame_path.is_file():
                print(f"[i2v] WARNING: missing frame {frame_path}; skip.")
                continue
            vid_path = out_dir / f"{i:03d}.mp4"
            if vid_path.is_file():
                manifest.append({
                    "index": i, "prompt": item["prompt"], "seed": seed,
                    "file": str(vid_path.relative_to(out_root)),
                })
                continue
            img = Image.open(frame_path).convert("RGB")
            x0 = tf(img).unsqueeze(0)
            vid = render_one(policies, scheduler, x0, item["prompt"],
                             num_frames=args.num_frames, fps=args.fps, seed=seed)
            write_video(str(vid_path), vid.permute(0, 2, 3, 1), fps=args.fps)
            manifest.append({
                "index": i, "prompt": item["prompt"], "seed": seed,
                "file": str(vid_path.relative_to(out_root)),
            })
            if (i + 1) % 10 == 0:
                print(f"[i2v] seed={seed} {i+1}/{len(prompts)}")

    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[i2v] done. {len(manifest)} videos in {out_root}.")


if __name__ == "__main__":
    main()
