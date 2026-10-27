"""VBench-I2V evaluation wrapper.

VBench expects a directory of generated videos with a specific naming convention. This script:
  1. Loads a trained LoRA checkpoint into v_theta.
  2. Runs inference on the held-out eval prompt split.
  3. Dumps videos to a per-method folder.
  4. Invokes the official VBench-I2V evaluator (via subprocess) to compute all 9 dimensions.
  5. Collects results into a single CSV.

Run:
    python -m src.eval.vbench_i2v_eval \
        --ckpt outputs/opsd_i2v_default/lora_step00500 \
        --tag opsd_i2v \
        --out_dir outputs/vbench_eval/opsd_i2v
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List

import torch
from torchvision.io import write_video

from ..data.i2v_dataset import I2VDataset
from ..dynamicrafter_wrapper import I2VConfig, build_opsd_policies
from ..utils.mps_patch import apply_mps_patches, get_device


VBENCH_I2V_DIMS: List[str] = [
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
    "i2v_subject",
    "i2v_background",
    "camera_motion",
]


@torch.no_grad()
def generate_video(
    policies, scheduler, cond: dict, resolution=(320, 512), num_frames: int = 16
) -> torch.Tensor:
    """Return a video tensor [T, 3, H, W] uint8 for VBench."""
    B = cond["first_frame_latent"].shape[0]
    T = num_frames
    C = cond["first_frame_latent"].shape[1]
    H_, W_ = cond["first_frame_latent"].shape[-2], cond["first_frame_latent"].shape[-1]
    z = torch.randn(B, C, T, H_, W_, device=policies.device,
                    dtype=cond["first_frame_latent"].dtype)
    for t in scheduler.timesteps:
        pred = policies.v_theta(z, t, cond=cond)
        z = scheduler.step(pred, t, z).prev_sample
    vid = policies.vae.decode(z / 0.18215).sample      # [B, 3, T, H, W] or [B, T, 3, H, W]
    if vid.dim() == 5 and vid.shape[1] == 3:
        vid = vid.permute(0, 2, 1, 3, 4)
    vid = (vid.clamp(-1, 1) + 1) * 127.5
    return vid[0].to(torch.uint8).cpu()                # [T, 3, H, W]


def dump_videos(ckpt: str, out_dir: str, prompts_json: str, first_frame_dir: str) -> Path:
    apply_mps_patches()
    device = get_device()

    model_cfg = I2VConfig()
    policies = build_opsd_policies(model_cfg, device=device)

    # Load LoRA
    ckpt_p = Path(ckpt)
    try:
        from peft import PeftModel  # type: ignore
        PeftModel.from_pretrained(policies.v_theta, str(ckpt_p))
    except Exception:
        lora_pt = ckpt_p / "lora.pt"
        if lora_pt.is_file():
            sd = torch.load(lora_pt, map_location="cpu")
            missing, unexpected = policies.v_theta.load_state_dict(sd, strict=False)
            print(f"[eval] raw LoRA load: missing={len(missing)}, unexpected={len(unexpected)}")
        else:
            print(f"[eval] WARNING: no checkpoint found at {ckpt_p}; evaluating base model.")

    policies.v_theta.eval()

    # Set up scheduler (we re-use the default DynamiCrafter scheduler).
    scheduler = None
    try:
        from diffusers import DDIMScheduler
        scheduler = DDIMScheduler.from_pretrained(
            "stabilityai/stable-diffusion-2-1", subfolder="scheduler"
        )
        scheduler.set_timesteps(model_cfg.num_inference_steps, device=device)
    except Exception as e:
        print(f"[eval] WARNING: could not load DDIMScheduler: {e}")
        print("[eval] you likely need to attach DynamiCrafter's own sampler here.")

    ds = I2VDataset(first_frame_dir, prompts_json, resolution=model_cfg.resolution,
                    num_frames=model_cfg.num_frames)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest = []
    for i in range(len(ds)):
        item = ds[i]
        cond = item["cond"]
        cond["first_frame_latent"] = (
            policies.vae.encode(item["first_frame"].unsqueeze(0).to(device)).latent_dist.sample()
            * 0.18215
        )
        vid = generate_video(policies, scheduler, cond,
                             resolution=model_cfg.resolution,
                             num_frames=model_cfg.num_frames)
        vid_path = out / f"{i:03d}.mp4"
        write_video(str(vid_path), vid.permute(0, 2, 3, 1), fps=model_cfg.fps)
        manifest.append({"index": i, "prompt": item["prompt"], "file": vid_path.name})
        if (i + 1) % 10 == 0:
            print(f"[eval] generated {i+1}/{len(ds)}")

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return out


def run_vbench(video_dir: Path, out_dir: Path, dims: List[str] | None = None) -> dict:
    """Invoke the official VBench evaluator. VBench must be installed from source."""
    dims = dims or VBENCH_I2V_DIMS
    cmd = [
        sys.executable, "-m", "vbench.evaluate",
        "--videos_path", str(video_dir),
        "--output_path", str(out_dir / "vbench"),
        "--mode", "custom_input",
        "--dimension", *dims,
    ]
    print("[vbench] running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    results_path = out_dir / "vbench" / "eval_results.json"
    if not results_path.is_file():
        raise RuntimeError(f"VBench did not produce {results_path}")
    return json.loads(results_path.read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to LoRA checkpoint dir")
    ap.add_argument("--tag", required=True, help="name used for the output folder")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--prompts", default="data/vbench_i2v_prompts/eval_split.json")
    ap.add_argument("--first_frames", default="data/first_frames")
    ap.add_argument("--skip_generation", action="store_true")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    video_dir = out / "videos"
    if not args.skip_generation:
        dump_videos(args.ckpt, str(video_dir), args.prompts, args.first_frames)

    results = run_vbench(video_dir, out)
    print(json.dumps(results, indent=2))
    (out / "summary.json").write_text(json.dumps({"tag": args.tag, "results": results}, indent=2))


if __name__ == "__main__":
    main()
