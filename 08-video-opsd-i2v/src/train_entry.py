"""Unified entry point: pick trainer by --method.

Usage:
    python -m src.train_entry --method opsd_i2v --config configs/opsd_i2v_default.yaml
    python -m src.train_entry --method instructvideo --config configs/baseline_instructvideo.yaml
    python -m src.train_entry --method dpo_video --config configs/baseline_dpovideo.yaml
"""
from __future__ import annotations

import argparse
import logging

import yaml

from .baselines.dpo_video_trainer import DPOVideoConfig, DPOVideoTrainer
from .baselines.instructvideo_trainer import InstructVideoConfig, InstructVideoTrainer
from .data.i2v_dataset import I2VDataset
from .dynamicrafter_wrapper import I2VConfig, build_opsd_policies
from .opsd_trainer_i2v import OPSDI2VTrainer, OPSDTrainConfig
from .rewards.combined import CombinedReward
from .utils.mps_patch import apply_mps_patches, get_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def _make_scheduler(num_steps: int, device):
    """DynamiCrafter uses a DDIM-like linear-beta sampler. Create locally
    to avoid requiring HuggingFace downloads."""
    from diffusers import DDIMScheduler
    sched = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
    )
    sched.set_timesteps(num_steps, device=device)
    return sched


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["opsd_i2v", "instructvideo", "dpo_video"], required=True)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    apply_mps_patches()
    device = get_device()

    model_cfg = I2VConfig(**cfg["model"])
    policies = build_opsd_policies(model_cfg, device=device)

    reward_fn = CombinedReward(cfg["reward"]["components"], device=str(device))

    dataset = I2VDataset(
        first_frame_dir=cfg["train"]["first_frame_dir"],
        prompts_json=cfg["train"]["train_prompts"],
        resolution=model_cfg.resolution,
        num_frames=model_cfg.num_frames,
        fps=model_cfg.fps,
    )

    scheduler = _make_scheduler(model_cfg.num_inference_steps, device)

    if args.method == "opsd_i2v":
        train_cfg = OPSDTrainConfig(**cfg["train"])
        trainer = OPSDI2VTrainer(train_cfg, model_cfg, policies, reward_fn, scheduler, dataset)
    elif args.method == "instructvideo":
        train_cfg = InstructVideoConfig(**cfg["train"])
        trainer = InstructVideoTrainer(train_cfg, model_cfg, policies, reward_fn, scheduler, dataset)
    elif args.method == "dpo_video":
        train_cfg = DPOVideoConfig(**cfg["train"])
        trainer = DPOVideoTrainer(train_cfg, model_cfg, policies, reward_fn, scheduler, dataset)
    else:
        raise ValueError(f"Unknown method {args.method}")

    trainer.train()


if __name__ == "__main__":
    main()
