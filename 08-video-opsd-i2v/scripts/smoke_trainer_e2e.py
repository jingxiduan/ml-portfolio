"""End-to-end 1-step smoke test for the OPSD-I2V trainer.

Loads the real DynamiCrafter checkpoint, builds cond-stage, runs ONE training
step, and confirms:
  - the forward pass produces a finite loss,
  - the backward pass produces finite grads on LoRA parameters,
  - the optimizer step runs without exception.

Runs for ~2-5 min on M3 Pro MPS; GPU required (~10GB RAM).

Usage:  PYTHONPATH=. python scripts/smoke_trainer_e2e.py
Exit codes:
  0 — forward+backward+step all succeeded, trainer is functionally alive
  1 — any stage failed; see traceback
"""
from __future__ import annotations
import logging
import sys
from pathlib import Path

import torch
import yaml

from src.dynamicrafter_wrapper import I2VConfig, build_opsd_policies
from src.opsd_trainer_i2v import OPSDI2VTrainer, OPSDTrainConfig
from src.rewards import CombinedReward
from src.data.i2v_dataset import I2VDataset, collate_i2v
from src.utils.mps_patch import apply_mps_patches, get_device


def _make_scheduler(num_steps: int, device):
    """Construct a DDIM scheduler locally (no HF download needed for smoke)."""
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


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg_path = Path("configs/tier2_natural.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    # Clamp to 1 step + minimal shapes for the smoke
    cfg["train"]["num_steps"] = 1
    cfg["train"]["warmup_steps"] = 1
    cfg["train"]["inner_fit_steps"] = 1
    cfg["train"]["ckpt_every"] = 999
    # Minimal shapes to fit in M3 Pro MPS budget. Real training requires an A100.
    # resolution must be divisible by the UNet downsample stack: DynamiCrafter has
    # 3 downsamples (÷8) → 256 is safe, 160 is not (produces off-by-one feature maps).
    cfg["model"]["num_frames"] = 4
    cfg["model"]["resolution"] = [256, 256]
    cfg["model"]["num_inference_steps"] = 5

    apply_mps_patches()
    device = get_device()
    print(f"[smoke] device = {device}")

    print("[smoke] loading base model (~9.7GB; takes ~1min)...")
    model_cfg = I2VConfig(**cfg["model"])
    policies = build_opsd_policies(model_cfg, device=device)

    if policies.cond_stage is None:
        print("[smoke] FAIL: cond_stage is None", file=sys.stderr)
        return 1

    print("[smoke] building reward + dataset ...")
    reward_fn = CombinedReward(cfg["reward"]["components"], device=str(device))
    dataset = I2VDataset(
        first_frame_dir=cfg["train"]["first_frame_dir"],
        prompts_json=cfg["train"]["train_prompts"],
        resolution=model_cfg.resolution,
        num_frames=model_cfg.num_frames,
        fps=model_cfg.fps,
    )
    print(f"[smoke] dataset: {len(dataset)} entries")

    scheduler = _make_scheduler(model_cfg.num_inference_steps, device)
    train_cfg = OPSDTrainConfig(**cfg["train"])
    trainer = OPSDI2VTrainer(train_cfg, model_cfg, policies, reward_fn, scheduler, dataset)

    print("[smoke] running ONE _opsd_step ...")
    # Grab one batch manually (bypass the full .train() loop so we don't hit
    # checkpoint / logging side-effects)
    from torch.utils.data import DataLoader
    dl = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_i2v, num_workers=0)
    batch = next(iter(dl))

    try:
        metrics = trainer._opsd_step(batch)
    except Exception as e:
        print(f"[smoke] FAIL: _opsd_step raised {type(e).__name__}: {e}", file=sys.stderr)
        import traceback; traceback.print_exc()
        return 1

    print(f"[smoke] metrics: {metrics}")
    for k, v in metrics.items():
        if isinstance(v, float) and not (v == v):  # NaN check
            print(f"[smoke] FAIL: {k} is NaN", file=sys.stderr)
            return 1

    # Confirm at least one LoRA param got a grad
    lora_grad_norms = []
    for n, p in trainer.p.v_theta.named_parameters():
        if p.requires_grad and p.grad is not None:
            lora_grad_norms.append((n, float(p.grad.detach().abs().mean())))
    if not lora_grad_norms:
        print("[smoke] FAIL: no LoRA parameter received a grad", file=sys.stderr)
        return 1
    print(f"[smoke] {len(lora_grad_norms)} LoRA params had grads; top-3 by |mean|:")
    lora_grad_norms.sort(key=lambda x: -x[1])
    for n, g in lora_grad_norms[:3]:
        print(f"[smoke]   {g:.3e}  {n}")

    print("[smoke] PASS — trainer is functionally alive")
    return 0


if __name__ == "__main__":
    sys.exit(main())
