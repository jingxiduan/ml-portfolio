"""DynamiCrafter wrapper: load, LoRA inject, build frozen behavior policy.

This module is deliberately thin — it wraps the vendored DynamiCrafter checkpoint and provides
the three objects OPSD training needs:

    (1) `v_theta` — a trainable LoRA'd copy of the DiT (temporal blocks only).
    (2) `v_old`   — a frozen copy, identical init, used for on-policy rollouts.
    (3) `vae`     — the VAE decoder, used to decode latents into pixel space for reward eval.
                     Kept in fp32 on MPS to avoid NaNs; fp16 on CUDA.

DynamiCrafter is loaded from the local vendored checkout at ``third_party/DynamiCrafter``.
Run ``scripts/02_download_weights.sh`` first.
"""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .utils.lora_utils import (
    count_total_params,
    count_trainable_params,
    inject_temporal_lora,
)
from .utils.mps_patch import get_device, recommended_dtype


@dataclass
class I2VConfig:
    """DynamiCrafter-320 defaults (res 320×512, 16 frames @ 8 fps)."""

    ckpt_path: str = "weights/dynamicrafter_320/model.ckpt"
    resolution: Tuple[int, int] = (320, 512)
    num_frames: int = 16
    fps: int = 8
    guidance_scale: float = 7.5
    num_inference_steps: int = 50
    lora_rank: int = 16
    lora_alpha: int = 32
    dtype_override: Optional[str] = None  # "fp16" / "fp32" / None (auto)

    def dtype(self, device: torch.device) -> torch.dtype:
        if self.dtype_override == "fp16":
            return torch.float16
        if self.dtype_override == "fp32":
            return torch.float32
        return recommended_dtype(device)


# ---------------------------------------------------------------------- loader


def _import_dynamicrafter():
    """Import DynamiCrafter's config/model loader from the vendored checkout."""
    try:
        from omegaconf import OmegaConf  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise ImportError("omegaconf is required (`pip install omegaconf`)") from e
    try:
        # path is set up by scripts/02_download_weights.sh which clones the repo under third_party/
        import sys

        third_party = Path(__file__).resolve().parents[1] / "third_party" / "DynamiCrafter"
        if third_party.is_dir() and str(third_party) not in sys.path:
            sys.path.insert(0, str(third_party))
        from utils.utils import instantiate_from_config  # type: ignore
        from lvdm.models.ddpm3d import LatentVisualDiffusion  # type: ignore
        return instantiate_from_config, LatentVisualDiffusion
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "DynamiCrafter not vendored. Run `bash scripts/02_download_weights.sh` to clone it."
        ) from e


def load_base_model(cfg: I2VConfig, device: torch.device) -> nn.Module:
    """Load the pretrained DynamiCrafter model in eval mode on ``device``."""
    instantiate_from_config, _ = _import_dynamicrafter()
    from omegaconf import OmegaConf

    cfg_path = (
        Path(__file__).resolve().parents[1]
        / "third_party"
        / "DynamiCrafter"
        / "configs"
        / "inference_320_v1.0.yaml"
    )
    if not cfg_path.is_file():
        raise FileNotFoundError(f"DynamiCrafter config not found at {cfg_path}")
    config = OmegaConf.load(cfg_path)
    model = instantiate_from_config(config.model)

    ckpt_path = Path(cfg.ckpt_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found at {ckpt_path}. Run scripts/02_download_weights.sh."
        )
    state = torch.load(ckpt_path, map_location="cpu")
    state = state.get("state_dict", state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[DynamiCrafter] loaded, missing={len(missing)}, unexpected={len(unexpected)}")

    model.to(device=device, dtype=cfg.dtype(device))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# ---------------------------------------------------------------------- policies


@dataclass
class OPSDPolicies:
    """Container for the pair (v_theta, v_old) + the shared VAE."""

    v_theta: nn.Module  # trainable (LoRA'd)
    v_old: nn.Module    # frozen behavior copy
    vae: nn.Module      # VAE (shared, frozen)
    device: torch.device
    cfg: I2VConfig

    # ---- stats
    def summary(self) -> str:
        n_train = count_trainable_params(self.v_theta)
        n_total = count_total_params(self.v_theta)
        return (
            f"v_theta: trainable={n_train/1e6:.2f}M / total={n_total/1e6:.2f}M "
            f"({100.0 * n_train / max(n_total, 1):.2f}%)"
        )


def build_opsd_policies(cfg: I2VConfig, device: Optional[torch.device] = None) -> OPSDPolicies:
    """Build the (v_theta, v_old, vae) triple for OPSD-I2V training."""
    device = device or get_device()

    base = load_base_model(cfg, device)

    # The VAE lives on `base.first_stage_model` in DynamiCrafter's LatentVisualDiffusion.
    vae = getattr(base, "first_stage_model", None) or getattr(base, "vae", None)
    if vae is None:
        raise RuntimeError("Could not locate VAE on DynamiCrafter base model.")

    # The denoiser DiT is on `base.model.diffusion_model` (ldm convention).
    denoiser = base.model.diffusion_model

    # v_theta: inject LoRA
    v_theta = inject_temporal_lora(denoiser, rank=cfg.lora_rank, alpha=cfg.lora_alpha)

    # v_old: deep-copy of the same denoiser (no LoRA — we EMA LoRA weights only from v_theta
    #        into a frozen clone that mirrors the LoRA structure).
    v_old = copy.deepcopy(v_theta)
    for p in v_old.parameters():
        p.requires_grad_(False)
    v_old.eval()

    policies = OPSDPolicies(
        v_theta=v_theta,
        v_old=v_old,
        vae=vae,
        device=device,
        cfg=cfg,
    )
    print("[OPSDPolicies]", policies.summary())
    return policies


# ---------------------------------------------------------------------- rollout


@torch.no_grad()
def rollout_until(
    v: nn.Module,
    first_frame_latent: torch.Tensor,     # [B, C, H', W']
    cond: dict,                            # DynamiCrafter conditioning dict
    sigma_q: torch.Tensor,                 # [B] stop-noise level
    scheduler,                             # the diffusion scheduler (DDIM/EulerFlow)
) -> torch.Tensor:
    """Partial rollout: run reverse diffusion until noise level ``sigma_q``.

    Returns the intermediate latent ``z_q``. The caller is responsible for providing
    ``scheduler`` already configured with the correct timestep schedule.

    This is `step 1 — on-policy query` of the OPSD-I2V loop.
    """
    device, dtype = first_frame_latent.device, first_frame_latent.dtype
    B = first_frame_latent.shape[0]

    # Initialize z_T from pure Gaussian noise, same shape as a video latent clip.
    # DynamiCrafter latent shape: [B, C, T, H', W']
    T = cond.get("video_length", 16)
    C, H_, W_ = first_frame_latent.shape[1], first_frame_latent.shape[-2], first_frame_latent.shape[-1]
    z = torch.randn(B, C, T, H_, W_, device=device, dtype=dtype)

    # Concatenate first-frame latent in the time axis (DynamiCrafter image-conditioning scheme).
    # The exact concat convention is handled inside DynamiCrafter's forward; we pass the cond dict.
    for t in scheduler.timesteps:
        if t < sigma_q.max():
            break  # stop at the intermediate step we care about
        noise_pred = v(z, t, cond=cond)
        z = scheduler.step(noise_pred, t, z).prev_sample
    return z
