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

    ckpt_path: str = "weights/dynamicrafter_512/model.ckpt"
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
        / "inference_512_v1.0.yaml"
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
class CondStage:
    """DynamiCrafter conditioning-stage bundle.

    Three frozen modules that convert ``(prompt_str, first_frame_px) → cond dict``
    in the exact format DynamiCrafter's ``DiffusionWrapper`` expects. Lifted off
    the base model so the trainer never has to touch the LatentVisualDiffusion
    object itself.

    Forward signatures (verified against DynamiCrafter source):
        cond_stage_model.encode(list[str])  → [B, 77, 1024]      (CLIP text)
        embedder(img_bchw)                  → [B, L, 1280]       (OpenCLIP image)
        image_proj_model([B,L,1280])        → [B, L', 1024]      (resampler to text dim)

    Final cond dict (hybrid conditioning):
        {
            "c_crossattn": [cat([text_emb, img_emb], dim=1)],    # [B, 77+L', 1024]
            "c_concat":    [img_cat_cond_bcthw],                 # [B, 4, T, H', W']
            "fs":          fs_long_tensor,                        # [B]
        }
    """
    cond_stage_model: nn.Module  # CLIP text encoder (frozen)
    embedder:         nn.Module  # CLIP image encoder (frozen)
    image_proj_model: nn.Module  # resampler (frozen)
    scale_factor:     float = 0.18215
    temporal_length:  int   = 16
    default_fs:       int   = 8  # frame stride; DynamiCrafter uses this as "fps" conditioning


def build_cond(
    cs: "CondStage",
    prompts: list[str],
    first_frame_latent: torch.Tensor,   # [B, C, H', W']  (VAE-encoded first frame)
    first_frame_px:     torch.Tensor,   # [B, 3, H, W]    in [-1, 1]
    *,
    T: Optional[int] = None,
    fs: Optional[torch.Tensor] = None,
) -> dict:
    """Build the DynamiCrafter cond dict for a batch, in a single no-grad pass.

    Mirrors ``LatentVisualDiffusion.get_batch_input`` (ddpm3d.py:1052-1108)
    with ``random_uncond=False`` and ``cond_frame_index=0`` — i.e. deterministic
    first-frame conditioning, no classifier-free dropout at train time (OPSD
    does its own trust-region exploration via c_adv).

    Returns a dict ready for ``_call_unet(unet, z, t, cond)``.
    """
    T = T or cs.temporal_length
    B, C, Hp, Wp = first_frame_latent.shape
    device, dtype = first_frame_latent.device, first_frame_latent.dtype

    with torch.no_grad():
        # 1) text branch
        text_emb = cs.cond_stage_model.encode(prompts)              # [B, 77, 1024]
        if text_emb.dtype != dtype:
            text_emb = text_emb.to(dtype)

        # 2) image branch — cast to embedder's dtype (fp16 on CUDA)
        emb_dtype = next(cs.embedder.parameters()).dtype
        img_emb = cs.embedder(first_frame_px.to(dtype=emb_dtype))    # [B, L, 1280]
        img_emb = cs.image_proj_model(img_emb)                       # [B, L', 1024]
        if img_emb.dtype != dtype:
            img_emb = img_emb.to(dtype)

        # 3) hybrid concat conditioning: repeat first-frame latent T times along time
        # Shape: [B, C, 1, H', W'] → [B, C, T, H', W']
        img_cat_cond = first_frame_latent.unsqueeze(2).expand(B, C, T, Hp, Wp).contiguous()

    # 4) frame-stride conditioning (DynamiCrafter uses `fs` as the fps proxy)
    if fs is None:
        fs = torch.full((B,), int(cs.default_fs), device=device, dtype=torch.long)

    return {
        "c_crossattn": [torch.cat([text_emb, img_emb], dim=1)],  # [B, 77+L', 1024]
        "c_concat":    [img_cat_cond],                           # [B, C, T, H', W']
        "fs":          fs,
    }


@dataclass
class OPSDPolicies:
    """Container for the pair (v_theta, v_old) + the shared VAE + cond-stage."""

    v_theta: nn.Module  # trainable (LoRA'd)
    v_old: nn.Module    # frozen behavior copy
    vae: nn.Module      # VAE (shared, frozen)
    device: torch.device
    cfg: I2VConfig
    cond_stage: Optional[CondStage] = None  # text+image embedders + proj (frozen)

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

    # Lift cond-stage off base (frozen). All three attributes are required for
    # DynamiCrafter's hybrid conditioning; raise early with a clear message if
    # a non-LatentVisualDiffusion checkpoint is loaded.
    cond_stage: Optional[CondStage] = None
    cs_mod = getattr(base, "cond_stage_model", None)
    emb    = getattr(base, "embedder", None)
    proj   = getattr(base, "image_proj_model", None)
    if cs_mod is not None and emb is not None and proj is not None:
        # DynamiCrafter's FrozenOpenCLIPEmbedder hard-codes `self.device = "cuda"` in
        # __init__ (ignores the actual torch device of the module). We patch it so
        # token tensors land on MPS / the chosen device instead of trying to go to
        # CUDA on a mac. .to(device) is also required because open_clip creates the
        # inner model on cpu.
        for _m in (cs_mod, emb, proj):
            try:
                _m.to(device)
            except Exception:
                pass
            if hasattr(_m, "device"):
                _m.device = device

        # Patch: open_clip >=2.26 + torch >=2.3 changed MultiheadAttention mask
        # handling — the old (77, 77) causal mask triggers a shape mismatch.
        # Setting it to None is safe because we use the text encoder for embedding
        # extraction (not autoregressive generation).
        if hasattr(cs_mod, "model") and hasattr(cs_mod.model, "attn_mask"):
            cs_mod.model.attn_mask = None

        # Patch: open_clip >=2.26 removed `input_patchnorm` from VisionTransformer;
        # DynamiCrafter's FrozenOpenCLIPImageEmbedderV2 still checks it.
        if hasattr(emb, "model") and hasattr(emb.model, "visual"):
            vis = emb.model.visual
            if not hasattr(vis, "input_patchnorm"):
                vis.input_patchnorm = False
            if not hasattr(vis, "patch_dropout"):
                vis.patch_dropout = None

        cond_stage = CondStage(
            cond_stage_model=cs_mod,
            embedder=emb,
            image_proj_model=proj,
            temporal_length=cfg.num_frames,
            default_fs=cfg.fps,
        )
        print(f"[OPSDPolicies] cond-stage ok on {device}: text={type(cs_mod).__name__}, "
              f"image={type(emb).__name__}, proj={type(proj).__name__}")
    else:
        print("[OPSDPolicies] WARN: cond-stage incomplete "
              f"(cond_stage_model={cs_mod is not None}, embedder={emb is not None}, "
              f"image_proj_model={proj is not None}); training will need an external cond builder")

    policies = OPSDPolicies(
        v_theta=v_theta,
        v_old=v_old,
        vae=vae,
        device=device,
        cfg=cfg,
        cond_stage=cond_stage,
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
    device = first_frame_latent.device
    B = first_frame_latent.shape[0]

    # Initialize z_T from pure Gaussian noise, same shape as a video latent clip.
    # DynamiCrafter latent shape: [B, C, T, H', W']. If cond already carries the
    # c_concat tensor (the T-repeated first-frame latent), use its T directly —
    # this is the ground truth and avoids any mismatch with cfg.num_frames.
    if "c_concat" in cond and cond["c_concat"]:
        T = cond["c_concat"][0].shape[2]
    else:
        T = cond.get("video_length", 16)
    C, H_, W_ = first_frame_latent.shape[1], first_frame_latent.shape[-2], first_frame_latent.shape[-1]
    # Keep z in fp32 for numerical stability in scheduler arithmetic;
    # _call_unet handles casting via autocast internally.
    z = torch.randn(B, C, T, H_, W_, device=device, dtype=torch.float32)

    # Concatenate first-frame latent in the time axis (DynamiCrafter image-conditioning scheme).
    # The exact concat convention is handled inside DynamiCrafter's forward; we pass the cond dict.
    for t in scheduler.timesteps:
        if t < sigma_q.max():
            break  # stop at the intermediate step we care about
        # NB: cond is a DynamiCrafter dict (c_concat / c_crossattn / fs); the raw UNetModel
        # does NOT accept `cond=...` — it must be unpacked. We call the shared helper.
        from .opsd_trainer_i2v import _call_unet
        noise_pred = _call_unet(v, z, t, cond)
        # Scheduler step in fp32 for stability
        z = scheduler.step(noise_pred.float(), t, z).prev_sample
    return z
