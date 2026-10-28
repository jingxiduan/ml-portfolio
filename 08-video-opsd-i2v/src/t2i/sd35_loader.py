"""SD3.5-M loader with optional OPSD LoRA adapter.

We load `stabilityai/stable-diffusion-3.5-medium` via diffusers'
``StableDiffusion3Pipeline`` (SD3 architecture, flow-matching + MMDiT).

To attach the parent-paper's OPSD LoRA, we download `WeiChow/DiffusionOPSD` and apply the
`sd35-m-hpsv3` variant to the transformer. Everything else (VAE, text encoders, scheduler) stays
at the base model's weights.

Memory on MPS 36 GB: SD3.5-M needs ~ 12 GB in fp16 for the transformer alone. Keep VAE / text
encoders in fp16 as well. If OOM, enable `enable_sequential_cpu_offload()`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

try:
    from diffusers import StableDiffusion3Pipeline  # type: ignore
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "diffusers>=0.30 required (`pip install -U diffusers`)."
    ) from e


# ---------------------------------------------------------------------- config


@dataclass
class SD35Config:
    model_id: str = "stabilityai/stable-diffusion-3.5-medium"
    opsd_lora_repo: str = "WeiChow/DiffusionOPSD"
    opsd_lora_variant: str = "sd35-m-hpsv3"      # one of: sd35-m-hpsv3 / z-image-turbo-hpsv3 / ...
    cache_dir: str = "weights/sd35m_cache"
    opsd_lora_dir: str = "weights/opsd_lora_sd35m_hpsv3"
    enable_cpu_offload: bool = False             # set True on < 24 GB
    dtype: str = "fp16"                           # "fp16" / "fp32" / "bf16"


def _torch_dtype(name: str) -> torch.dtype:
    return {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}[name]


# ---------------------------------------------------------------------- loaders


def load_sd35_base(cfg: SD35Config, device: str | torch.device) -> StableDiffusion3Pipeline:
    """Load SD3.5-M base pipeline (no LoRA).

    Prefers a local flat snapshot at ``cfg.cache_dir`` if it looks like a diffusers
    pretrained directory (has ``model_index.json``); otherwise falls back to pulling
    ``cfg.model_id`` from the HF hub (using ``cfg.cache_dir`` as the HF cache root).
    """
    local = Path(cfg.cache_dir)
    if (local / "model_index.json").is_file():
        pretrained = str(local)
        from_kwargs = {}
    else:
        pretrained = cfg.model_id
        from_kwargs = {"cache_dir": cfg.cache_dir}
    pipe = StableDiffusion3Pipeline.from_pretrained(
        pretrained,
        torch_dtype=_torch_dtype(cfg.dtype),
        **from_kwargs,
    )
    if cfg.enable_cpu_offload:
        pipe.enable_sequential_cpu_offload()
    else:
        pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def load_sd35_with_opsd_lora(
    cfg: SD35Config, device: str | torch.device
) -> StableDiffusion3Pipeline:
    """Load SD3.5-M + OPSD LoRA.

    The parent-paper's checkpoint (``WeiChow/DiffusionOPSD`` sd35-m-hpsv3) is a raw PEFT
    adapter whose keys are prefixed with ``base_model.model.`` — diffusers'
    ``pipe.load_lora_weights`` does not pick those up correctly. We therefore attach the
    adapter directly to ``pipe.transformer`` through ``PeftModel.from_pretrained`` and
    merge it, so inference uses the merged weights without any PEFT dispatch overhead.
    """
    pipe = load_sd35_base(cfg, device)
    lora_dir = Path(cfg.opsd_lora_dir)
    if not lora_dir.is_dir() or not (lora_dir / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(
            f"OPSD LoRA not found at {lora_dir}. "
            "Run `scripts/02_download_weights.sh` to fetch WeiChow/DiffusionOPSD."
        )
    from peft import PeftModel  # local import keeps base loader lean
    pipe.transformer = PeftModel.from_pretrained(pipe.transformer, str(lora_dir))
    # Fuse so later generate() calls do not go through PEFT dispatch.
    pipe.transformer = pipe.transformer.merge_and_unload()
    pipe.transformer = pipe.transformer.to(device, dtype=_torch_dtype(cfg.dtype))
    pipe.set_progress_bar_config(disable=True)
    return pipe


# ---------------------------------------------------------------------- generate


@torch.no_grad()
def generate_first_frame(
    pipe: StableDiffusion3Pipeline,
    prompt: str,
    seed: int,
    height: int = 320,
    width: int = 512,
    num_inference_steps: int = 28,
    guidance_scale: float = 4.5,
    device: str | torch.device = "cpu",
) -> "torch.Tensor | None":
    """Return a PIL image (RGB) ready to be saved or fed to I2V."""
    g = torch.Generator(device=device if device != "mps" else "cpu").manual_seed(seed)
    out = pipe(
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=g,
    )
    return out.images[0]           # PIL.Image.Image
