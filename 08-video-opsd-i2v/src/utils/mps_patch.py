"""MPS compatibility patches for DynamiCrafter / diffusers on Apple Silicon.

Several ops used by DynamiCrafter are either missing or buggy on MPS as of PyTorch 2.4:
- ``aten::_fft_r2c`` and ``aten::_fft_c2r`` (used by some positional-embedding utilities)
- ``aten::nonzero_mps`` (used sporadically by diffusers samplers)
- Half-precision autocast is unstable on temporal attention

These patches silently fall back to CPU for the offending op and keep the rest on MPS.

Usage:
    from src.utils.mps_patch import apply_mps_patches
    apply_mps_patches()  # call once at program start
"""
from __future__ import annotations

import os
import warnings

import torch


def _is_mps() -> bool:
    return torch.backends.mps.is_available() and torch.backends.mps.is_built()


def _patch_fft_fallback() -> None:
    """Fallback FFT ops to CPU on MPS."""
    if not _is_mps():
        return
    _orig_fft_rfft = torch.fft.rfft

    def rfft(x, *args, **kwargs):
        if x.device.type == "mps":
            return _orig_fft_rfft(x.cpu(), *args, **kwargs).to("mps")
        return _orig_fft_rfft(x, *args, **kwargs)

    torch.fft.rfft = rfft  # type: ignore[attr-defined]


def _patch_nonzero_fallback() -> None:
    """``nonzero`` on MPS silently returns empty for some dtypes; fallback to CPU."""
    if not _is_mps():
        return
    _orig = torch.nonzero

    def nonzero(x, *args, **kwargs):
        if x.device.type == "mps":
            return _orig(x.cpu(), *args, **kwargs).to("mps")
        return _orig(x, *args, **kwargs)

    torch.nonzero = nonzero  # type: ignore[assignment]


def _set_mps_fallback_env() -> None:
    """Tell PyTorch to silently fall back any un-implemented MPS op to CPU."""
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def apply_mps_patches() -> None:
    _set_mps_fallback_env()
    _patch_fft_fallback()
    _patch_nonzero_fallback()
    if not _is_mps():
        warnings.warn("MPS not available; patches are no-ops.", RuntimeWarning)


def get_device(prefer: str = "auto") -> torch.device:
    """Return the best available device.

    Order of preference: CUDA > MPS > CPU, unless ``prefer`` is explicit.
    """
    if prefer == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if _is_mps():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(prefer)


def recommended_dtype(device: torch.device) -> torch.dtype:
    """fp16 on CUDA, fp32 on MPS (half-attention is still buggy as of 2.4), fp32 on CPU."""
    if device.type == "cuda":
        return torch.float16
    return torch.float32
