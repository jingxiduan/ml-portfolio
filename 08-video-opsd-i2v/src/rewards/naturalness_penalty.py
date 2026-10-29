"""Differentiable naturalness penalty — anti-reward-hacking signal.

Mirrors the logic of ``src/eval/naturalness_score.py`` but uses pure PyTorch
ops so gradients can flow through the OPSD training loop.

    pen_sat   = relu(mean_saturation - 0.40)
    pen_over  = fraction(max_channel >= 0.95)
    pen_edge  = relu((sobel_p95 - 70) / 80)
    naturalness = clamp(1 - (pen_sat + pen_over + pen_edge), 0, 1)

All ops are pure RGB arithmetic + 3×3 Sobel conv → fully differentiable,
no learned subnetwork, no external weights.

See docs/tier2/METHOD.md §2.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import BaseReward, register


def _sobel_p95(grey: torch.Tensor) -> torch.Tensor:
    """Approximate Sobel 95th percentile across spatial + temporal dims.

    Args:
        grey: [B, T, 1, H, W] greyscale in [0, 255]

    Returns:
        [B] tensor with approximate 95th percentile of edge magnitude.
    """
    B, T = grey.shape[:2]
    g = grey.reshape(B * T, 1, grey.shape[-2], grey.shape[-1])
    # Sobel-x and Sobel-y kernels
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                       dtype=g.dtype, device=g.device).reshape(1, 1, 3, 3)
    ky = kx.permute(0, 1, 3, 2)
    gx = F.conv2d(g, kx, padding=1)
    gy = F.conv2d(g, ky, padding=1)
    mag = (gx * gx + gy * gy + 1e-8).sqrt()  # [B*T, 1, H, W] — eps prevents grad(sqrt(0))=inf
    mag_flat = mag.reshape(B, -1)     # [B, T*H*W]
    # differentiable p95 approximation: sort and index
    k = max(1, int(mag_flat.shape[1] * 0.05))
    topk = mag_flat.topk(k, dim=1).values
    return topk[:, -1]  # the k-th largest ≈ 95th percentile


@register("naturalness_penalty")
class NaturalnessPenalty(BaseReward):
    """Differentiable naturalness penalty for anti-reward-hacking training.

    Returns a [0, 1] score where 1.0 = perfectly natural.
    """

    requires_grad = True

    def __init__(
        self,
        sat_threshold: float = 0.40,
        sobel_threshold: float = 70.0,
        sobel_scale: float = 80.0,
        device: str | torch.device = "cpu",
        **kwargs,
    ) -> None:
        super().__init__()
        self.sat_threshold = sat_threshold
        self.sobel_threshold = sobel_threshold
        self.sobel_scale = sobel_scale

    def forward(self, video: torch.Tensor, prompts: List[str]) -> torch.Tensor:
        """Score naturalness of video clips.

        Args:
            video: [B, T, 3, H, W] in [-1, 1]
            prompts: ignored (naturalness is prompt-agnostic)

        Returns:
            [B] tensor — unclamped so gradients always flow.
            Positive values = natural, negative = very unnatural.
            The OPSD reward-gradient step uses autograd.grad, so the actual
            scale doesn't matter — only the gradient *direction* is used
            (it gets L2-normalized anyway).
        """
        # Map from [-1, 1] to [0, 1]
        v = (video.clamp(-1, 1) + 1.0) * 0.5  # [B, T, 3, H, W] in [0, 1]

        # --- pen_sat: mean saturation penalty ---
        # Use smooth penalty instead of hard ReLU threshold
        r, g, b = v[:, :, 0], v[:, :, 1], v[:, :, 2]
        mx = torch.maximum(torch.maximum(r, g), b)
        mn = torch.minimum(torch.minimum(r, g), b)
        sat = (mx - mn) / mx.clamp(min=1e-8)
        sat_mean = sat.flatten(1).mean(dim=1)  # [B]
        # Soft penalty with sigmoid: smooth around threshold, always has gradient
        pen_sat = torch.sigmoid((sat_mean - self.sat_threshold) * 10.0)  # [0, 1]

        # --- pen_over: fraction of overexposed pixels ---
        pen_over = torch.sigmoid((mx - 0.95) * 50.0).flatten(1).mean(dim=1)  # [B]

        # --- pen_edge: Sobel edge penalty ---
        # Smooth version: sigmoid instead of ReLU clamp
        grey = (0.299 * r + 0.587 * g + 0.114 * b).unsqueeze(2) * 255.0  # [B,T,1,H,W]
        sob = _sobel_p95(grey)  # [B]
        pen_edge = torch.sigmoid((sob - self.sobel_threshold) / self.sobel_scale)  # [0, 1]

        # DO NOT clamp — let negative values flow so gradient always exists.
        # Each penalty is in [0, 1], so naturalness is in [-2, 1].
        # Typical values: ~0.5 for normal video, ~-1.5 for very unnatural.
        naturalness = 1.0 - pen_sat - pen_over - pen_edge
        return naturalness
