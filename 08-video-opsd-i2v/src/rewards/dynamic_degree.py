"""Dynamic-degree reward — the explicit anti-first-frame-leakage signal.

This is the second half of the motion story: motion_smoothness penalizes jittery motion,
but if left alone it is maximized by a frozen video. ``DynamicDegree`` is the counter-force,
rewarding average optical flow magnitude up to a cap τ to prevent chaos.

See docs/METHOD.md §2.3.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import BaseReward, register

try:
    from torchvision.models.optical_flow import Raft_Small_Weights, raft_small  # type: ignore
except ImportError as e:  # pragma: no cover
    raise ImportError("torchvision>=0.19 required") from e


@register("dynamic_degree")
class DynamicDegree(BaseReward):
    def __init__(self, cap_tau: float = 0.6, device: str | torch.device = "cpu") -> None:
        super().__init__()
        self.cap = cap_tau
        weights = Raft_Small_Weights.DEFAULT
        self.model = raft_small(weights=weights, progress=False).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.to(device)
        self._tf = weights.transforms()

    def forward(self, video: torch.Tensor, prompts: List[str]) -> torch.Tensor:
        B, T = video.shape[0], video.shape[1]
        v = (video.clamp(-1, 1) + 1.0) * 0.5 * 255.0
        a = v[:, :-1].flatten(0, 1)
        b = v[:, 1:].flatten(0, 1)
        a, b = self._tf(a, b)
        flows = self.model(a, b)[-1]                          # [B*(T-1), 2, H, W]
        # mean |flow| per sample, normalized by image diagonal to be resolution-agnostic:
        mag = flows.flatten(1).abs().mean(dim=1)              # [B*(T-1)]
        H, W = video.shape[-2], video.shape[-1]
        diag = (H * H + W * W) ** 0.5
        mag_norm = mag / diag
        per_video = mag_norm.view(B, T - 1).mean(dim=1)       # [B]
        # Clip at cap_tau to avoid chaotic-motion reward hacking:
        return per_video.clamp(0.0, self.cap) / self.cap
