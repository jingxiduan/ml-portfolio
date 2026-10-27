"""Motion smoothness reward via RAFT optical flow.

VBench's definition: 1 - mean(|flow(t) - flow(t-1)|) normalized to [0,1]. Lower temporal
derivative of the flow field = smoother motion.

We back-prop through the RAFT flow (RAFT is a lightweight differentiable network, works on
MPS with the mps_fallback env var set).
"""
from __future__ import annotations

import os
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import BaseReward, register

# ``torchvision.models.optical_flow.Raft_Small`` is the easiest-to-install differentiable RAFT.
try:
    from torchvision.models.optical_flow import Raft_Small_Weights, raft_small  # type: ignore
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "torchvision>=0.19 with optical_flow module is required (`pip install torchvision`)"
    ) from e


@register("motion_smoothness")
class MotionSmoothness(BaseReward):
    def __init__(self, device: str | torch.device = "cpu") -> None:
        super().__init__()
        weights = Raft_Small_Weights.DEFAULT
        self.model = raft_small(weights=weights, progress=False).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.to(device)
        self._device = torch.device(device)
        self._tf = weights.transforms()

    def _pairs(self, video: torch.Tensor) -> torch.Tensor:
        """Return (frame_t, frame_t+1) pairs as [B*(T-1), 2, 3, H, W]."""
        B, T = video.shape[0], video.shape[1]
        f_a = video[:, :-1]
        f_b = video[:, 1:]
        pair = torch.stack([f_a, f_b], dim=2)                 # [B, T-1, 2, 3, H, W]
        return pair.flatten(0, 1)                             # [B*(T-1), 2, 3, H, W]

    def forward(self, video: torch.Tensor, prompts: List[str]) -> torch.Tensor:
        """[B, T, 3, H, W] in [-1,1] -> [B] motion smoothness."""
        B, T = video.shape[0], video.shape[1]
        # torchvision raft expects uint8-like [0, 255]
        v = (video.clamp(-1, 1) + 1.0) * 0.5 * 255.0
        pair = self._pairs(v)                                 # [N, 2, 3, H, W]
        # Normalize per torchvision transforms (they want float in [0,255]):
        a, b = pair[:, 0], pair[:, 1]
        a, b = self._tf(a, b)
        # raft_small outputs a list of flow iterates; take the final one.
        flows = self.model(a, b)[-1]                          # [N, 2, H, W]
        flows = flows.view(B, T - 1, 2, flows.shape[-2], flows.shape[-1])

        if T < 3:
            return torch.ones(B, device=video.device)

        # Temporal 2nd derivative of flow = jerk magnitude; smaller is smoother.
        d = flows[:, 1:] - flows[:, :-1]                      # [B, T-2, 2, H, W]
        jerk = d.flatten(2).norm(dim=-1).mean(dim=1)          # [B]
        # Smaller jerk → larger reward; squash to (0, 1) via exp(-jerk/scale):
        return torch.exp(-jerk / (jerk.detach().mean() + 1e-3))
