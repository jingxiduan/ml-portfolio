"""Weighted combination of reward components.

Default weights (set in ``configs/vbench_rewards.yaml``):
    subject_consistency  : 1.0
    motion_smoothness    : 0.5
    dynamic_degree       : 1.0    # critical — see docs/METHOD.md §2.3
    aesthetic            : 0.5
"""
from __future__ import annotations

from typing import Dict, List

import torch

from .registry import BaseReward, build_reward


class CombinedReward(BaseReward):
    def __init__(self, components: Dict[str, Dict], device: str = "cpu") -> None:
        super().__init__()
        self.weights: Dict[str, float] = {}
        self.rewards = torch.nn.ModuleDict()
        for name, spec in components.items():
            w = float(spec.pop("weight", 1.0))
            self.weights[name] = w
            self.rewards[name] = build_reward(name, device=device, **spec)
        # Ensure all reward sub-models are in fp32 — reward gradients must be
        # computed in full precision (fp16 causes underflow/mismatch on H20).
        self.float()

    def forward(self, video: torch.Tensor, prompts: List[str]) -> torch.Tensor:
        out = None
        for name, r in self.rewards.items():
            val = r(video, prompts)
            w = self.weights[name]
            out = val * w if out is None else out + val * w
        assert out is not None
        return out

    def components(self, video: torch.Tensor, prompts: List[str]) -> Dict[str, torch.Tensor]:
        """Return per-component values (for logging / debugging)."""
        return {name: r(video, prompts) for name, r in self.rewards.items()}
