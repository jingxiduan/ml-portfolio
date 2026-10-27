"""Reward registry — all rewards must subclass ``BaseReward`` and register themselves."""
from __future__ import annotations

from typing import Callable, Dict, List, Type

import torch
import torch.nn as nn


class BaseReward(nn.Module):
    """Abstract base for a (video, prompts) → scalar reward.

    Shape conventions:
        video   : [B, T, 3, H, W] in [-1, 1]
        prompts : list[str] of length B
    Returns     : [B] tensor
    """

    name: str = "base"
    requires_grad: bool = True   # some rewards (e.g. RAFT flow) back-prop; some don't.

    def forward(self, video: torch.Tensor, prompts: List[str]) -> torch.Tensor:  # noqa: D401
        raise NotImplementedError


REWARD_REGISTRY: Dict[str, Type[BaseReward]] = {}


def register(name: str) -> Callable[[Type[BaseReward]], Type[BaseReward]]:
    def deco(cls: Type[BaseReward]) -> Type[BaseReward]:
        if name in REWARD_REGISTRY:
            raise ValueError(f"Reward '{name}' already registered.")
        REWARD_REGISTRY[name] = cls
        cls.name = name
        return cls

    return deco


def build_reward(name: str, **kwargs) -> BaseReward:
    if name not in REWARD_REGISTRY:
        raise KeyError(f"Unknown reward '{name}'. Available: {list(REWARD_REGISTRY)}")
    return REWARD_REGISTRY[name](**kwargs)
