"""EMA behavior-policy refresh.

Used in OPSD-I2V step 5: periodically pull the trainable LoRA weights into the frozen
behavior policy so on-policy queries don't drift away from the trainable policy.

Only LoRA parameters are EMAed — the rest of both copies share the frozen base weights.
"""
from __future__ import annotations

from typing import Iterable, Iterator, Tuple

import torch
import torch.nn as nn


class EMA:
    """Exponential moving average over a filtered subset of named parameters."""

    def __init__(self, decay: float = 0.995, name_filter: str = "lora") -> None:
        assert 0.0 <= decay < 1.0, "decay must be in [0, 1)"
        self.decay = decay
        self.name_filter = name_filter

    # ------------------------------------------------------------------ utils
    def _iter_lora_pairs(
        self, dst: nn.Module, src: nn.Module
    ) -> Iterator[Tuple[nn.Parameter, nn.Parameter]]:
        src_dict = dict(src.named_parameters())
        for name, dst_param in dst.named_parameters():
            if self.name_filter not in name:
                continue
            if name not in src_dict:
                continue
            yield dst_param, src_dict[name]

    # ----------------------------------------------------------------- update
    @torch.no_grad()
    def update(self, dst: nn.Module, src: nn.Module) -> int:
        """`dst` is the frozen behavior policy; `src` is the trainable policy.

        Returns the number of parameters updated (for logging).
        """
        n = 0
        for dst_p, src_p in self._iter_lora_pairs(dst, src):
            dst_p.data.mul_(self.decay).add_(src_p.data, alpha=1.0 - self.decay)
            n += dst_p.numel()
        return n

    @torch.no_grad()
    def hard_copy(self, dst: nn.Module, src: nn.Module) -> int:
        """Hard copy (used for the first refresh after warmup)."""
        n = 0
        for dst_p, src_p in self._iter_lora_pairs(dst, src):
            dst_p.data.copy_(src_p.data)
            n += dst_p.numel()
        return n
