"""LoRA injection restricted to temporal blocks of the DynamiCrafter DiT.

Why temporal-only: see docs/METHOD.md §2.2. Preserves spatial priors (identity rendered from
the first frame) while giving the OPSD loop reign over temporal evolution.

We rely on PEFT's ``LoraConfig`` with a hand-curated regex of module names.
"""
from __future__ import annotations

from typing import List

import torch.nn as nn

try:
    from peft import LoraConfig, get_peft_model
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "peft is required for LoRA injection. `pip install peft>=0.11`"
    ) from e


# ---- regex patterns matching DynamiCrafter's temporal-attention modules.
# These names match the official DynamiCrafter code (openaimodel.py); keep in sync.
TEMPORAL_TARGETS: List[str] = [
    # temporal self-attention in each ResBlock
    r".*temporal_transformer\.transformer_blocks\.\d+\.attn1\.(to_q|to_k|to_v)",
    r".*temporal_transformer\.transformer_blocks\.\d+\.attn1\.to_out\.0",
    # temporal cross-attention (text/image conditioning in the temporal axis)
    r".*temporal_transformer\.transformer_blocks\.\d+\.attn2\.(to_q|to_k|to_v)",
    r".*temporal_transformer\.transformer_blocks\.\d+\.attn2\.to_out\.0",
    # motion module linear layers (if present)
    r".*motion_module.*\.(to_q|to_k|to_v|proj_out)",
]


def inject_temporal_lora(
    model: nn.Module,
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.0,
    extra_patterns: List[str] | None = None,
) -> nn.Module:
    """Wrap ``model`` with LoRA on the temporal path only.

    Returns a PEFT-wrapped model. All non-LoRA parameters are set ``requires_grad=False``.
    """
    patterns = TEMPORAL_TARGETS + (extra_patterns or [])
    cfg = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=patterns,
        task_type=None,
    )
    peft_model = get_peft_model(model, cfg)
    # PEFT already freezes base; be explicit for safety:
    for name, p in peft_model.named_parameters():
        p.requires_grad_("lora_" in name)
    return peft_model


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
