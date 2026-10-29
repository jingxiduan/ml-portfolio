"""LoRA injection restricted to temporal blocks of the DynamiCrafter DiT.

Why temporal-only: see docs/METHOD.md §2.2. Preserves spatial priors (identity rendered from
the first frame) while giving the OPSD loop reign over temporal evolution.

Strategy: enumerate the model at runtime and collect fully-qualified names of
``to_q`` / ``to_k`` / ``to_v`` / ``to_out.0`` linear layers whose ancestor is a
``TemporalTransformer``. This is more robust than hand-written regex because
DynamiCrafter's module tree depends on load-time config (number of
``TimestepEmbedSequential`` nesting layers varies by resolution).
"""
from __future__ import annotations

from typing import List, Set

import torch.nn as nn

try:
    from peft import LoraConfig, get_peft_model
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "peft is required for LoRA injection. `pip install peft>=0.11`"
    ) from e


def _collect_temporal_attn_module_names(model: nn.Module) -> List[str]:
    """Return fully-qualified names of attention linears inside any TemporalTransformer.

    Looks for the sub-paths ``...transformer_blocks.<i>.attn{1,2}.{to_q,to_k,to_v}``
    and ``...transformer_blocks.<i>.attn{1,2}.to_out.0``.
    """
    # Lazy import to avoid pulling third_party at module-load time
    try:
        from lvdm.modules.attention import TemporalTransformer  # type: ignore
    except ImportError:
        # The caller already ensured DynamiCrafter was imported earlier in the loader
        # path; if that failed we re-raise with a hint.
        raise RuntimeError(
            "Could not import TemporalTransformer. Make sure third_party/DynamiCrafter "
            "is on sys.path (handled by dynamicrafter_wrapper._import_dynamicrafter)."
        )

    # Build a parent-path -> module map for every TemporalTransformer in the tree.
    tt_prefixes: List[str] = []
    for name, mod in model.named_modules():
        if isinstance(mod, TemporalTransformer):
            tt_prefixes.append(name)

    want_leaves = {"to_q", "to_k", "to_v"}
    targets: Set[str] = set()
    for name, mod in model.named_modules():
        if not any(name.startswith(p + ".") for p in tt_prefixes):
            continue
        # keep only to_q/to_k/to_v Linears and to_out.0 Linear
        if isinstance(mod, nn.Linear):
            leaf = name.rsplit(".", 1)[-1]
            if leaf in want_leaves:
                targets.add(name)
            elif leaf == "0" and name.rsplit(".", 2)[-2] == "to_out":
                targets.add(name)
    return sorted(targets)


# Legacy regex kept for backwards-compatibility and documentation value;
# the live code path uses the runtime-enumeration helper above.
TEMPORAL_TARGETS: List[str] = [
    r".*temporal_transformer\.transformer_blocks\.\d+\.attn1\.(to_q|to_k|to_v)",
    r".*temporal_transformer\.transformer_blocks\.\d+\.attn1\.to_out\.0",
    r".*temporal_transformer\.transformer_blocks\.\d+\.attn2\.(to_q|to_k|to_v)",
    r".*temporal_transformer\.transformer_blocks\.\d+\.attn2\.to_out\.0",
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
    # Runtime enumeration first — this gives *fully-qualified* module names which
    # PEFT accepts directly (no regex ambiguity).
    targets: List[str] = _collect_temporal_attn_module_names(model)
    if not targets:
        # Fall back to the legacy regex set (in case this is a tweaked derivative
        # that no longer has a TemporalTransformer class but keeps similar names).
        targets = list(TEMPORAL_TARGETS) + list(extra_patterns or [])
    elif extra_patterns:
        targets = targets + list(extra_patterns)
    print(f"[lora] enumerated {len(targets)} temporal-attn target modules "
          f"(e.g. {targets[0] if targets else 'NONE'})")

    cfg = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=targets,
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
