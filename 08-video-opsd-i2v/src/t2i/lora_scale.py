"""Load OPSD LoRA at an arbitrary scale by rescaling the lora_B tensors.

Why not just use PEFT's `set_adapters(..., adapter_weights=[scale])`?
Because in our main loader we `merge_and_unload()` after attaching for inference
speed, which locks scale at 1.0. The cleanest way to sweep scales without
mutating that path is to write a temporary scaled copy of the adapter and feed
it to the same loader.

Scaling identity:
    merged_ΔW = scale * (B @ A)        # LoRA output on each matmul
             == (scale * B) @ A        # rescale B
So multiplying all lora_B rows by `scale` is mathematically equivalent to
`set_adapters([adapter], adapter_weights=[scale])`.

Usage:
    from src.t2i.lora_scale import prepare_scaled_lora
    scaled_dir = prepare_scaled_lora("weights/opsd_lora_sd35m_hpsv3", scale=0.3)
    # feed `scaled_dir` as cfg.opsd_lora_dir
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def prepare_scaled_lora(src_dir: str | Path, scale: float, dst_root: str | Path | None = None) -> Path:
    """Create (or reuse) a copy of ``src_dir`` whose lora_B weights are scaled.

    Returns the path of the scaled directory, usable as a drop-in replacement
    for the original ``opsd_lora_dir`` in SD35Config.
    """
    src = Path(src_dir)
    dst_root = Path(dst_root) if dst_root else src.parent / "lora_scaled"
    dst = dst_root / f"{src.name}__scale_{scale:.2f}"
    marker = dst / ".ready"
    if marker.is_file():
        return dst
    dst.mkdir(parents=True, exist_ok=True)
    # 1. config: copy verbatim
    shutil.copy(src / "adapter_config.json", dst / "adapter_config.json")
    # 2. weights: rescale lora_B
    sd = load_file(str(src / "adapter_model.safetensors"))
    scaled = {}
    for k, v in sd.items():
        if "lora_B" in k:
            scaled[k] = (v.float() * float(scale)).to(v.dtype)
        else:
            scaled[k] = v
    save_file(scaled, str(dst / "adapter_model.safetensors"))
    marker.write_text(json.dumps({"scale": scale}))
    return dst


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="weights/opsd_lora_sd35m_hpsv3")
    ap.add_argument("--scale", type=float, required=True)
    args = ap.parse_args()
    out = prepare_scaled_lora(args.src, args.scale)
    print(out)
