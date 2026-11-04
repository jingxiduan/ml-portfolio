"""Quantify effective LoRA updates and paired visual effects.

This script is intentionally diagnostic: it does not claim that a larger
sharpness or motion value is universally better. It reports those values
alongside exposure, temporal flicker, and the existing naturalness score.

It computes

    Delta W_l(t) = inference_scale * (alpha / rank) * B_l(t) @ A_l(t)
    U(t) = sqrt(sum_l ||Delta W_l(t)||_F^2)
           / (sqrt(sum_l ||W_l(base)||_F^2) + eps)

and, for paired videos,

    Delta Q_i = Q(candidate_i) - Q(base_i)

where Q is an explicitly labeled diagnostic composite:

    Q = 0.35*C_safe + 0.25*N + 0.20*D_safe + 0.20*T_safe

C_safe is edge detail discounted by exposure and flicker, N is the existing
naturalness score, D_safe is frame motion discounted by flicker, and T_safe
is one minus normalized luminance flicker. Q is not a calibrated perceptual
metric; its components and per-video deltas must be reported alongside it.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, Tuple

import imageio.v3 as iio
import numpy as np
import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
EPS = 1e-12


def _adapter_pairs(path: Path) -> Iterable[Tuple[str, torch.Tensor, torch.Tensor]]:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        for key in sorted(keys):
            suffix = ".lora_A.weight"
            if not key.endswith(suffix):
                continue
            bkey = key[: -len(suffix)] + ".lora_B.weight"
            if bkey not in keys:
                raise ValueError(f"missing paired LoRA B tensor for {key}")
            module = key[len("base_model.model.") : -len(suffix)]
            yield module, handle.get_tensor(key).float(), handle.get_tensor(bkey).float()


def effective_update_ratio(
    adapter_path: Path,
    base_state: Dict[str, torch.Tensor],
    inference_scale: float,
    alpha: float,
    rank: int,
) -> Dict[str, float | int | str]:
    numerator_sq = 0.0
    denominator_sq = 0.0
    modules = 0
    missing = []
    factor = float(inference_scale) * float(alpha) / float(rank)
    for module, a, b in _adapter_pairs(adapter_path):
        base_key = f"model.diffusion_model.{module}.weight"
        if base_key not in base_state:
            missing.append(base_key)
            continue
        delta = factor * (b @ a)
        numerator_sq += float(torch.sum(delta * delta).item())
        base = base_state[base_key].float()
        denominator_sq += float(torch.sum(base * base).item())
        modules += 1
    if missing:
        raise KeyError(f"{len(missing)} adapter modules missing in base; first={missing[0]}")
    numerator = float(np.sqrt(numerator_sq))
    denominator = float(np.sqrt(denominator_sq))
    return {
        "adapter": str(adapter_path.relative_to(ROOT)),
        "inference_scale": float(inference_scale),
        "lora_alpha": float(alpha),
        "lora_rank": int(rank),
        "modules": int(modules),
        "effective_delta_frobenius": numerator,
        "base_target_frobenius": denominator,
        "effective_update_ratio": numerator / (denominator + EPS),
    }


def hsv_saturation(frames: np.ndarray) -> float:
    x = frames.astype(np.float32) / 255.0
    mx = x.max(axis=-1)
    mn = x.min(axis=-1)
    return float(np.where(mx == 0, 0.0, (mx - mn) / np.maximum(mx, 1e-8)).mean())


def overexposed(frames: np.ndarray) -> float:
    return float(((frames.astype(np.float32) / 255.0).max(axis=-1) >= 0.95).mean())


def sobel_p95(frames: np.ndarray) -> float:
    gray = frames.astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    gy = np.diff(gray, axis=1, prepend=gray[:, :1])
    gx = np.diff(gray, axis=2, prepend=gray[:, :, :1])
    return float(np.percentile(np.sqrt(gx * gx + gy * gy), 95))


def frame_rms(frames: np.ndarray) -> float:
    if len(frames) < 2:
        return 0.0
    diff = frames[1:].astype(np.float32) - frames[:-1].astype(np.float32)
    return float(np.sqrt((diff * diff).mean()))


def luminance_flicker(frames: np.ndarray) -> float:
    gray = frames.astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    means = gray.mean(axis=(1, 2))
    if len(means) < 2:
        return 0.0
    return float(np.std(np.diff(means)))


def video_metrics(path: Path) -> Dict[str, float]:
    frames = np.stack(list(iio.imiter(str(path)))
                      ).astype(np.uint8)
    sat = hsv_saturation(frames)
    over = overexposed(frames)
    sob = sobel_p95(frames)
    rms = frame_rms(frames)
    flicker = luminance_flicker(frames)
    # Same naturalness definition already used by the repository evaluator.
    nat = max(
        0.0,
        1.0
        - max(0.0, sat - 0.40)
        - max(0.0, over)
        - max(0.0, (sob - 70.0) / 80.0)
        - max(0.0, (rms - 50.0) / 60.0),
    )
    dynamic = float(np.clip(rms / 60.0, 0.0, 1.0))
    return {
        "sat": sat,
        "overexposed": over,
        "sobel_p95": sob,
        "frame_rms": rms,
        "luminance_flicker": flicker,
        "naturalness": nat,
        "dynamic": dynamic,
    }


def _safe_detail(m: Dict[str, float]) -> float:
    return m["sobel_p95"] / (1.0 + 4.0 * m["overexposed"] + m["luminance_flicker"] / 20.0)


def _safe_motion(m: Dict[str, float]) -> float:
    return m["frame_rms"] / (1.0 + m["luminance_flicker"] / 20.0)


def composite_q(m: Dict[str, float], reference: Dict[str, float]) -> float:
    # Ratios are clipped only to prevent one proxy from dominating. The raw
    # components remain in the output and are the primary evidence.
    c = float(np.clip(_safe_detail(m) / max(_safe_detail(reference), EPS), 0.0, 2.0))
    d = float(np.clip(_safe_motion(m) / max(_safe_motion(reference), EPS), 0.0, 2.0))
    t = float(np.clip(1.0 - m["luminance_flicker"] / 20.0, 0.0, 1.0))
    return 0.35 * c + 0.25 * m["naturalness"] + 0.20 * d + 0.20 * t


def bootstrap_mean(values: Iterable[float], seed: int = 2026, draws: int = 10000) -> Dict[str, float]:
    values = np.asarray(list(values), dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "lo": float(np.quantile(samples, 0.025)),
        "hi": float(np.quantile(samples, 0.975)),
    }


def paired_video_metrics(base_dir: Path, candidate_dir: Path) -> Dict[str, object]:
    base_files = {p.name: p for p in (base_dir / "seed_00").glob("*.mp4")}
    cand_files = {p.name: p for p in (candidate_dir / "seed_00").glob("*.mp4")}
    common = sorted(set(base_files) & set(cand_files))
    if not common:
        raise FileNotFoundError(f"no paired videos: {base_dir} vs {candidate_dir}")
    rows = []
    for name in common:
        b = video_metrics(base_files[name])
        c = video_metrics(cand_files[name])
        bq = composite_q(b, b)
        cq = composite_q(c, b)
        rows.append({
            "file": name,
            "base": b,
            "candidate": c,
            "delta": {key: c[key] - b[key] for key in b},
            "Q_base": bq,
            "Q_candidate": cq,
            "delta_Q": cq - bq,
        })
    def mean(field: str, side: str | None = None) -> float:
        vals = []
        for row in rows:
            obj = row[side] if side else row
            vals.append(float(obj[field]))
        return float(np.mean(vals))
    return {
        "base_dir": str(base_dir.relative_to(ROOT)),
        "candidate_dir": str(candidate_dir.relative_to(ROOT)),
        "n_pairs": len(rows),
        "mean_base": {key: mean(key, "base") for key in rows[0]["base"]},
        "mean_candidate": {key: mean(key, "candidate") for key in rows[0]["candidate"]},
        "mean_delta": {key: float(np.mean([r["delta"][key] for r in rows])) for key in rows[0]["delta"]},
        "mean_Q_base": float(np.mean([r["Q_base"] for r in rows])),
        "mean_Q_candidate": float(np.mean([r["Q_candidate"] for r in rows])),
        "mean_delta_Q": float(np.mean([r["delta_Q"] for r in rows])),
        "delta_Q_bootstrap_95ci": bootstrap_mean([r["delta_Q"] for r in rows]),
        "wins_delta_Q": int(sum(r["delta_Q"] > 0 for r in rows)),
        "per_video": rows,
    }


def load_base(path: Path) -> Dict[str, torch.Tensor]:
    obj = torch.load(str(path), map_location="cpu")
    return obj.get("state_dict", obj)


def checkpoint_table(adapter_root: Path, steps: Iterable[int], scale: float, base_state: Dict[str, torch.Tensor]) -> list[dict]:
    rows = []
    for step in steps:
        path = adapter_root / f"lora_step{step:05d}" / "adapter_model.safetensors"
        if not path.exists():
            continue
        row = effective_update_ratio(path, base_state, scale, alpha=32.0, rank=16)
        row["step"] = step
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-ckpt", type=Path, default=ROOT / "weights/dynamicrafter_512/model.ckpt")
    ap.add_argument("--adapter-root", type=Path, default=ROOT / "outputs/opsd_long_v4_gradclip_rerun")
    ap.add_argument("--scale", type=float, default=0.05)
    ap.add_argument("--scales", nargs="+", type=float, default=None)
    ap.add_argument("--steps", nargs="+", type=int, default=[50, 100, 500, 1000, 1500, 2000])
    ap.add_argument("--video-base", type=Path, default=ROOT / "outputs/eval_v4/videos_base")
    ap.add_argument("--video-candidate", type=Path, action="append", default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/quantified_effect")
    args = ap.parse_args()
    args.base_ckpt = args.base_ckpt.resolve()
    args.adapter_root = args.adapter_root.resolve()
    args.video_base = args.video_base.resolve()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)

    print("[quantify] loading base checkpoint")
    base_state = load_base(args.base_ckpt)
    scales = args.scales or [args.scale]
    updates_by_scale = {
        str(scale): checkpoint_table(args.adapter_root, args.steps, scale, base_state)
        for scale in scales
    }
    updates = updates_by_scale[str(args.scale)]
    candidate_dirs = [Path(candidate).resolve() for candidate in (args.video_candidate or [ROOT / "outputs/eval_v4/videos_step1000_scale0.05"])]
    paired = [paired_video_metrics(args.video_base, candidate) for candidate in candidate_dirs]
    result = {
        "formula": {
            "effective_update": "DeltaW = inference_scale * (alpha / rank) * B @ A",
            "effective_update_ratio": "U = sqrt(sum ||DeltaW||_F^2) / (sqrt(sum ||W_base||_F^2) + eps)",
            "quality": "Q = 0.35*C_safe + 0.25*N + 0.20*D_safe + 0.20*T_safe",
            "caveat": "Q is a diagnostic composite, not a calibrated human perceptual metric; raw components are primary.",
        },
        "training": {
            "adapter_root": str(args.adapter_root.relative_to(ROOT)),
            "scale_for_U": args.scale,
            "scales_for_U": scales,
            "alpha": 32,
            "rank": 16,
            "checkpoints": updates,
            "checkpoints_by_scale": updates_by_scale,
        },
        "paired_videos": paired,
    }
    output = args.out / "quantified_effect.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "checkpoints": updates,
        "video_summaries": [
            {key: item[key] for key in ("candidate_dir", "n_pairs", "mean_delta", "mean_delta_Q", "wins_delta_Q")}
            for item in paired
        ],
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
