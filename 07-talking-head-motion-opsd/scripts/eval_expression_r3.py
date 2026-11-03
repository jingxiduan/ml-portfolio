"""Held-out coefficient evaluation for calibrated V5 expression runs.

This evaluator uses the same expression statistic as calibration but does not use
training audio. It reports paired bootstrap intervals against the frozen base and
keeps absolute pose jitter as a safety gate.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from motion_reward import _bandpass  # noqa: E402
from motion_reward_v5 import NON_LIP  # noqa: E402


def expression_rms(coef: torch.Tensor) -> float:
    exp = coef[:, :63][:, NON_LIP].unsqueeze(0)
    low = _bandpass(exp, 0.3, 3.0)
    return float((low.square().mean(1) + 1e-6).sqrt().mean(-1).item())


def pose_rms(coef: torch.Tensor, pose_range: torch.Tensor, lo: float, hi: float) -> float:
    pose = coef[:, 67:70].unsqueeze(0)
    filtered = _bandpass(pose, lo, hi)
    per_axis = (filtered.square().mean(1) + 1e-6).sqrt() * pose_range
    return float(per_axis.norm(dim=-1).div(np.sqrt(3.0)).item())


def bootstrap_ci(values: np.ndarray, seed: int = 0, draws: int = 10000) -> list[float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    samples = values[indices].mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def load_adapter(layers, path: Path) -> None:
    state = torch.load(path, map_location="cpu", weights_only=False)["lora"]
    with torch.no_grad():
        for index, layer in enumerate(layers):
            layer.A_t.copy_(state[f"{index}.A_t"].to(layer.A_t.device))
            layer.B_t.copy_(state[f"{index}.B_t"].to(layer.B_t.device))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-clips", type=int, default=40)
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--target", type=float, default=0.3928757608)
    parser.add_argument("--base-jitter-limit", type=float, default=1.3)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "expression_eval_r3.json")
    parser.add_argument(
        "--adapters",
        nargs="+",
        default=[
            f"seed{seed}={ROOT / 'outputs' / f'opsd_v5_r3_seed{seed}' / 'lora.pt'}"
            for seed in range(3)
        ],
    )
    args = parser.parse_args()

    from engine import TalkingHeadEngine  # noqa: E402
    import opsd_train_motion as trainer  # noqa: E402

    engine = TalkingHeadEngine(load_renderer=False)
    layers = trainer.add_lora(engine.mg.denoising_net, 8, 8)
    pose_lo = torch.tensor([float(engine.tpl[f"min_{key}"][0]) for key in ("pitch", "yaw", "roll")])
    pose_hi = torch.tensor([float(engine.tpl[f"max_{key}"][0]) for key in ("pitch", "yaw", "roll")])
    pose_range = pose_hi - pose_lo

    clips = sorted(glob.glob(str(ROOT / "data" / "audio_public" / "eval" / "*.wav")))[:args.n_clips]
    if len(clips) != args.n_clips:
        raise SystemExit(f"requested {args.n_clips} clips, found {len(clips)}")

    variants: list[tuple[str, Optional[Path]]] = [("base", None)]
    for item in args.adapters:
        name, raw_path = item.split("=", 1)
        path = Path(raw_path)
        if not path.is_file():
            raise SystemExit(f"missing adapter: {path}")
        variants.append((name, path))

    rows: dict[str, list[dict[str, float | str]]] = {}
    for name, adapter in variants:
        if adapter is None:
            trainer.set_mode("base")
        else:
            load_adapter(layers, adapter)
            trainer.set_mode("trainable")

        measurements = []
        for clip in clips:
            audio = engine.load_audio(clip, args.seconds)
            with torch.no_grad():
                coef = engine.gen_motion(audio, cfg_scale=2.0, seed=0).float().cpu()
            measurements.append(
                {
                    "clip": os.path.basename(clip),
                    "expression_rms": expression_rms(coef),
                    "pose_low_deg": pose_rms(coef, pose_range, 0.3, 3.0),
                    "pose_high_deg": pose_rms(coef, pose_range, 6.0, 12.5),
                }
            )
        rows[name] = measurements
        print(f"[eval] {name}: {len(measurements)} clips", flush=True)

    base = rows["base"]
    base_expr = np.array([float(row["expression_rms"]) for row in base])
    base_jitter = np.array([float(row["pose_high_deg"]) for row in base])
    base_error = np.abs(base_expr - args.target)

    summaries = {}
    for index, (name, _) in enumerate(variants):
        values = rows[name]
        expr = np.array([float(row["expression_rms"]) for row in values])
        jitter = np.array([float(row["pose_high_deg"]) for row in values])
        error = np.abs(expr - args.target)
        delta_error = error - base_error
        delta_jitter = jitter - base_jitter
        summary = {
            "n": len(values),
            "expression_rms_mean": float(expr.mean()),
            "expression_rms_median": float(np.median(expr)),
            "target_abs_error_mean": float(error.mean()),
            "pose_high_deg_mean": float(jitter.mean()),
            "jitter_ratio_vs_base": float(jitter.mean() / max(base_jitter.mean(), 1e-8)),
            "delta_target_error_vs_base_mean": float(delta_error.mean()),
            "delta_target_error_vs_base_ci95": bootstrap_ci(delta_error, seed=100 + index),
            "delta_jitter_vs_base_mean": float(delta_jitter.mean()),
            "delta_jitter_vs_base_ci95": bootstrap_ci(delta_jitter, seed=200 + index),
        }
        summary["target_error_gate"] = bool(
            name == "base" or summary["delta_target_error_vs_base_ci95"][1] < 0
        )
        summary["jitter_gate"] = bool(
            name == "base" or summary["jitter_ratio_vs_base"] <= args.base_jitter_limit
        )
        summaries[name] = summary

    result = {
        "target": args.target,
        "base_jitter_limit": args.base_jitter_limit,
        "n_clips": len(clips),
        "same_audio_and_generation_seed": True,
        "summaries": summaries,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(summaries, indent=2))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
