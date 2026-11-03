"""Measure expression-dynamics targets from extracted real-video coefficients.

The input directory is produced by ``build_video_motion_dataset.py`` and contains
paired ``*.wav`` and ``*.coef.pt`` files. Coefficients are already normalized with
the same JoyVASA/LivePortrait template used by the trainer, so no renderer or
model weights are required here.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from motion_reward import LIP_DIMS, _bandpass  # noqa: E402

NON_LIP = [i for i in range(63) if i not in LIP_DIMS]


def _quantiles(values: torch.Tensor) -> dict[str, float]:
    return {
        "p10": float(torch.quantile(values, 0.10)),
        "p25": float(torch.quantile(values, 0.25)),
        "median": float(torch.quantile(values, 0.50)),
        "p75": float(torch.quantile(values, 0.75)),
        "p90": float(torch.quantile(values, 0.90)),
    }


def measure(path: Path, window: int) -> list[float]:
    coef = torch.load(path, map_location="cpu").float()
    if coef.ndim == 3 and coef.shape[0] == 1:
        coef = coef[0]
    if coef.ndim != 2 or coef.shape[1] < 70:
        raise ValueError(f"expected [T, >=70], got {tuple(coef.shape)}")

    values: list[float] = []
    for start in range(0, coef.shape[0] - window + 1, window):
        exp = coef[start:start + window, :63][:, NON_LIP].unsqueeze(0)
        low = _bandpass(exp, 0.3, 3.0)
        value = (low.square().mean(1) + 1e-6).sqrt().mean(-1)
        values.append(float(value.item()))
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coef-dir", type=Path, default=ROOT / "data" / "motion_ds")
    parser.add_argument("--window", type=int, default=100, help="frames per measurement window")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "behaviour_bands.json")
    args = parser.parse_args()

    files = sorted(args.coef_dir.glob("*.coef.pt"))
    if not files:
        raise SystemExit(f"no *.coef.pt files found in {args.coef_dir}")

    per_file: dict[str, list[float]] = {}
    skipped: dict[str, str] = {}
    pooled: list[float] = []
    for path in files:
        try:
            values = measure(path, args.window)
            if values:
                per_file[path.name] = values
                pooled.extend(values)
        except Exception as exc:
            skipped[path.name] = f"{type(exc).__name__}: {exc}"

    if not pooled:
        raise SystemExit("no complete windows could be measured")

    pooled_tensor = torch.tensor(pooled)
    summary = {
        "definition": "mean per-dimension RMS of 0.3-3 Hz non-lip normalized expression coefficients",
        "units": "JoyVASA normalized coefficient units",
        "window_frames": args.window,
        "fps": 25,
        "files_seen": len(files),
        "files_used": len(per_file),
        "windows": len(pooled),
        "pooled": _quantiles(pooled_tensor),
        "recommended_target": float(torch.quantile(pooled_tensor, 0.50)),
        "recommended_tolerance": float(
            max(torch.quantile(pooled_tensor, 0.75) - torch.quantile(pooled_tensor, 0.25), 1e-3)
        ),
        "per_file": per_file,
        "skipped": skipped,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_file"}, indent=2))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
