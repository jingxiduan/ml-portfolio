"""Plot V4 stability, objective and reward curves from the immutable JSONL record."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def ema(values: np.ndarray, window: int = 25) -> np.ndarray:
    output = np.empty_like(values, dtype=float)
    output[0] = values[0]
    alpha = 2.0 / (window + 1)
    for index in range(1, len(values)):
        output[index] = alpha * values[index] + (1 - alpha) * output[index - 1]
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.metrics.read_text().splitlines() if line.strip()]
    if len(rows) < 2:
        raise SystemExit("at least two metric rows are required")

    steps = np.array([row["step"] for row in rows])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    axes[0, 0].plot(steps, ema(np.array([row["loss_plus"] for row in rows])), label="positive-target fit")
    axes[0, 0].plot(steps, ema(np.array([row.get("loss_anchor", 0.0) for row in rows])), label="behavior anchor")
    axes[0, 0].set_title("Bounded fit objective")
    axes[0, 0].set_ylabel("MSE")
    axes[0, 0].legend()

    raw = np.array([row.get("raw_param_grad_norm", row["param_grad_norm"]) for row in rows])
    effective = np.array([row["param_grad_norm"] for row in rows])
    axes[0, 1].plot(steps, ema(raw), label="raw gradient")
    axes[0, 1].plot(steps, ema(effective), label="effective post-clip gradient")
    axes[0, 1].axhline(1e-3, color="black", linestyle="--", linewidth=1, label="clip=1e-3")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_title("Gradient control")
    axes[0, 1].legend()

    axes[1, 0].plot(steps, ema(np.array([row["shift_ratio"] for row in rows])), label="target shift ratio")
    axes[1, 0].plot(steps, ema(np.array([row["lora_B_norm"] for row in rows])), label="LoRA-B norm")
    axes[1, 0].axhline(0.25, color="red", linestyle="--", linewidth=1, label="shift guard")
    axes[1, 0].set_title("Update diagnostics")
    axes[1, 0].legend()

    for key, label in (
        ("reward", "total reward"),
        ("R_subject_consistency", "subject consistency"),
        ("R_naturalness_penalty", "naturalness component"),
        ("R_dynamic_degree", "dynamic degree"),
    ):
        axes[1, 1].plot(steps, ema(np.array([row[key] for row in rows])), label=label)
    axes[1, 1].set_title("Training reward components — not held-out quality")
    axes[1, 1].legend()

    for axis in axes.flat:
        axis.set_xlabel("training step")
        axis.grid(alpha=0.25)
    fig.suptitle(f"V4 gradient-controlled OPSD-I2V · {len(rows)} recorded steps")
    fig.tight_layout()
    fig.savefig(args.out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
