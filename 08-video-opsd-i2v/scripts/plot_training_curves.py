"""Plot OPSD-I2V training curves from train_metrics.jsonl.

Produces publication-quality training figures:
  - Fig 1: Total reward + per-component rewards vs GPU-hours
  - Fig 2: Loss (L+, L-, contrastive) vs steps
  - Fig 3: Learning rate schedule + gradient norm
"""
import json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def load_metrics(path: str):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def smooth(y, window=5):
    """Exponential moving average."""
    s = np.zeros_like(y)
    s[0] = y[0]
    alpha = 2.0 / (window + 1)
    for i in range(1, len(y)):
        s[i] = alpha * y[i] + (1 - alpha) * s[i-1]
    return s


def plot_rewards(rows, out_dir: Path, label: str = ""):
    steps = np.array([r["step"] for r in rows])
    gpu_h = np.array([r.get("gpu_hours", r["wall_sec"] / 3600) for r in rows])
    reward = np.array([r["reward"] for r in rows])
    
    # Find per-component reward keys
    comp_keys = sorted(set(k for r in rows for k in r if k.startswith("R_")))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: reward vs GPU-hours
    ax = axes[0]
    ax.plot(gpu_h, smooth(reward, 10), "k-", linewidth=2, label="Total Reward")
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(comp_keys), 1)))
    for i, k in enumerate(comp_keys):
        vals = np.array([r.get(k, float("nan")) for r in rows])
        valid = ~np.isnan(vals)
        if valid.sum() > 0:
            ax.plot(gpu_h[valid], smooth(vals[valid], 10), "--", color=colors[i],
                    linewidth=1.5, label=k.replace("R_", ""))
    ax.set_xlabel("GPU-hours")
    ax.set_ylabel("Reward")
    ax.set_title(f"Training Reward vs GPU-hours {label}")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    # Right: reward vs steps
    ax = axes[1]
    ax.plot(steps, smooth(reward, 10), "k-", linewidth=2, label="Total Reward")
    for i, k in enumerate(comp_keys):
        vals = np.array([r.get(k, float("nan")) for r in rows])
        valid = ~np.isnan(vals)
        if valid.sum() > 0:
            ax.plot(steps[valid], smooth(vals[valid], 10), "--", color=colors[i],
                    linewidth=1.5, label=k.replace("R_", ""))
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Reward")
    ax.set_title(f"Training Reward vs Steps {label}")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(out_dir / "fig_reward_curves.png")
    fig.savefig(out_dir / "fig_reward_curves.pdf")
    plt.close(fig)
    print(f"Saved: {out_dir}/fig_reward_curves.{{png,pdf}}")


def plot_loss(rows, out_dir: Path, label: str = ""):
    steps = np.array([r["step"] for r in rows])
    loss = np.array([r["loss"] for r in rows])
    loss_p = np.array([r["loss_plus"] for r in rows])
    loss_m = np.array([r["loss_minus"] for r in rows])
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, smooth(loss_p, 10), "-", color="#2196F3", label="L+ (towards reward)")
    ax.plot(steps, smooth(loss_m, 10), "-", color="#F44336", label="L- (away from reward)")
    ax.plot(steps, smooth(loss, 10), "-", color="black", linewidth=2, label="OPSD loss (L+ − L−)")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Loss")
    ax.set_title(f"OPSD Training Loss {label}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(out_dir / "fig_loss_curves.png")
    fig.savefig(out_dir / "fig_loss_curves.pdf")
    plt.close(fig)
    print(f"Saved: {out_dir}/fig_loss_curves.{{png,pdf}}")


def plot_diagnostics(rows, out_dir: Path, label: str = ""):
    steps = np.array([r["step"] for r in rows])
    lr = np.array([r["lr"] for r in rows])
    gnorm = np.array([r["grad_norm"] for r in rows])
    sigma = np.array([r["sigma_q"] for r in rows])
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    
    axes[0].plot(steps, lr, "-", color="#4CAF50")
    axes[0].set_xlabel("Steps"); axes[0].set_ylabel("Learning Rate")
    axes[0].set_title("LR Schedule"); axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(steps, smooth(gnorm, 10), "-", color="#FF9800")
    axes[1].set_xlabel("Steps"); axes[1].set_ylabel("Grad Norm (after rescale)")
    axes[1].set_title("Gradient Norm"); axes[1].grid(True, alpha=0.3)
    
    axes[2].scatter(steps, sigma, s=3, alpha=0.3, color="#9C27B0")
    axes[2].set_xlabel("Steps"); axes[2].set_ylabel("σ_q (noise level)")
    axes[2].set_title("Query Noise Level"); axes[2].grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(out_dir / "fig_diagnostics.png")
    fig.savefig(out_dir / "fig_diagnostics.pdf")
    plt.close(fig)
    print(f"Saved: {out_dir}/fig_diagnostics.{{png,pdf}}")


if __name__ == "__main__":
    metrics_path = sys.argv[1] if len(sys.argv) > 1 else "outputs/opsd_tier2/train_metrics.jsonl"
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(metrics_path).parent / "figures"
    label = sys.argv[3] if len(sys.argv) > 3 else ""
    
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_metrics(metrics_path)
    print(f"Loaded {len(rows)} training steps from {metrics_path}")
    
    if len(rows) < 2:
        print("Not enough data points for plotting.")
        sys.exit(1)
    
    plot_rewards(rows, out_dir, label)
    plot_loss(rows, out_dir, label)
    plot_diagnostics(rows, out_dir, label)
    print(f"\nAll figures saved to {out_dir}/")
