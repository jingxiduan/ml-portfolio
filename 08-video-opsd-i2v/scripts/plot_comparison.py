"""Plot comparison training curves for all 3 OPSD-I2V experiments.

Produces a single figure with 3 panels showing how rewards evolve
across different configurations.
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

EXPERIMENTS = {
    "v1_nat_clamp": {
        "path": "outputs/opsd_tier2/train_metrics.jsonl",
        "label": "V1: Naturalness (clamped)",
        "color": "#2196F3",
        "ls": "--",
    },
    "v2_nat_sigmoid": {
        "path": "outputs/opsd_tier2_v2/train_metrics.jsonl",
        "label": "V2: Naturalness (sigmoid) [Ours]",
        "color": "#4CAF50",
        "ls": "-",
    },
    "no_nat": {
        "path": "outputs/opsd_tier2_no_nat/train_metrics.jsonl",
        "label": "Ablation: No Naturalness",
        "color": "#F44336",
        "ls": "-.",
    },
}

def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def smooth(y, w=15):
    s = np.zeros_like(y)
    s[0] = y[0]
    a = 2.0 / (w + 1)
    for i in range(1, len(y)):
        s[i] = a * y[i] + (1 - a) * s[i-1]
    return s

def main():
    out_dir = Path("outputs/figures_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    data = {}
    for key, cfg in EXPERIMENTS.items():
        p = Path(cfg["path"])
        if p.exists():
            data[key] = load(str(p))
            print(f"Loaded {key}: {len(data[key])} steps")
        else:
            print(f"SKIP {key}: {p} not found")
    
    # ===== Figure 1: Total Reward comparison =====
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Panel 1: Total Reward vs Steps
    ax = axes[0]
    for key, rows in data.items():
        cfg = EXPERIMENTS[key]
        steps = np.array([r["step"] for r in rows])
        reward = np.array([r["reward"] for r in rows])
        ax.plot(steps, smooth(reward), cfg["ls"], color=cfg["color"],
                linewidth=2, label=cfg["label"])
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Total Reward")
    ax.set_title("(a) Total Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 2: Subject Consistency
    ax = axes[1]
    for key, rows in data.items():
        cfg = EXPERIMENTS[key]
        steps = np.array([r["step"] for r in rows])
        sc = np.array([r.get("R_subject_consistency", 0) for r in rows])
        ax.plot(steps, smooth(sc), cfg["ls"], color=cfg["color"],
                linewidth=2, label=cfg["label"])
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Subject Consistency")
    ax.set_title("(b) Subject Consistency (CLIP)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.95, 1.0)
    
    # Panel 3: Dynamic Degree
    ax = axes[2]
    for key, rows in data.items():
        cfg = EXPERIMENTS[key]
        steps = np.array([r["step"] for r in rows])
        dd = np.array([r.get("R_dynamic_degree", 0) for r in rows])
        ax.plot(steps, smooth(dd), cfg["ls"], color=cfg["color"],
                linewidth=2, label=cfg["label"])
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Dynamic Degree")
    ax.set_title("(c) Dynamic Degree")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.suptitle("OPSD-I2V Training Comparison: Reward Components", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "comparison_rewards.png")
    fig.savefig(out_dir / "comparison_rewards.pdf")
    plt.close(fig)
    print(f"Saved: comparison_rewards.{{png,pdf}}")
    
    # ===== Figure 2: Naturalness evolution (V2 only) =====
    if "v2_nat_sigmoid" in data:
        rows = data["v2_nat_sigmoid"]
        steps = np.array([r["step"] for r in rows])
        nat = np.array([r.get("R_naturalness_penalty", 0) for r in rows])
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(steps, smooth(nat, 20), "-", color="#4CAF50", linewidth=2,
                label="V2 Naturalness Score")
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5, label="Natural threshold")
        ax.fill_between(steps, smooth(nat, 20), 0, alpha=0.15, color="#4CAF50")
        ax.set_xlabel("Training Steps")
        ax.set_ylabel("Naturalness Score")
        ax.set_title("Naturalness Penalty Evolution (V2: Sigmoid)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "naturalness_evolution.png")
        fig.savefig(out_dir / "naturalness_evolution.pdf")
        plt.close(fig)
        print(f"Saved: naturalness_evolution.{{png,pdf}}")
    
    # ===== Figure 3: GPU-hours comparison =====
    fig, ax = plt.subplots(figsize=(8, 5))
    for key, rows in data.items():
        cfg = EXPERIMENTS[key]
        gpu_h = np.array([r.get("gpu_hours", r["wall_sec"]/3600) for r in rows])
        reward = np.array([r["reward"] for r in rows])
        ax.plot(gpu_h, smooth(reward), cfg["ls"], color=cfg["color"],
                linewidth=2, label=cfg["label"])
    ax.set_xlabel("GPU-hours (H100)")
    ax.set_ylabel("Total Reward")
    ax.set_title("Training Efficiency: Reward vs GPU-hours")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "reward_vs_gpu_hours.png")
    fig.savefig(out_dir / "reward_vs_gpu_hours.pdf")
    plt.close(fig)
    print(f"Saved: reward_vs_gpu_hours.{{png,pdf}}")
    
    print(f"\nAll comparison figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
