"""Produce the Tier-2 story figures from on-disk artifacts:

  fig_tier2_blink.png     bar chart of blink rate across 7 variants (motion-space, n=20)
  fig_tier2_summary.png   3-panel: blink_rate / jitter / total naturalness
  fig_v1_vs_v2_curve.png  reward training curves, OPSD v1 vs v2 side by side

Everything is reproduced from JSONL metrics + summary JSON that are already
committed, so the figures are regeneratable by anyone cloning the repo.
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "outputs", "tier2")
os.makedirs(FIG, exist_ok=True)

VARIANTS = ["base_naive", "base_bon", "opsd_naive",
            "opsd_bon", "opsd_bon_thin", "opsd_v2_naive", "opsd_v2_bon"]
LABEL = {
    "base_naive":    "base\n(JoyVASA)",
    "base_bon":      "base\n+BoN",
    "opsd_naive":    "OPSD v1\n(naive)\n← hacked",
    "opsd_bon":      "v1 + BoN\n(Track A)",
    "opsd_bon_thin": "v1 + BoN\n+ thin\n(Track A+B)",
    "opsd_v2_naive": "OPSD v2\n(naive)\n(Track C)",
    "opsd_v2_bon":   "v2 + BoN\n(Track C+A)",
}
COLOR = {
    "base_naive":    "#999999",
    "base_bon":      "#bbbbbb",
    "opsd_naive":    "#d62728",  # red — the hacked baseline
    "opsd_bon":      "#ff9896",
    "opsd_bon_thin": "#2ca02c",  # green — fix path 1 (treat symptom)
    "opsd_v2_naive": "#9467bd",
    "opsd_v2_bon":   "#1f77b4",  # blue — fix path 2 (treat cause)
}


def load_motion_summary():
    with open(os.path.join(ROOT, "outputs", "tier2", "bon_motion_summary.json")) as f:
        return json.load(f)


def fig_blink():
    """Single-panel blink-rate bar chart with the 10-20/min human band shaded."""
    d = load_motion_summary()["avg"]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    xs = np.arange(len(VARIANTS))
    vals = [d[v]["blink_rate_est"] for v in VARIANTS]
    bars = ax.bar(xs, vals, color=[COLOR[v] for v in VARIANTS], edgecolor="#333", linewidth=0.5)
    # human band 10-20
    ax.axhspan(10, 20, color="#e8f5e8", zorder=0, label="human range (10-20/min)")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}",
                ha="center", fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels([LABEL[v] for v in VARIANTS], fontsize=8.5)
    ax.set_ylabel("blink rate (/min, motion-space estimate via eye probe)")
    ax.set_title("Tier-2 · blink rate (n=20 held-out clips, motion-space)\n"
                 "OPSD v1 reward-hacks blink to ~49/min; Track A+B treats the symptom, "
                 "Track C treats the cause — both land in the human range",
                 fontsize=11)
    ax.set_ylim(0, max(vals) * 1.2)
    ax.axhline(0, color="#333", linewidth=0.5)
    ax.legend(loc="upper right", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = os.path.join(FIG, "fig_tier2_blink.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[write] {out}")


def fig_summary():
    """Three-panel: blink / jitter / total naturalness, side by side."""
    d = load_motion_summary()["avg"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.0))

    panels = [
        ("blink_rate_est", "blink rate (/min)", (10, 20), "human 10-20/min"),
        ("jitter_hp_power", "head jitter (high-freq power, deg²)", None, None),
        ("total", "naturalness v2 score (higher = better)", None, None),
    ]
    xs = np.arange(len(VARIANTS))
    for ax, (key, ylabel, band, band_label) in zip(axes, panels):
        vals = [d[v][key] for v in VARIANTS]
        bars = ax.bar(xs, vals, color=[COLOR[v] for v in VARIANTS],
                      edgecolor="#333", linewidth=0.5)
        if band is not None:
            ax.axhspan(band[0], band[1], color="#e8f5e8", zorder=0, label=band_label)
            ax.legend(loc="upper right", fontsize=8)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2,
                    v + (max(vals) - min(vals)) * 0.03,
                    f"{v:.2f}", ha="center", fontsize=8)
        ax.set_xticks(xs)
        ax.set_xticklabels([LABEL[v] for v in VARIANTS], fontsize=7.5)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.axhline(0, color="#333", linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("Tier-2 motion-space summary (n=20 held-out clips)",
                 fontsize=12, y=1.00)
    plt.tight_layout()
    out = os.path.join(FIG, "fig_tier2_summary.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[write] {out}")


def fig_curves():
    """Reward training curve, OPSD v1 vs v2, with blink term inset."""
    def load_jsonl(p):
        return [json.loads(l) for l in open(p)]

    v1 = load_jsonl(os.path.join(ROOT, "outputs", "opsd", "metrics.jsonl"))
    v2 = load_jsonl(os.path.join(ROOT, "outputs", "opsd_v2", "metrics.jsonl"))

    def smooth(x, k=15):
        x = np.asarray(x, dtype=float)
        if len(x) < k:
            return x
        ker = np.ones(k) / k
        return np.convolve(x, ker, mode="same")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharex=True)

    # left: total reward
    ax = axes[0]
    ax.plot([r["step"] for r in v1], smooth([r["reward"] for r in v1]),
            color="#d62728", label="OPSD v1 reward (reward_v1)", linewidth=1.6)
    ax.plot([r["step"] for r in v2], smooth([r["reward"] for r in v2]),
            color="#1f77b4", label="OPSD v2 reward (reward_v2)", linewidth=1.6)
    ax.set_xlabel("training step")
    ax.set_ylabel("reward (smoothed, window=15)")
    ax.set_title("Total reward (not directly comparable between v1/v2 —\n"
                 "both saturate healthily, meaning the OPSD mechanism itself works)",
                 fontsize=10)
    ax.legend(loc="lower right", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # right: blink term
    ax = axes[1]
    ax.plot([r["step"] for r in v1], smooth([r["blink"] for r in v1]),
            color="#d62728", label="v1 blink term (unbounded above)", linewidth=1.6)
    ax.plot([r["step"] for r in v2], smooth([r["blink"] for r in v2]),
            color="#1f77b4", label="v2 blink term (gaussian + overshoot penalty)", linewidth=1.6)
    ax.set_xlabel("training step")
    ax.set_ylabel("blink term (smoothed)")
    ax.set_title("Blink reward component:\n"
                 "v1 keeps climbing (policy keeps gaming it) → produces 49/min overshoot\n"
                 "v2 saturates earlier (penalty activates) → produces 41/min, more human",
                 fontsize=10)
    ax.legend(loc="lower right", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle("OPSD v1 vs v2 training curves (600 rounds on Mac M3 Pro, ~11 min)",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    out = os.path.join(FIG, "fig_v1_vs_v2_curve.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[write] {out}")


def main():
    fig_blink()
    fig_summary()
    fig_curves()


if __name__ == "__main__":
    main()
