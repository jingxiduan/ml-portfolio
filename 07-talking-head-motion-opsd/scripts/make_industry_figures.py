"""Build the figures for the three-industry round.

Every panel reads numbers straight off disk -- training logs, the measured
industry bands, and the held-out evaluation -- so no value is hand-typed.

  1. fig_industry_training.png   reward + amplitude vs step, one line per industry
  2. fig_industry_bands.png      measured band per industry vs where each persona
                                 actually landed (held-out clips)
  3. fig_main_comparison.png     base / v2 / v3 / v4 vs real human speech:
                                 amplitude and jitter ratios
  4. fig_method_motion.png       OPSD in motion space (schematic)
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORDER = ["finance", "realestate", "shortvideo"]
COLORS = {"finance": "#2b6cb0", "realestate": "#2f855a", "shortvideo": "#c53030",
          "base": "#4a5568", "v2": "#975a16", "v3": "#6b46c1", "v4": "#319795",
          "real": "#1a202c"}


def _read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def _smooth(x, w=50):
    if len(x) < w:
        return np.array(x, dtype=float)
    c = np.cumsum(np.insert(np.asarray(x, dtype=float), 0, 0.0))
    return (c[w:] - c[:-w]) / w


def fig_training(logs, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for name in ORDER:
        if name not in logs:
            continue
        rows = logs[name]
        step = [r["step"] for r in rows]
        rew = _smooth([r["reward"] for r in rows])
        amp = _smooth([r.get("amp", 0.0) for r in rows])
        axes[0].plot(step[len(step) - len(rew):], rew, label=name, color=COLORS[name], lw=1.6)
        axes[1].plot(step[len(step) - len(amp):], amp, label=name, color=COLORS[name], lw=1.6)
    axes[0].set_title("OPSD reward (50-step mean)")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("reward")
    axes[1].set_title("Amplitude term (peaked reward; 1.0 = on target)")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("amp reward")
    for ax in axes:
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("[fig]", out)


def fig_bands(bands, ev, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for i, key in enumerate(("low_std_deg", "high_std_deg")):
        ax = axes[i]
        for j, name in enumerate(ORDER):
            b = bands.get(name)
            if not b:
                continue
            s = b[key]
            ax.bar(j - 0.18, s["median"], width=0.35, color=COLORS[name], alpha=0.35,
                   label="measured from real footage" if j == 0 else None)
            ax.errorbar(j - 0.18, s["median"],
                        yerr=[[s["median"] - s["p25"]], [s["p75"] - s["median"]]],
                        fmt="none", ecolor="black", capsize=4, lw=1.2)
            got = next((r for r in ev["variants"] if r["variant"] == name), None)
            if got:
                v = got[key]
                ax.bar(j + 0.18, v, width=0.35, color=COLORS[name],
                       label="persona after OPSD" if j == 0 else None)
                ax.plot(j + 0.18, v, "k_", ms=14)
        ax.set_xticks(range(len(ORDER)))
        ax.set_xticklabels(ORDER)
        ax.set_title(f"{'Amplitude' if i == 0 else 'Jitter'}  "
                     f"({'0.3-3 Hz' if i == 0 else '6-12.5 Hz'}, deg)")
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("[fig]", out)


def fig_main(nat, ev, out):
    """base / v2 / v3 / v4 vs the real professional-speech band."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    names = [r["variant"] for r in nat["variants"]]
    amps = [r["low_std_deg"] for r in nat["variants"]]
    jits = [r["high_std_deg"] for r in nat["variants"]]
    xs = np.arange(len(names))
    cols = [COLORS.get(n, "#4a5568") for n in names]

    axes[0].bar(xs, amps, color=cols)
    axes[0].axhline(nat["real_band"]["low_std_deg"], color=COLORS["real"], ls="--", lw=1.4,
                    label=f"real professional speech ({nat['real_band']['low_std_deg']} deg)")
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels(names)
    axes[0].set_ylabel("low-band pose std (deg)")
    axes[0].set_title("Amplitude: v2/v3 overshoot, v4 lands in the human band")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3, axis="y")

    axes[1].bar(xs, jits, color=cols)
    axes[1].axhline(nat["real_band"]["high_std_deg"], color=COLORS["real"], ls="--", lw=1.4,
                    label=f"real ({nat['real_band']['high_std_deg']} deg)")
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels(names)
    axes[1].set_ylabel("high-band pose std (deg)")
    axes[1].set_title("Jitter: v4 matches the human band")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("[fig]", out)


def fig_method(out):
    """Schematic of OPSD migrated from pixel diffusion to motion space."""
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    box_w, box_h, y0 = 0.15, 0.52, 0.22
    xs = [0.01, 0.21, 0.41, 0.61, 0.81]
    texts = [
        "on-policy\nrollout\nz_q = rollout(audio, t1)",
        "clean anchor\ny0 = z_q -\nsigma_q * v_old",
        "reward-guided\nbounded targets\n+/- c_adv * g/||g||",
        "detached fit\nL = ||y_theta - y+||^2\n+ lam*relu(m -\n||y_theta - y-||^2)",
        "EMA refresh\nbehaviour policy",
    ]
    for x, txt in zip(xs, texts):
        ax.add_patch(plt.Rectangle((x, y0), box_w, box_h, fill=False,
                                   ec="#2b6cb0", lw=1.6))
        ax.text(x + box_w / 2, y0 + box_h / 2, txt, ha="center", va="center",
                fontsize=7.5, linespacing=1.35)
    for k in range(len(xs) - 1):
        start = xs[k] + box_w + 0.005
        end = xs[k + 1] - 0.005
        ax.annotate("", xy=(end, y0 + box_h / 2), xytext=(start, y0 + box_h / 2),
                    arrowprops=dict(arrowstyle="->", color="#4a5568", lw=1.4))
    ax.set_title("DiffusionOPSD in motion space: "
                 "sample = 73-d motion coefficients, reward = differentiable liveliness",
                 fontsize=10, pad=10)
    fig.subplots_adjust(left=0.015, right=0.985, top=0.86, bottom=0.05)
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    print("[fig]", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "outputs", "figures"))
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    logs = {}
    for name in ORDER:
        p = os.path.join(ROOT, "outputs", f"opsd_ind_{name}", "metrics.jsonl")
        if os.path.exists(p):
            logs[name] = _read_jsonl(p)
    bands = json.load(open(os.path.join(ROOT, "outputs", "industry_bands.json")))["industries"]
    ev = json.load(open(os.path.join(ROOT, "outputs", "industry_eval.json")))
    nat = json.load(open(os.path.join(ROOT, "outputs", "naturalness_eval_r3.json")))

    if logs:
        fig_training(logs, os.path.join(a.out_dir, "fig_industry_training.png"))
    fig_bands(bands, ev, os.path.join(a.out_dir, "fig_industry_bands.png"))
    fig_main(nat, ev, os.path.join(a.out_dir, "fig_main_comparison.png"))
    fig_method(os.path.join(a.out_dir, "fig_method_motion.png"))


if __name__ == "__main__":
    main()
