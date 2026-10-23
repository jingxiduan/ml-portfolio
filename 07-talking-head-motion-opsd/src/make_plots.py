"""Training-curve and before/after metric plots for the README."""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")


def smooth(x, k=15):
    if len(x) < k:
        return x
    return np.convolve(x, np.ones(k) / k, mode="valid")


def training_curve():
    rows = [json.loads(l) for l in open(os.path.join(OUT, "opsd", "metrics.jsonl"))]
    step = [r["step"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].plot(step, [r["reward"] for r in rows], color="tab:blue", alpha=0.25)
    ax[0].plot(step[14:], smooth([r["reward"] for r in rows]), color="tab:blue", lw=2)
    ax[0].set_title("Total liveliness reward")
    ax[0].set_xlabel("OPSD round")
    ax[0].grid(alpha=0.3)
    for k, c in [("amp", "tab:green"), ("blink", "tab:orange"), ("gaze", "tab:purple"),
                 ("couple", "tab:red"), ("jitter", "tab:brown")]:
        ax[1].plot(step[14:], smooth([r[k] for r in rows]), label=k, color=c, lw=1.8)
    ax[1].set_title("Reward components (smoothed)")
    ax[1].set_xlabel("OPSD round")
    ax[1].legend(fontsize=8, ncol=2)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "samples", "training_curve.png"), dpi=130)
    print("training_curve.png")


def before_after():
    res = json.load(open(os.path.join(OUT, "eval", "eval_results.json")))
    vs = res["video"]["summary"]
    keys = ["head_pitch_std", "head_yaw_std", "head_speed_mean", "nod_rate", "blink_rate", "gaze_std",
            "gaze_shift_rate", "speech_head_corr"]
    labels = ["pitch std\n(deg)", "yaw std\n(deg)", "head speed\n(deg/s)", "nod rate\n(/s)", "blink rate\n(/min)",
              "gaze std", "gaze shift\n(/s)", "speech-head\ncorr"]
    base = [vs["base"][k] for k in keys]
    opsd = [vs["opsd"][k] for k in keys]
    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.bar(x - 0.2, base, 0.4, label="base", color="tab:gray")
    ax.bar(x + 0.2, opsd, 0.4, label="+ OPSD", color="tab:green")
    for i, (b, o) in enumerate(zip(base, opsd)):
        ax.text(i - 0.2, b, f"{b:.1f}", ha="center", va="bottom", fontsize=7)
        ax.text(i + 0.2, o, f"{o:.1f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_title("Held-out video metrics (MediaPipe, independent of the training reward)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "samples", "before_after_metrics.png"), dpi=130)
    print("before_after_metrics.png")


if __name__ == "__main__":
    os.makedirs(os.path.join(OUT, "samples"), exist_ok=True)
    training_curve()
    before_after()
