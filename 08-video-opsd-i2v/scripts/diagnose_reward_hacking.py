"""Reward-hacking diagnostic for OPSD-SD3.5-M first frames and DynamiCrafter clips.

Produces four quantitative measurements that each targets a specific HPSv3 reward-hacking
failure mode documented in the literature:

    1. HSV saturation histogram shift (Clark+2024, Fan+2024)
    2. RGB brightness / overexposure pixel ratio (Zhang+2024)
    3. Edge-contrast via Sobel magnitude percentile (reward favors "crispy" outputs)
    4. Frame-to-frame flow magnitude (RAFT-free, cheap frame diff) — tests whether the
       dynamic_degree +0.180 improvement is a saturation-driven artifact.

Everything reads from on-disk mini-pipeline outputs; no re-inference, no MPS use.

Outputs:
    docs/figures/reward_hacking_saturation.png
    docs/figures/reward_hacking_brightness.png
    docs/figures/reward_hacking_edge_contrast.png
    docs/figures/reward_hacking_motion_vs_pixelvar.png
    docs/figures/reward_hacking_summary.json
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
import os
MINI = ROOT / os.environ.get("OPSD_I2V_SOURCE", "outputs/tier1_mini")
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
print(f"[diagnose] source = {MINI}")


# ---------------------------------------------------------- loaders


def _list_frames(prefix: str) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    d = MINI / f"frames_{prefix}" / "seed_00"
    for p in sorted(d.glob("*.png")):
        out.append(np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8))
    return out


def _list_clips(prefix: str) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    d = MINI / f"videos_{prefix}" / "seed_00"
    for p in sorted(d.glob("*.mp4")):
        out.append(np.stack(list(iio.imiter(p))))
    return out


# ---------------------------------------------------------- diagnostics


def rgb_to_hsv_sat(img: np.ndarray) -> np.ndarray:
    """Return HSV saturation channel in [0, 1]."""
    r, g, b = img[..., 0] / 255.0, img[..., 1] / 255.0, img[..., 2] / 255.0
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    s = np.where(mx == 0, 0.0, (mx - mn) / np.maximum(mx, 1e-8))
    return s


def overexposure_ratio(img: np.ndarray, thresh: float = 0.95) -> float:
    """Fraction of pixels where ANY channel is >= thresh*255 (burned highlights)."""
    return float(((img / 255.0).max(axis=-1) >= thresh).mean())


def sobel_edge_p95(img: np.ndarray) -> float:
    """95th-percentile Sobel magnitude on grayscale."""
    g = img.astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    gy = np.diff(g, axis=0, prepend=g[:1])
    gx = np.diff(g, axis=1, prepend=g[:, :1])
    mag = np.sqrt(gx * gx + gy * gy)
    return float(np.percentile(mag, 95))


def frame_diff_rms(clip: np.ndarray) -> float:
    """Mean of RMS pixel differences across consecutive frames (0..255 scale)."""
    diffs = []
    for i in range(1, len(clip)):
        d = (clip[i].astype(np.float32) - clip[i - 1].astype(np.float32))
        diffs.append(float(np.sqrt((d * d).mean())))
    return float(np.mean(diffs)) if diffs else 0.0


def pixel_variance(clip: np.ndarray) -> float:
    """Spatial variance averaged over frames — proxy for 'how high-contrast' the clip is."""
    v = []
    for f in clip:
        v.append(float(f.astype(np.float32).var()))
    return float(np.mean(v))


# ---------------------------------------------------------- plots


def plot_saturation_hist(frames_base: List[np.ndarray],
                        frames_opsd: List[np.ndarray]) -> Dict[str, float]:
    sb = np.concatenate([rgb_to_hsv_sat(f).ravel() for f in frames_base])
    so = np.concatenate([rgb_to_hsv_sat(f).ravel() for f in frames_opsd])
    stats = {
        "base_sat_mean": float(sb.mean()),
        "opsd_sat_mean": float(so.mean()),
        "base_sat_p90":  float(np.percentile(sb, 90)),
        "opsd_sat_p90":  float(np.percentile(so, 90)),
    }
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    bins = np.linspace(0, 1, 41)
    ax.hist(sb, bins=bins, alpha=0.55, label=f"base  (μ={stats['base_sat_mean']:.3f})",
            color="#8aa6c1", density=True)
    ax.hist(so, bins=bins, alpha=0.55, label=f"OPSD (μ={stats['opsd_sat_mean']:.3f})",
            color="#e07b4a", density=True)
    ax.set_xlabel("HSV saturation")
    ax.set_ylabel("density")
    ax.set_title("Pixel-level saturation distribution\n"
                 "(first frames, both methods, all mini-pipeline samples)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "reward_hacking_saturation.png", dpi=140)
    plt.close(fig)
    return stats


def plot_brightness(frames_base, frames_opsd) -> Dict[str, float]:
    bb = [float((f / 255.0).mean()) for f in frames_base]
    oo = [float((f / 255.0).mean()) for f in frames_opsd]
    rb = [overexposure_ratio(f) for f in frames_base]
    ro = [overexposure_ratio(f) for f in frames_opsd]
    stats = {
        "base_brightness_mean": float(np.mean(bb)),
        "opsd_brightness_mean": float(np.mean(oo)),
        "base_overexposed_frac_mean": float(np.mean(rb)),
        "opsd_overexposed_frac_mean": float(np.mean(ro)),
    }
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    axs[0].bar([0, 1], [np.mean(bb), np.mean(oo)], color=["#8aa6c1", "#e07b4a"])
    axs[0].set_xticks([0, 1]); axs[0].set_xticklabels(["base", "OPSD"])
    axs[0].set_ylabel("mean brightness (0..1)")
    axs[0].set_title("Mean brightness")
    axs[0].grid(axis="y", alpha=0.3)
    axs[1].bar([0, 1], [100 * np.mean(rb), 100 * np.mean(ro)], color=["#8aa6c1", "#e07b4a"])
    axs[1].set_xticks([0, 1]); axs[1].set_xticklabels(["base", "OPSD"])
    axs[1].set_ylabel("% pixels with any channel ≥ 0.95")
    axs[1].set_title("Overexposed-pixel ratio\n(highlights clipping)")
    axs[1].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "reward_hacking_brightness.png", dpi=140)
    plt.close(fig)
    return stats


def plot_edge_contrast(frames_base, frames_opsd) -> Dict[str, float]:
    eb = [sobel_edge_p95(f) for f in frames_base]
    eo = [sobel_edge_p95(f) for f in frames_opsd]
    stats = {
        "base_sobel_p95_mean": float(np.mean(eb)),
        "opsd_sobel_p95_mean": float(np.mean(eo)),
    }
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.bar([0, 1], [np.mean(eb), np.mean(eo)], color=["#8aa6c1", "#e07b4a"])
    ax.set_xticks([0, 1]); ax.set_xticklabels(["base", "OPSD"])
    ax.set_ylabel("Sobel-magnitude p95")
    ax.set_title("Edge-contrast proxy\n(reward-hacking tends to overcrisp outputs)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "reward_hacking_edge_contrast.png", dpi=140)
    plt.close(fig)
    return stats


def plot_motion_vs_pixelvar(clips_base, clips_opsd) -> Dict[str, float]:
    """Does the +0.180 dynamic_degree improvement correlate with real motion,
    or with elevated spatial variance (which inflates optical-flow magnitude)?"""
    pts_base = [(frame_diff_rms(c), pixel_variance(c)) for c in clips_base]
    pts_opsd = [(frame_diff_rms(c), pixel_variance(c)) for c in clips_opsd]
    stats = {
        "base_frame_diff_rms_mean": float(np.mean([p[0] for p in pts_base])),
        "opsd_frame_diff_rms_mean": float(np.mean([p[0] for p in pts_opsd])),
        "base_pixel_var_mean":       float(np.mean([p[1] for p in pts_base])),
        "opsd_pixel_var_mean":       float(np.mean([p[1] for p in pts_opsd])),
    }
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter([p[1] for p in pts_base], [p[0] for p in pts_base],
               s=120, c="#8aa6c1", label="base clips", edgecolor="black")
    ax.scatter([p[1] for p in pts_opsd], [p[0] for p in pts_opsd],
               s=120, c="#e07b4a", label="OPSD clips", edgecolor="black")
    ax.set_xlabel("spatial pixel variance (↑ = higher contrast)")
    ax.set_ylabel("frame-to-frame RMS pixel diff (↑ = more apparent motion)")
    ax.set_title("Is 'dynamic_degree' real motion or saturation artifact?\n"
                 "If OPSD lies to the upper-right, the metric is being fooled.")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "reward_hacking_motion_vs_pixelvar.png", dpi=140)
    plt.close(fig)
    return stats


# ---------------------------------------------------------- main


def main() -> None:
    frames_base = _list_frames("base")
    frames_opsd = _list_frames("opsd")
    clips_base = _list_clips("base")
    clips_opsd = _list_clips("opsd")
    print(f"[in] frames base={len(frames_base)} opsd={len(frames_opsd)} "
          f"clips base={len(clips_base)} opsd={len(clips_opsd)}")

    summary: Dict[str, object] = {
        "source": "outputs/tier1_mini",
        "n_frames_per_method": len(frames_base),
        "n_clips_per_method": len(clips_base),
    }
    summary["saturation"] = plot_saturation_hist(frames_base, frames_opsd)
    summary["brightness"] = plot_brightness(frames_base, frames_opsd)
    summary["edge_contrast"] = plot_edge_contrast(frames_base, frames_opsd)
    summary["motion_vs_pixelvar"] = plot_motion_vs_pixelvar(clips_base, clips_opsd)

    (OUT / "reward_hacking_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"[done] figures and summary at {OUT}")


if __name__ == "__main__":
    main()
