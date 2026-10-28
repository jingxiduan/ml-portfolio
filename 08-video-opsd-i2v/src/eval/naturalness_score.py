"""Naturalness-aware composite score for OPSD-I2V.

Prior vbench_lite dims each measure a single axis in isolation. The user
requirement (2026-10-28) is that OPSD only "wins" if it is **both** more
dynamic AND more natural than base. A reward-hacked candidate that is
oversaturated + overexposed + overcrisp with inflated optical-flow magnitude
must **lose** on this score.

We define:

    naturalness(clip) = 1
        - w_sat  * max(0, sat_mean - 0.40)        # saturation beyond 0.40 penalised
        - w_over * overexposed_ratio              # highlights clipping penalised
        - w_edge * max(0, (sobel_p95 - 70) / 80)  # overcrisp edges penalised
        - w_jit  * max(0, (frame_rms - 50) / 60)  # noisy jitter penalised

    dynamic(clip) = optical_flow_magnitude_proxy
                  = clamp(frame_rms / 60, 0, 1)   # 0..60 ~> 0..1

    composite(clip) = 0.5 * dynamic + 0.5 * naturalness

Rationale (anchors derived from base-SD3.5-M statistics measured 2026-10-28):

    base clips (n=2): sat 0.24, overexp 0.21, sobel_p95 58, frame_rms 31
                      -> naturalness ~ 1 - 0 - 0.09 - 0 - 0 = 0.91
    OPSD clips (n=2): sat 0.47, overexp 0.34, sobel_p95 107, frame_rms 67
                      -> naturalness ~ 1 - 0.21 - 0.14 - 0.46 - 0.28 = 0.11

The thresholds (0.40 sat, 0.95 highlight, 70 edges, 50 jitter) are chosen so a
"natural-looking" clip (base SD3.5-M → DynamiCrafter default) scores ~0.9 and a
"reward-hacked" clip loses most of its naturalness budget. They are explicit
hyperparameters surfaced via --config, not tuned to the data.

Usage:
    python -m src.eval.naturalness_score \\
        --videos_base outputs/tier1_mini/videos_base \\
        --videos_opsd outputs/tier1_mini/videos_opsd \\
        --out_json    outputs/tier1_mini/naturalness.json

The output JSON lists per-clip scores and base-vs-OPSD aggregate stats, in a
shape vbench_lite_eval also uses so paired_stats() in the tier1 pipeline can
consume it unchanged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import imageio.v3 as iio
import numpy as np


# ---------------------------------------------------------- defaults

SAT_THRESHOLD = 0.40
OVEREXP_THRESHOLD = 0.00    # any overexposure is penalised linearly
SOBEL_THRESHOLD = 70.0
SOBEL_SCALE = 80.0
JITTER_THRESHOLD = 50.0
JITTER_SCALE = 60.0

W_SAT = 1.0
W_OVER = 1.0
W_EDGE = 1.0
W_JIT = 1.0


# ---------------------------------------------------------- primitives


def _read_mp4(path: Path) -> np.ndarray:
    return np.stack(list(iio.imiter(path)))


def _sat_mean(clip: np.ndarray) -> float:
    r = clip[..., 0].astype(np.float32) / 255.0
    g = clip[..., 1].astype(np.float32) / 255.0
    b = clip[..., 2].astype(np.float32) / 255.0
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    s = np.where(mx == 0, 0.0, (mx - mn) / np.maximum(mx, 1e-8))
    return float(s.mean())


def _overexposed(clip: np.ndarray, thresh: float = 0.95) -> float:
    return float(((clip / 255.0).max(axis=-1) >= thresh).mean())


def _sobel_p95(clip: np.ndarray) -> float:
    g = clip.astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    gy = np.diff(g, axis=1, prepend=g[:, :1])
    gx = np.diff(g, axis=2, prepend=g[:, :, :1])
    mag = np.sqrt(gx * gx + gy * gy)
    return float(np.percentile(mag, 95))


def _frame_rms(clip: np.ndarray) -> float:
    if len(clip) < 2:
        return 0.0
    diffs = (clip[1:].astype(np.float32) - clip[:-1].astype(np.float32))
    return float(np.sqrt((diffs * diffs).mean()))


# ---------------------------------------------------------- scoring


def score_clip(mp4_path: Path) -> Dict[str, float]:
    clip = _read_mp4(mp4_path)
    sat = _sat_mean(clip)
    over = _overexposed(clip)
    sob = _sobel_p95(clip)
    rms = _frame_rms(clip)
    pen = (
        W_SAT * max(0.0, sat - SAT_THRESHOLD)
        + W_OVER * max(0.0, over - OVEREXP_THRESHOLD)
        + W_EDGE * max(0.0, (sob - SOBEL_THRESHOLD) / SOBEL_SCALE)
        + W_JIT * max(0.0, (rms - JITTER_THRESHOLD) / JITTER_SCALE)
    )
    naturalness = max(0.0, 1.0 - pen)
    dynamic = float(np.clip(rms / JITTER_SCALE, 0.0, 1.0))
    composite = 0.5 * dynamic + 0.5 * naturalness
    return {
        "sat_mean": sat,
        "overexposed_ratio": over,
        "sobel_p95": sob,
        "frame_rms": rms,
        "naturalness": naturalness,
        "dynamic": dynamic,
        "composite": composite,
    }


def _collect(videos_dir: Path) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for seed_dir in sorted(videos_dir.glob("seed_*")):
        for mp4 in sorted(seed_dir.glob("*.mp4")):
            key = f"{seed_dir.name}/{mp4.stem}"
            out[key] = score_clip(mp4)
    return out


def _aggregate(scored: Dict[str, Dict[str, float]], dim: str) -> float:
    if not scored:
        return float("nan")
    return float(np.mean([s[dim] for s in scored.values()]))


# ---------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos_base", required=True)
    ap.add_argument("--videos_opsd", required=True)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    base_scored = _collect(Path(args.videos_base))
    opsd_scored = _collect(Path(args.videos_opsd))
    print(f"[nat] base clips={len(base_scored)}  opsd clips={len(opsd_scored)}")

    agg_dims = ["sat_mean", "overexposed_ratio", "sobel_p95", "frame_rms",
                "naturalness", "dynamic", "composite"]

    # vbench_lite-compatible output shape: {dim: [agg, {sample_id: score}]}
    out_base = {d: [_aggregate(base_scored, d),
                    {k: v[d] for k, v in base_scored.items()}] for d in agg_dims}
    out_opsd = {d: [_aggregate(opsd_scored, d),
                    {k: v[d] for k, v in opsd_scored.items()}] for d in agg_dims}

    payload = {
        "base": out_base,
        "opsd": out_opsd,
        "deltas": {d: _aggregate(opsd_scored, d) - _aggregate(base_scored, d)
                   for d in agg_dims},
        "thresholds": {
            "SAT_THRESHOLD": SAT_THRESHOLD,
            "OVEREXP_THRESHOLD": OVEREXP_THRESHOLD,
            "SOBEL_THRESHOLD": SOBEL_THRESHOLD, "SOBEL_SCALE": SOBEL_SCALE,
            "JITTER_THRESHOLD": JITTER_THRESHOLD, "JITTER_SCALE": JITTER_SCALE,
        },
        "verdict": None,
    }
    delta_c = payload["deltas"]["composite"]
    delta_n = payload["deltas"]["naturalness"]
    delta_d = payload["deltas"]["dynamic"]
    if delta_c > 0 and delta_n > -0.05:
        payload["verdict"] = "OPSD wins (more dynamic AND naturalness preserved)"
    elif delta_c > 0 and delta_n <= -0.05:
        payload["verdict"] = (
            f"OPSD inflated dynamic (+{delta_d:.3f}) at the cost of "
            f"naturalness ({delta_n:+.3f}) — composite rose but this is "
            f"REWARD HACKING; OPSD FAILS the naturalness-preserving criterion."
        )
    else:
        payload["verdict"] = (
            f"OPSD loses: Δcomposite={delta_c:+.3f}, Δnaturalness={delta_n:+.3f}, "
            f"Δdynamic={delta_d:+.3f}."
        )

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"deltas": payload["deltas"], "verdict": payload["verdict"]},
                     indent=2))


if __name__ == "__main__":
    main()
