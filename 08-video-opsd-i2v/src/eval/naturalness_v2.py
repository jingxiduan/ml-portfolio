"""Naturalness v2 scoring for rendered I2V video clips.
Used by h20_infer_compare.sh for the final metrics table.
"""
from __future__ import annotations
from typing import Dict
import numpy as np


SAT_THRESHOLD = 0.40
SOBEL_THRESHOLD = 70.0
SOBEL_SCALE = 80.0
JITTER_THRESHOLD = 50.0
JITTER_SCALE = 60.0


def _sat_mean(clip: np.ndarray) -> float:
    r = clip[..., 0].astype(np.float32) / 255.0
    g = clip[..., 1].astype(np.float32) / 255.0
    b = clip[..., 2].astype(np.float32) / 255.0
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    s = np.where(mx == 0, 0.0, (mx - mn) / np.maximum(mx, 1e-8))
    return float(s.mean())


def _overexposed(clip: np.ndarray) -> float:
    return float(((clip / 255.0).max(axis=-1) >= 0.95).mean())


def _sobel_p95(clip: np.ndarray) -> float:
    g = clip.astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    gy = np.diff(g, axis=1, prepend=g[:, :1])
    gx = np.diff(g, axis=2, prepend=g[:, :, :1])
    mag = np.sqrt(gx * gx + gy * gy)
    return float(np.percentile(mag, 95))


def _frame_rms(clip: np.ndarray) -> float:
    if len(clip) < 2:
        return 0.0
    diffs = clip[1:].astype(np.float32) - clip[:-1].astype(np.float32)
    return float(np.sqrt((diffs * diffs).mean()))


def naturalness_v2_video(frames: np.ndarray) -> Dict[str, float]:
    """Score a video clip (uint8, [T, H, W, 3]) for naturalness + appeal."""
    sat = _sat_mean(frames)
    over = _overexposed(frames)
    sob = _sobel_p95(frames)
    rms = _frame_rms(frames)
    pen = (max(0, sat - SAT_THRESHOLD) + max(0, over)
           + max(0, (sob - SOBEL_THRESHOLD) / SOBEL_SCALE)
           + max(0, (rms - JITTER_THRESHOLD) / JITTER_SCALE))
    nat = max(0.0, 1.0 - pen)
    dynamic = float(np.clip(rms / JITTER_SCALE, 0.0, 1.0))
    appeal = 0.5 * dynamic + 0.5 * nat
    return {"nat": nat, "dynamic": dynamic, "appeal": appeal,
            "sat": sat, "overexp": over, "sobel_p95": sob, "frame_rms": rms}
