"""T1 diagnostic for the jitter term.

Answers three questions (task book section 3):
  1. Is the v2 hard cap (jitter_cap_deg = 1.0) ever active?
     -> report std_high mean / p90 / max for base, v1, v2.
  2. Gradient share of each reward term w.r.t. the motion coefs
     (amp / jitter / couple / blink / gaze / ood): is jitter swamped?
  3. Does the reward-side std_high correlate with the video-side head_jitter
     (MediaPipe) across the 20 clips x variants?

Writes outputs/diag/jitter_diag.json
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(ROOT, "src")))

from motion_reward import _bandpass, LIP_DIMS  # noqa: E402
from motion_reward_v2 import LivelinessRewardV2, POSE_DIMS  # noqa: E402

FPS = 25.0


def std_high_of(coef, tpl):
    """pose 3-axis high-band (6-12.5 Hz) std in degrees, matching the reward."""
    lo = torch.tensor([float(tpl[f"min_{k}"][0]) for k in ("pitch", "yaw", "roll")])
    hi = torch.tensor([float(tpl[f"max_{k}"][0]) for k in ("pitch", "yaw", "roll")])
    rng = hi - lo
    pose = coef[..., POSE_DIMS] * rng + lo
    high = _bandpass(pose, 6.0, FPS / 2)
    return (high.pow(2).mean(1) + 1e-6).sqrt()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coefs", default=os.path.join(ROOT, "outputs", "tier2", "selected_coefs.pt"))
    ap.add_argument("--bon-metrics", default=os.path.join(ROOT, "outputs", "tier2", "bon_metrics.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "diag"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    from engine import TalkingHeadEngine  # noqa: E402
    from eye_probe import EyeProbe  # noqa: E402
    eng = TalkingHeadEngine(load_renderer=False)
    probe = EyeProbe()
    probe.load_state_dict(torch.load(os.path.join(ROOT, "outputs", "eye_probe", "eye_probe.pt"),
                                     map_location="cpu"))
    reward_fn = LivelinessRewardV2(eng.tpl, eye_probe=probe, jitter_cap_deg=1.0).to("cpu")

    bundle = torch.load(a.coefs, map_location="cpu")
    selected = bundle["selected_tuples"]
    # Each entry: ((clip_idx, variant_name), coef_tensor, ?, metrics_dict)
    print(f"[diag] {len(selected)} entries")

    # ---- Q1: std_high distribution per variant + cap activity ----
    res = {"per_variant": {}, "cap_deg": 1.0}
    by_tag, first_coef = {}, None
    def unpack(item):
        """selected_tuples entry: ((clip_idx, variant), (coef,)?, ...metrics)."""
        key, rest = item[0], item[1]
        tag = key[1] if isinstance(key, (tuple, list)) else str(key)
        coef = rest[0] if isinstance(rest, (tuple, list)) else rest
        metrics = None
        for x in item[1:]:
            if isinstance(x, dict):
                metrics = x
                break
        return tag, coef, metrics

    for item in selected:
        tag, coef, _ = unpack(item)
        if not isinstance(coef, torch.Tensor):
            continue
        c = coef.float()
        if c.dim() == 2:
            c = c.unsqueeze(0)
        if first_coef is None:
            first_coef = c
        sh = std_high_of(c, eng.tpl)
        by_tag.setdefault(tag, []).append(sh)

    for tag, vals in by_tag.items():
        allv = torch.cat(vals).flatten()
        allv = allv.numpy()
        res["per_variant"][tag] = {
            "std_high_mean": float(np.mean(allv)),
            "std_high_p90": float(np.percentile(allv, 90)),
            "std_high_max": float(np.max(allv)),
            "frac_above_cap_1deg": float(np.mean(allv > 1.0)),
            "frac_above_0p2deg": float(np.mean(allv > 0.2)),
        }

    # ---- Q2: per-term gradient share ----
    # reward_fn returns DETACHED terms, so backprop per-term by building a reward
    # whose weights isolate a single term.
    c0 = first_coef
    TERMS = ["amp", "jitter", "couple", "blink", "gaze", "ood"]
    grads = {}
    if c0 is not None:
        for term in TERMS:
            w = {t: (1.0 if t == term else 0.0) for t in TERMS}
            rf = LivelinessRewardV2(eng.tpl, eye_probe=probe, weights=w,
                                    jitter_cap_deg=1.0).to("cpu")
            x = c0.clone().detach().requires_grad_(True)
            env = torch.zeros(x.shape[0], x.shape[1])
            r = rf(x, env)
            g, = torch.autograd.grad(r.sum(), x)
            grads[term] = float(g.norm().item())
        tot = sum(v for v in grads.values() if v)
        for k, v in grads.items():
            grads[k] = {"l2": v, "share": (v / tot) if (v and tot) else None}
    res["grad_share"] = grads

    # ---- Q3: correlation reward-side std_high vs motion-side jitter_hp_power ----
    # bon_metrics.json may hold video-side numbers; selected_tuples carry motion-side
    # 'jitter_hp_power' per (clip, variant). Use per-point pairs where available.
    xs, ys = [], []
    for item in selected:
        tag, coef, metrics = unpack(item)
        if not isinstance(coef, torch.Tensor):
            continue
        c = coef.float()
        if c.dim() == 2:
            c = c.unsqueeze(0)
        sh = float(std_high_of(c, eng.tpl).mean().item())
        jhp = metrics.get("jitter_hp_power") if isinstance(metrics, dict) else None
        if jhp is not None:
            xs.append(sh)
            ys.append(float(jhp))
    corr = None
    if len(xs) >= 3:
        from scipy.stats import spearmanr
        rho, p = spearmanr(xs, ys)
        corr = {"n": len(xs), "spearman_rho": float(rho), "p_value": float(p),
                "note": "reward-side std_high (deg) vs motion-side jitter_hp_power"}
    res["corr_std_high_vs_jitter_hp"] = corr

    out = os.path.join(a.out, "jitter_diag.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    print("[ok] wrote", out)


if __name__ == "__main__":
    main()
