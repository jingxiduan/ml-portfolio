"""Reward v4 — parameters calibrated from REAL professional-speech video.

Why v3 was wrong
----------------
v3 scored amplitude as `tanh(std_low / 3.0)`, which is MONOTONE: the reward keeps
rising with motion size, so OPSD pushes amplitude without bound. Measured in the
reward's own units (low-band 0.3-3 Hz pose std, degrees):

    real professional speech : 1.878 deg   <- measured from 25 HDTF clips
    base (untrained)         : 1.273 deg   (0.68x real)
    OPSD v3 (trained)        : 4.446 deg   (2.37x real)  <- overshoot = uncanny

More training therefore made it look worse, not better. v4 fixes the objective.

Calibrated constants (scripts/calibrate_reward_targets.py + direct measurement)
------------------------------------------------------------------------------
    AMP_TARGET_DEG   1.878  real median low-band pose std
    AMP_TOL_DEG      1.0    tolerance; at 1.273 -> 0.83, at 4.446 -> 0.04
    JITTER_CAP_DEG   0.20   real median high-band std is 0.107; 0.2 stays safe
    BLINK_TARGET     15/min real median (band 7.5-37.5), matches human 10-20
    LUMINANCE        128.2  real median (band 105-145) - reference for rendering

Change vs v3
------------
ONLY the amplitude term: monotone tanh -> peaked Gaussian centered on the measured
human band. jitter / couple / blink / gaze / ood are unchanged from v3.
"""
import torch
import torch.nn.functional as F

from motion_reward import _bandpass, _pearson, _smoothmax  # noqa: F401

FPS = 25.0
LIP_KP = [6, 12, 14, 17, 19, 20]
LIP_DIMS = [k * 3 + j for k in LIP_KP for j in range(3)]
POSE_DIMS = [67, 68, 69]

# ---- calibrated from real professional speech -------------------------------
AMP_TARGET_DEG = 1.878
AMP_TOL_DEG = 1.0
JITTER_CAP_DEG = 0.20

DEFAULT_WEIGHTS = dict(amp=1.0, jitter=0.5, couple=0.5, blink=0.5, gaze=0.3, ood=1.0)


class LivelinessRewardV4(torch.nn.Module):
    """Peaked amplitude reward centered on the measured human band."""

    def __init__(self, template: dict, eye_probe=None, weights=None,
                 blink_target_mean: float = 0.15,
                 blink_overshoot_penalty: float = 4.0,
                 amp_target_deg: float = AMP_TARGET_DEG,
                 amp_tol_deg: float = AMP_TOL_DEG,
                 jitter_cap_deg: float = JITTER_CAP_DEG,
                 jitter_cap_penalty: float = 5.0):
        super().__init__()
        self.w = {**DEFAULT_WEIGHTS, **(weights or {})}
        lo = torch.tensor([float(template[f"min_{k}"][0]) for k in ("pitch", "yaw", "roll")])
        hi = torch.tensor([float(template[f"max_{k}"][0]) for k in ("pitch", "yaw", "roll")])
        self.register_buffer("pose_lo", lo)
        self.register_buffer("pose_rng", hi - lo)
        self.register_buffer("axis_w", torch.tensor([1.5, 1.0, 0.7]))
        self.blink_target_mean = blink_target_mean
        self.blink_overshoot_penalty = blink_overshoot_penalty
        self.amp_target_deg = amp_target_deg
        self.amp_tol_deg = amp_tol_deg
        self.jitter_cap_deg = jitter_cap_deg
        self.jitter_cap_penalty = jitter_cap_penalty
        self.eye_probe = eye_probe
        if eye_probe is not None:
            for p in eye_probe.parameters():
                p.requires_grad_(False)

    def pose_deg(self, coef):
        return coef[..., POSE_DIMS] * self.pose_rng + self.pose_lo

    def forward(self, coef, env=None, return_terms=False):
        B, T, _ = coef.shape
        pose = self.pose_deg(coef)
        low = _bandpass(pose, 0.3, 3.0)
        high = _bandpass(pose, 6.0, FPS / 2)
        std_low = (low.pow(2).mean(1) + 1e-6).sqrt()
        std_high = (high.pow(2).mean(1) + 1e-6).sqrt()

        # ---- amp v4: PEAKED at the human band (this is the fix) ----
        # v3: tanh(std_low/3) monotone -> unbounded overshoot.
        # v4: Gaussian centred on the measured real-speech value, so both
        #     under-shoot (stiff) and over-shoot (uncanny) are penalised.
        dev = (std_low - self.amp_target_deg) / self.amp_tol_deg
        amp_per_axis = torch.exp(-0.5 * dev * dev)
        # reduce over the 3 pose axes (pitch/yaw/roll) with the same axis weights
        amp = (amp_per_axis * self.axis_w).sum(-1) / self.axis_w.sum()

        # ---- jitter: bounded penalty + hard cap (unchanged from v3) ----
        jitter_bounded = -(std_high * self.axis_w).sum(-1) / self.axis_w.sum() / 0.5
        jitter_hard = -self.jitter_cap_penalty * \
            (F.relu(std_high - self.jitter_cap_deg) * self.axis_w).sum(-1) / self.axis_w.sum()
        jitter = jitter_bounded + jitter_hard
        self._last_cap_active_frac = float((std_high > self.jitter_cap_deg).float().mean().cpu())

        terms = {"amp": amp, "jitter": jitter}
        if env is not None:
            speed = (low[:, 1:] - low[:, :-1]).norm(dim=-1) * FPS
            e = env[:, 1:]
            lag = 4
            terms["couple"] = _pearson(e[:, : T - 1 - lag], speed[:, lag:])
        else:
            terms["couple"] = torch.zeros(B, device=coef.device)

        if self.eye_probe is not None:
            d_exp = coef[..., :63] - coef[:, :1, :63]
            eye = self.eye_probe(d_exp)
            b = eye["blink"]
            n_w = max(T // 50, 1)
            bw = b[:, : n_w * 50].reshape(B, n_w, 50)
            blink_min = _smoothmax(bw, dim=-1).mean(-1)
            blink_overshoot = -self.blink_overshoot_penalty * \
                F.relu(b.mean(-1) - self.blink_target_mean)
            terms["blink"] = blink_min + blink_overshoot
            g = torch.stack([eye["gaze_h"], eye["gaze_v"]], -1)
            g_std = g.std(1)
            terms["gaze"] = torch.tanh(g_std / 0.1).mean(-1) - F.relu(g.mean(1).abs() - 0.3).sum(-1)
        else:
            terms["blink"] = torch.zeros(B, device=coef.device)
            terms["gaze"] = torch.zeros(B, device=coef.device)

        exp = coef[..., :63]
        pose_n = coef[..., POSE_DIMS]
        terms["ood"] = -(F.relu(exp.abs() - 3.0).pow(2).mean((1, 2))
                         + (F.relu(pose_n - 1).pow(2) + F.relu(-pose_n).pow(2)).mean((1, 2)) * 10)

        total = sum(self.w[k] * v for k, v in terms.items())
        if return_terms:
            return total, {k: v.detach() for k, v in terms.items()}
        return total
