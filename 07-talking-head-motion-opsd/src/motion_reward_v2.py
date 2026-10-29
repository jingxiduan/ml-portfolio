"""Reward v2 — fixes the two reward-hacking failure modes in v1 before training:
   1. blink: v1 said "at least one blink per ~2s window" and let the policy
      overshoot to 55/min. v2 adds a hard penalty on blink-signal mean above
      a human target (~0.15 of the sigmoid signal), so going higher *hurts*.
   2. jitter: v1 was a bounded term that softened penalty near zero. v2 adds
      a quadratic hard cap: anything above ~1 degree of high-frequency std
      is heavily penalized.

Everything else (amp, couple, gaze, ood, lip-mask) is identical to v1.
"""
import torch
import torch.nn.functional as F

from motion_reward import _band_mask, _bandpass, _pearson, _smoothmax  # noqa: F401

FPS = 25.0
LIP_KP = [6, 12, 14, 17, 19, 20]
LIP_DIMS = [k * 3 + j for k in LIP_KP for j in range(3)]
POSE_DIMS = [67, 68, 69]

DEFAULT_WEIGHTS = dict(amp=1.0, jitter=0.5, couple=0.5, blink=0.5, gaze=0.3, ood=1.0)


class LivelinessRewardV2(torch.nn.Module):
    def __init__(self, template: dict, eye_probe=None, weights=None,
                 blink_target_mean: float = 0.15,
                 blink_overshoot_penalty: float = 4.0,
                 jitter_cap_deg: float = 1.0,
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
        amp = (torch.tanh(std_low / 3.0) * self.axis_w).sum(-1) / self.axis_w.sum()

        # jitter v2: v1 bounded term + hard cap above `jitter_cap_deg`
        jitter_bounded = -(std_high * self.axis_w).sum(-1) / self.axis_w.sum() / 0.5
        jitter_hard = -self.jitter_cap_penalty * \
            (F.relu(std_high - self.jitter_cap_deg) * self.axis_w).sum(-1) / self.axis_w.sum()
        jitter = jitter_bounded + jitter_hard

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
            # v1 kept: encourage "at least one blink per window" (prevents 0 blink)
            blink_min = _smoothmax(bw, dim=-1).mean(-1)
            # v2 added: hard penalty if mean blink signal exceeds human target
            # (approx human-level mean of the sigmoid signal ≈ 0.15; above = overshoot)
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
