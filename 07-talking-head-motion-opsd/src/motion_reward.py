"""Differentiable "liveliness" reward on JoyVASA motion coefficients (no rendering).

Input: normalized motion coefficients (B, T, 73) at 25 fps, as the DiT outputs them.
Every term is computed from the motion itself (head pose) or through the learned
EyeProbe (expression offsets -> blink / gaze), so the whole reward is differentiable
w.r.t. the coefficients, which OPSD's reward-gradient step needs.

Terms (all bounded, so the policy cannot win by shaking the head to infinity):
  amp     low-frequency (0.3-3 Hz) head rotation amplitude, saturating at a few degrees
          (pitch weighted higher: that's where nods live)
  jitter  high-frequency (>6 Hz) head rotation energy, penalized
  couple  correlation between speech loudness and low-passed head speed (head moves when talking)
  blink   every ~2 s window should contain one blink, without keeping the eyes half-closed
  gaze    some variation of gaze direction, without drifting away from camera
  ood     keep coefficients inside the range the pretrained model was trained on
The mouth is not rewarded at all; OPSD also masks reward gradients on lip keypoints so
lip sync stays whatever the pretrained model does.
"""
import torch
import torch.nn.functional as F

FPS = 25.0
LIP_KP = [6, 12, 14, 17, 19, 20]
LIP_DIMS = [k * 3 + j for k in LIP_KP for j in range(3)]
POSE_DIMS = [67, 68, 69]

DEFAULT_WEIGHTS = dict(amp=1.0, jitter=0.5, couple=0.5, blink=0.5, gaze=0.3, ood=1.0)


def _band_mask(T, lo, hi, device):
    f = torch.fft.rfftfreq(T, d=1.0 / FPS).to(device)
    return ((f >= lo) & (f <= hi)).float()


def _bandpass(x, lo, hi):
    """x: (B, T, C) -> band-limited signal via FFT masking (differentiable)."""
    T = x.shape[1]
    X = torch.fft.rfft(x - x.mean(1, keepdim=True), dim=1)
    m = _band_mask(T, lo, hi, x.device).view(1, -1, 1)
    return torch.fft.irfft(X * m, n=T, dim=1)


def _pearson(a, b, eps=1e-6):
    a = a - a.mean(-1, keepdim=True)
    b = b - b.mean(-1, keepdim=True)
    return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1) + eps)


def _smoothmax(x, dim, temp=0.05):
    return temp * torch.logsumexp(x / temp, dim=dim) - temp * torch.log(torch.tensor(float(x.shape[dim])))


class LivelinessReward(torch.nn.Module):
    def __init__(self, template: dict, eye_probe=None, weights=None):
        super().__init__()
        self.w = {**DEFAULT_WEIGHTS, **(weights or {})}
        lo = torch.tensor([float(template[f"min_{k}"][0]) for k in ("pitch", "yaw", "roll")])
        hi = torch.tensor([float(template[f"max_{k}"][0]) for k in ("pitch", "yaw", "roll")])
        self.register_buffer("pose_lo", lo)
        self.register_buffer("pose_rng", hi - lo)
        self.register_buffer("axis_w", torch.tensor([1.5, 1.0, 0.7]))
        self.eye_probe = eye_probe
        if eye_probe is not None:
            for p in eye_probe.parameters():
                p.requires_grad_(False)

    def pose_deg(self, coef):
        return coef[..., POSE_DIMS] * self.pose_rng + self.pose_lo

    def forward(self, coef, env=None, return_terms=False):
        """coef: (B,T,73) normalized; env: (B,T) speech loudness (any scale) or None."""
        B, T, _ = coef.shape
        pose = self.pose_deg(coef)
        low = _bandpass(pose, 0.3, 3.0)
        high = _bandpass(pose, 6.0, FPS / 2)
        std_low = (low.pow(2).mean(1) + 1e-6).sqrt()              # (B,3) degrees
        std_high = (high.pow(2).mean(1) + 1e-6).sqrt()
        amp = (torch.tanh(std_low / 3.0) * self.axis_w).sum(-1) / self.axis_w.sum()
        jitter = -(std_high * self.axis_w).sum(-1) / self.axis_w.sum() / 0.5

        terms = {"amp": amp, "jitter": jitter}
        if env is not None:
            speed = (low[:, 1:] - low[:, :-1]).norm(dim=-1) * FPS  # (B,T-1)
            e = env[:, 1:]
            lag = 4  # ~160 ms, head follows speech
            terms["couple"] = _pearson(e[:, : T - 1 - lag], speed[:, lag:])
        else:
            terms["couple"] = torch.zeros(B, device=coef.device)

        if self.eye_probe is not None:
            d_exp = coef[..., :63] - coef[:, :1, :63]
            eye = self.eye_probe(d_exp)
            b = eye["blink"]
            n_w = max(T // 50, 1)
            bw = b[:, : n_w * 50].reshape(B, n_w, 50)
            terms["blink"] = _smoothmax(bw, dim=-1).mean(-1) - 2.0 * F.relu(b.mean(-1) - 0.2)
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
