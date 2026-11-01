"""Reward v5 = v4 + ONE new term: `expr` (expression dynamics).

Why this exists
---------------
v4's six terms touch essentially three numbers: the low- and high-frequency
energy of pitch/yaw/roll. `amp / jitter / couple` are all pose-only, `ood`
is a bound, and `blink` / `gaze` go through the eye probe. The 63 expression
dimensions -- brow, cheek, jaw, mouth corner -- have **no reward term at all**,
so OPSD leaves them wherever the frozen behaviour policy puts them: the face
is a static mask inherited from the source portrait, and the persona is
literally "base + slightly more head wobble" (finance measured at 1.31x base).

`expr` fixes exactly that, and nothing else.

Design rule (attribution)
-------------------------
`amp / jitter / couple / blink / gaze / ood` keep v4's weights and formulas
byte-for-byte. `expr` is added on top as a *peaked* Gaussian -- NOT a
monotone term -- because monotone amplitude rewards are what produced the
2.4-2.7x uncanny overshoot in v2/v3.

Units
-----
`coef[..., :63]` is normalised expression: `exp = coef * std_exp + mean_exp`
(see `engine.denorm_pose`). So `expr_target` must be measured in the same
normalised units, by running LivePortrait's motion extractor over real video
and normalising with the same template -- see
`scripts/calibrate_behaviour_bands.py`.
"""
import torch

from motion_reward import LIP_DIMS, _bandpass, _pearson  # noqa: F401
from motion_reward_v4 import LivelinessRewardV4

FPS = 25.0
EXP_DIM = 63
NON_LIP = [i for i in range(EXP_DIM) if i not in LIP_DIMS]


class LivelinessRewardV5(LivelinessRewardV4):
    """v4 + peaked expression-dynamics term.

    expr = exp(-0.5 * ((std_low(non-lip exp) - target) / tol)^2)

    Optionally (off by default) a second term rewarding the coupling between
    speech loudness and expression speed, so the face moves *with* the voice
    instead of oscillating at a constant rate.
    """

    def __init__(self, *args, expr_target: float = 1.0, expr_tol: float = 1.0,
                 expr_weight: float = 1.0, expr_env_weight: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.w["expr"] = float(expr_weight)
        self.w["expr_env"] = float(expr_env_weight)
        self.expr_target = float(expr_target)
        self.expr_tol = float(expr_tol)
        self._last_expr_active_frac = 0.0

    def _expr_terms(self, coef, env=None):
        exp = coef[..., NON_LIP]
        low = _bandpass(exp, 0.3, 3.0)                   # same filter as pose
        std_low = (low.pow(2).mean(1) + 1e-6).sqrt()     # (B, |NON_LIP|)
        m = std_low.mean(-1)
        dev = (m - self.expr_target) / self.expr_tol
        expr = torch.exp(-0.5 * dev * dev)
        # how often the target band is actually biting (>0 == gradient exists)
        self._last_expr_active_frac = float(
            ((m > self.expr_target - self.expr_tol) &
             (m < self.expr_target + self.expr_tol)).float().mean().cpu())
        out = {"expr": expr}
        if self.w.get("expr_env", 0.0) and env is not None:
            B, T, _ = coef.shape
            speed = (low[:, 1:] - low[:, :-1]).norm(dim=-1) * FPS
            lag = 2
            out["expr_env"] = _pearson(env[:, 1: T - lag], speed[:, lag:])
        return out

    def forward(self, coef, env=None, return_terms=False):
        total, terms = super().forward(coef, env, return_terms=True)
        new = self._expr_terms(coef, env)
        for k, v in new.items():
            terms[k] = v.detach()
            total = total + self.w[k] * v
        if return_terms:
            return total, terms
        return total
