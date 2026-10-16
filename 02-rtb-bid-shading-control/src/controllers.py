"""
Bid Shading feedback controllers: PID and a simplified MPC.

Both controllers output a global shading_coef (scaling the candidate bid produced by the
greedy search) used to pull realized ROI / spend pacing toward the target value while
preserving GMV / win-rate as much as possible.
"""
import numpy as np


class PIDShadingController:
    def __init__(self, target_roi: float, kp=0.5, ki=0.1, kd=0.05,
                 coef_min=0.5, coef_max=1.3, init_coef=1.0):
        self.target_roi = target_roi
        self.kp, self.ki, self.kd = kp, ki, kd
        self.coef_min, self.coef_max = coef_min, coef_max
        self.coef = init_coef
        self.integral = 0.0
        self.prev_error = 0.0

    def update(self, realized_roi: float) -> float:
        error = self.target_roi - realized_roi  # ROI below target (spending more, earning less) -> error>0 -> coefficient should decrease (be more conservative)
        self.integral += error
        derivative = error - self.prev_error
        self.prev_error = error

        # error>0 means ROI is below target, so the bid coefficient should be lowered (save money, raise ROI)
        delta = -(self.kp * error + self.ki * self.integral + self.kd * derivative)
        self.coef = float(np.clip(self.coef + delta, self.coef_min, self.coef_max))
        return self.coef


class MPCShadingController:
    """A single-step-lookahead simplified MPC: over a grid of candidate coefficients, uses the
    current window's known pctr/value distribution plus the current win-rate model to predict
    the ROI and spend of the next window under each candidate coefficient, and picks the one
    minimizing the cost function.
    """

    def __init__(self, target_roi: float, target_spend_rate: float,
                 candidate_coefs=np.linspace(0.5, 1.3, 17),
                 roi_weight: float = 1.0, spend_weight: float = 0.5,
                 coef_min=0.5, coef_max=1.3):
        self.target_roi = target_roi
        self.target_spend_rate = target_spend_rate  # target average spend per impression, used for budget pacing
        self.candidate_coefs = candidate_coefs
        self.roi_weight = roi_weight
        self.spend_weight = spend_weight
        self.coef_min, self.coef_max = coef_min, coef_max
        self.coef = 1.0

    def update(self, lookahead_value: np.ndarray, lookahead_pctr: np.ndarray,
               winrate_model, base_bid: np.ndarray) -> float:
        best_coef, best_cost = self.coef, float("inf")
        for c in self.candidate_coefs:
            bid_c = base_bid * c
            wr = winrate_model.predict_winrate(bid_c, lookahead_pctr)
            exp_value = wr * lookahead_value
            exp_cost = wr * bid_c  # uses the bid itself to approximate the expected upper bound of second-price cost, for lookahead decisions only
            gmv = exp_value.sum()
            cost = exp_cost.sum() + 1e-6
            pred_roi = gmv / cost
            pred_spend_rate = cost / len(lookahead_value)

            cost_fn = (self.roi_weight * (pred_roi - self.target_roi) ** 2 +
                       self.spend_weight * (pred_spend_rate - self.target_spend_rate) ** 2)
            if cost_fn < best_cost:
                best_cost = cost_fn
                best_coef = c

        self.coef = float(np.clip(best_coef, self.coef_min, self.coef_max))
        return self.coef
