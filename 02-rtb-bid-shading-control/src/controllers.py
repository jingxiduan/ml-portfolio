"""
Bid Shading 反馈控制器: PID 与简化版 MPC。

对应简历: "将 Bid Shading 建模为反馈控制问题, 采用 PID 与 MPC 依据利润率、ROI 及消耗
偏差动态修正系数"。

两者都输出一个全局 shading_coef(在贪心搜索给出的候选出价基础上整体缩放), 用于把
实际 ROI / 消耗 pacing 拉向目标值, 同时尽量保住 GMV/win-rate。
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
        error = self.target_roi - realized_roi  # ROI 偏低(花多赚少) -> error>0 -> 应该把系数调小(更抠)
        self.integral += error
        derivative = error - self.prev_error
        self.prev_error = error

        # error>0 表示 ROI 低于目标, 需要降低出价系数(更省钱, 提升ROI)
        delta = -(self.kp * error + self.ki * self.integral + self.kd * derivative)
        self.coef = float(np.clip(self.coef + delta, self.coef_min, self.coef_max))
        return self.coef


class MPCShadingController:
    """单步前瞻的简化 MPC: 在候选系数网格上, 用当前窗口已知的 pctr/value 分布 + 当前
    竞胜率模型, 前瞻预测下一窗口在各候选系数下的 ROI 与花费, 选择使代价函数最小的系数。
    """

    def __init__(self, target_roi: float, target_spend_rate: float,
                 candidate_coefs=np.linspace(0.5, 1.3, 17),
                 roi_weight: float = 1.0, spend_weight: float = 0.5,
                 coef_min=0.5, coef_max=1.3):
        self.target_roi = target_roi
        self.target_spend_rate = target_spend_rate  # 每条曝光的目标平均花费, 用于预算 pacing
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
            exp_cost = wr * bid_c  # 用出价近似二价成交成本的期望上界, 仅用于前瞻决策
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
