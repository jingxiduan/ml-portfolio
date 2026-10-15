"""
全渠道竞胜率模型(win-rate / bid landscape model)的简化复现。

对应简历: "搭建全渠道竞胜率模型 omni_pwrm, 联合 pCTR、出价与流量竞争特征估计竞胜概率,
经贪心搜索生成候选出价"。

这里用在线滚动更新的逻辑回归, 特征为 [log(bid), pctr, log(bid)*pctr], 标签为是否竞胜,
每个控制窗口结束后用累计样本重新拟合, 模拟线上竞胜率模型的滚动训练。
"""
import numpy as np
from sklearn.linear_model import LogisticRegression


class WinRateModel:
    def __init__(self):
        self.model = None
        self.bids_hist = []
        self.pctr_hist = []
        self.win_hist = []

    @staticmethod
    def _features(bid: np.ndarray, pctr: np.ndarray):
        logb = np.log(np.clip(bid, 1e-3, None))
        return np.stack([logb, pctr, logb * pctr], axis=1)

    def update(self, bid: np.ndarray, pctr: np.ndarray, win: np.ndarray):
        self.bids_hist.append(bid)
        self.pctr_hist.append(pctr)
        self.win_hist.append(win.astype(int))

        all_bid = np.concatenate(self.bids_hist)
        all_pctr = np.concatenate(self.pctr_hist)
        all_win = np.concatenate(self.win_hist)

        if len(np.unique(all_win)) < 2:
            return  # 还没有正负样本, 跳过本次拟合

        X = self._features(all_bid, all_pctr)
        self.model = LogisticRegression(max_iter=200)
        self.model.fit(X, all_win)

    def predict_winrate(self, bid: np.ndarray, pctr: np.ndarray) -> np.ndarray:
        if self.model is None:
            return np.full(len(bid), 0.5)
        X = self._features(bid, pctr)
        return self.model.predict_proba(X)[:, 1]

    def greedy_search_bid(self, value: np.ndarray, pctr: np.ndarray,
                           candidate_ratios=np.linspace(0.3, 1.3, 11)) -> np.ndarray:
        """对每条曝光, 在候选出价比例上贪心搜索使预期利润最大的出价。"""
        best_bid = value.copy()
        best_profit = np.full(len(value), -np.inf)
        for r in candidate_ratios:
            bid_r = value * r
            wr = self.predict_winrate(bid_r, pctr)
            profit_r = wr * (value - bid_r)
            improve = profit_r > best_profit
            best_bid = np.where(improve, bid_r, best_bid)
            best_profit = np.where(improve, profit_r, best_profit)
        return best_bid
