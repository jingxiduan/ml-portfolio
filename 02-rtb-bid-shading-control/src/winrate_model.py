"""
Simplified reproduction of an omni-channel win-rate (bid landscape) model.

Uses online rolling-updated logistic regression, with features [log(bid), pctr,
log(bid)*pctr] and the label being whether the bid won; refit on the accumulated
samples at the end of each control window, simulating the rolling training of an
online win-rate model.
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
            return  # no positive/negative samples yet, skip this fit

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
        """For each impression, greedily search candidate bid ratios to maximize expected profit."""
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
