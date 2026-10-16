"""
Second-price auction simulation environment.

Real RTB bidding logs typically involve advertisers' commercial secrets, and public
datasets rarely provide the "bid-winning price-win or not" triple directly. This project
therefore follows the standard academic approach for "bid landscape prediction" (e.g. Wu
et al., KDD'15, *Predicting Winning Price in RTB with Censored Data*): market competing
prices are parameterized via a log-normal distribution, layered on top of real pCTR from
a public CTR dataset (Avazu), to generate "bid-win-price" data for win-rate modeling and
bidding strategy evaluation.
"""
import numpy as np


class SecondPriceAuctionEnv:
    def __init__(self, pctr: np.ndarray, cpa: float = 8.0,
                 target_truthful_winrate: float = 0.55, sigma: float = 0.4, seed: int = 42):
        self.pctr = pctr
        self.value = pctr * cpa  # expected value per impression (eCPM proxy, e.g. pLTV*CPA for game ads)
        self.sigma = sigma
        self.n = len(pctr)

        from scipy.stats import norm
        offset = -sigma * norm.ppf(target_truthful_winrate)
        mu = np.log(self.value + 1e-6) + offset

        rng = np.random.default_rng(seed)
        z = rng.normal(0, 1, size=self.n)
        self.market_price = np.exp(mu + sigma * z)  # fixed seed, reused across strategies for a fair comparison

    def step(self, idx: np.ndarray, bid: np.ndarray):
        """Given a batch of impression indices and bids, returns whether won, clearing cost (second price), and expected value."""
        m = self.market_price[idx]
        win = bid > m
        cost = np.where(win, m, 0.0)          # second-price billing: winner only pays the second-highest price
        value = np.where(win, self.value[idx], 0.0)
        return win, cost, value
