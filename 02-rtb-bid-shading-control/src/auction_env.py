"""
二价拍卖(second-price auction)仿真环境。

真实 RTB 竞价日志通常涉及广告主商业机密, 公开数据集里极少直接提供
"出价-竞胜价-是否竞胜"三元组, 因此采用学术界处理"竞价 landscape 预测"
问题时的通用做法(如 Wu et al., KDD'15《Predicting Winning Price in RTB
with Censored Data》): 用对数正态分布参数化市场竞争价格, 在真实公开
CTR 数据集(Avazu)提供的 pCTR 基础上做拍卖仿真, 从而生成"出价-竞胜-
价格"数据用于竞胜率建模与出价策略评估。
"""
import numpy as np


class SecondPriceAuctionEnv:
    def __init__(self, pctr: np.ndarray, cpa: float = 8.0,
                 target_truthful_winrate: float = 0.55, sigma: float = 0.4, seed: int = 42):
        self.pctr = pctr
        self.value = pctr * cpa  # 每条曝光的期望价值(eCPM 代理, 例如游戏广告的 pLTV*CPA)
        self.sigma = sigma
        self.n = len(pctr)

        from scipy.stats import norm
        offset = -sigma * norm.ppf(target_truthful_winrate)
        mu = np.log(self.value + 1e-6) + offset

        rng = np.random.default_rng(seed)
        z = rng.normal(0, 1, size=self.n)
        self.market_price = np.exp(mu + sigma * z)  # 固定随机种子, 跨策略复用, 保证公平对比

    def step(self, idx: np.ndarray, bid: np.ndarray):
        """给定一批曝光下标与出价, 返回是否竞胜、成交成本(二价)、期望价值。"""
        m = self.market_price[idx]
        win = bid > m
        cost = np.where(win, m, 0.0)          # 二价计费: 竞胜只需支付次高价
        value = np.where(win, self.value[idx], 0.0)
        return win, cost, value
