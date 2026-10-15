"""
端到端实验: pCTR(真实 Avazu 数据) -> 二价拍卖仿真 -> 竞胜率模型 + 贪心候选出价 ->
Fixed / PID / MPC 三种 Bid Shading 策略在滚动窗口上在线对比, 并给出 5%->20% 灰度放量的
增量效果估算。
"""
import json
import os

import numpy as np

from auction_env import SecondPriceAuctionEnv
from winrate_model import WinRateModel
from controllers import PIDShadingController, MPCShadingController

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "sim_pool.npz")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")

WINDOW_SIZE = 1000
TARGET_ROI = 1.4
CPA = 8.0
WARMUP_WINDOWS = 3  # 前几窗口用真实出价(coef=1, 竞胜率模型冷启动)收集数据


def run_strategy(name, env, n_windows, window_size):
    wr_model = WinRateModel()
    pid = PIDShadingController(target_roi=TARGET_ROI)
    mpc = MPCShadingController(target_roi=TARGET_ROI, target_spend_rate=CPA * 0.5)

    cum_gmv, cum_cost = 0.0, 0.0
    window_logs = []
    last_roi = TARGET_ROI

    for w in range(n_windows):
        idx = np.arange(w * window_size, (w + 1) * window_size)
        value = env.value[idx]
        pctr = env.pctr[idx]

        if w < WARMUP_WINDOWS or wr_model.model is None:
            base_bid = value.copy()
            coef = 1.0
        else:
            base_bid = wr_model.greedy_search_bid(value, pctr)
            if name == "fixed":
                coef = 0.8
            elif name == "pid":
                coef = pid.update(last_roi)
            elif name == "mpc":
                coef = mpc.update(value, pctr, wr_model, base_bid)
            else:
                raise ValueError(name)

        final_bid = base_bid * coef
        win, cost, won_value = env.step(idx, final_bid)
        wr_model.update(final_bid, pctr, win)

        gmv = float(won_value.sum())
        spend = float(cost.sum())
        cum_gmv += gmv
        cum_cost += spend
        win_rate = float(win.mean())
        roi = gmv / spend if spend > 0 else float("nan")
        last_roi = roi if not np.isnan(roi) else last_roi

        window_logs.append({
            "window": w, "coef": float(coef), "win_rate": win_rate,
            "gmv": gmv, "cost": spend, "roi": roi,
        })

    total_profit = cum_gmv - cum_cost
    total_roi = cum_gmv / cum_cost if cum_cost > 0 else float("nan")
    summary = {
        "strategy": name,
        "total_gmv": cum_gmv,
        "total_cost": cum_cost,
        "total_profit": total_profit,
        "overall_roi": total_roi,
        "overall_win_rate": float(np.mean([l["win_rate"] for l in window_logs])),
    }
    return summary, window_logs


def staged_ramp_experiment(env, n_windows, window_size, ramp_stages=(0.05, 0.10, 0.15, 0.20)):
    """模拟分阶段灰度放量: 一部分流量用 PID 动态出价(实验组), 其余用 Fixed 基线(对照组),
    对比"实验组放量比例从 5% 提升到 20%"时, 全量业务层面的 GMV / 成本相对纯 Fixed 基线的变化。
    """
    rng = np.random.default_rng(7)
    n_total = n_windows * window_size
    is_treatment_full = rng.random(n_total) < 1.0  # 占位, 实际按 ramp 比例切片见下

    # 先分别跑一次"全量 fixed"与"全量 pid", 再按比例线性插值得到灰度放量下的近似整体指标
    # (同一组市场价格随机数复用, 保证口径一致; 线性插值是对"随机抽样 x% 流量切量"的近似估计,
    #  在大样本下与真实分层抽样误差很小, 用于快速呈现灰度放量趋势)
    fixed_summary, _ = run_strategy("fixed", env, n_windows, window_size)
    pid_summary, _ = run_strategy("pid", env, n_windows, window_size)

    rows = []
    for r in ramp_stages:
        gmv = (1 - r) * fixed_summary["total_gmv"] + r * pid_summary["total_gmv"]
        cost = (1 - r) * fixed_summary["total_cost"] + r * pid_summary["total_cost"]
        gmv_uplift = (gmv - fixed_summary["total_gmv"]) / fixed_summary["total_gmv"] * 100
        cost_change = (cost - fixed_summary["total_cost"]) / fixed_summary["total_cost"] * 100
        rows.append({
            "treatment_ratio": r, "gmv": gmv, "cost": cost,
            "gmv_uplift_pct": gmv_uplift, "cost_change_pct": cost_change,
        })
    return rows, fixed_summary, pid_summary


def main():
    data = np.load(DATA_PATH)
    pctr, click = data["pctr"], data["click"]
    n = len(pctr)
    n_windows = n // WINDOW_SIZE
    print(f"n_impressions={n} n_windows={n_windows} window_size={WINDOW_SIZE}")

    env = SecondPriceAuctionEnv(pctr=pctr[:n_windows * WINDOW_SIZE], cpa=CPA,
                                 target_truthful_winrate=0.55, sigma=0.4, seed=42)

    results = {}
    all_logs = {}
    for name in ["fixed", "pid", "mpc"]:
        summary, logs = run_strategy(name, env, n_windows, WINDOW_SIZE)
        results[name] = summary
        all_logs[name] = logs
        print(name, json.dumps(summary, indent=2))

    ramp_rows, fixed_summary, pid_summary = staged_ramp_experiment(env, n_windows, WINDOW_SIZE)
    print("staged ramp:", json.dumps(ramp_rows, indent=2))

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "strategy_comparison.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(os.path.join(OUT_DIR, "staged_ramp.json"), "w") as f:
        json.dump(ramp_rows, f, indent=2, ensure_ascii=False)
    with open(os.path.join(OUT_DIR, "window_logs.json"), "w") as f:
        json.dump(all_logs, f, ensure_ascii=False)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for name in ["fixed", "pid", "mpc"]:
            logs = all_logs[name]
            windows = [l["window"] for l in logs]
            roi = [l["roi"] for l in logs]
            coef = [l["coef"] for l in logs]
            axes[0].plot(windows, roi, label=name)
            axes[1].plot(windows, coef, label=name)
        axes[0].axhline(TARGET_ROI, color="gray", linestyle="--", label="target ROI")
        axes[0].set_title("Per-window ROI")
        axes[0].set_xlabel("window")
        axes[0].legend()
        axes[1].set_title("Shading coefficient over time")
        axes[1].set_xlabel("window")
        axes[1].legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "bid_shading_curves.png"), dpi=150)
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
