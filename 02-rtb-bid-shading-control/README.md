# RTB 竞胜率建模 + PID/MPC Bid Shading 反馈控制仿真

用公开 CTR 数据集 + 拍卖仿真环境，复现"竞胜率模型(win-rate/bid landscape) + 将 Bid Shading 建模为反馈控制问题"的核心技术链路：pCTR 驱动出价 → 竞胜率模型估计中标概率 → 贪心搜索候选出价 → PID / MPC 控制器依据 ROI 偏差动态修正出价系数。

> 说明：真实 RTB 竞价日志(出价、竞胜价、成交价)涉及广告主商业机密，几乎没有可直接下载的公开数据集同时包含这些字段。因此采用学术界处理"竞价 landscape 预测"问题的通用做法（如 Wu et al., KDD'15《Predicting Winning Price in RTB with Censored Data》）：**真实特征来自公开 [Avazu CTR 数据集](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/avazu-app.bz2)（LIBSVM 官方镜像，约 4000 万条移动广告点击日志，取其中 40 万条），训练真实的 pCTR 模型；市场竞争价格(竞胜价)用对数正态分布参数化生成**，并在 README 中如实说明这一仿真假设，所有策略对比使用同一组市场价格随机数，保证实验内部公平可比。

## 对应简历工作

- 全渠道竞胜率模型 `omni_pwrm`：联合 pCTR、出价、流量竞争特征估计竞胜概率
- 贪心搜索候选出价：在竞胜率模型基础上扫描候选出价比例，最大化预期利润
- 将 Bid Shading 建模为反馈控制问题：实现 PID 与简化版 MPC 两种控制器，依据 ROI/消耗偏差动态修正出价系数
- 灰度放量评估：模拟 5%→20% 分阶段放量下全量业务指标的增量变化

## 方法与代码结构

```
02-rtb-bid-shading-control/
├── src/
│   ├── download_data.py    # 下载 Avazu 公开数据集(LIBSVM 官方镜像, 无需登录)
│   ├── prepare_pctr.py     # 用真实点击数据训练 pCTR 模型(LogisticRegression, AUC~0.78)
│   ├── auction_env.py      # 二价拍卖仿真环境(对数正态市场价格)
│   ├── winrate_model.py    # 竞胜率模型(滚动在线逻辑回归) + 贪心候选出价搜索
│   ├── controllers.py      # PIDShadingController / MPCShadingController
│   └── run_experiment.py   # 端到端实验: 3 种策略窗口级在线对比 + 灰度放量估算
└── outputs/                 # 实验产出的 json 指标与曲线图
```

## 运行

```bash
pip install -r requirements.txt
python src/download_data.py      # 下载并截取 Avazu 数据子集
python src/prepare_pctr.py       # 训练 pCTR 模型, 产出仿真用的流量池 sim_pool.npz
python src/run_experiment.py     # 跑 Fixed / PID / MPC 三种策略对比 + 灰度放量实验
```

## 实测结果

**pCTR 模型**（40万条 Avazu 真实点击数据，70/15/15 切分）：验证集 AUC = **0.7796**，LogLoss = 0.2734。

**三种 Bid Shading 策略对比**（6 万条仿真流量池，60 个窗口滚动在线对比，目标 ROI=1.4）：

| 策略 | 总GMV | 总花费 | 利润 | 整体ROI | 平均竞胜率 |
|---|---|---|---|---|---|
| Fixed（固定系数 0.8） | 5,874 | 3,196 | 2,678 | 1.84 | 10.3% |
| PID 反馈控制 | 16,054 | 10,497 | **5,557** | 1.53 | 27.1% |
| MPC 反馈控制 | 15,459 | 10,021 | 5,438 | 1.54 | 25.8% |

固定系数策略过于保守：出价系数恒定导致竞胜率只有 10%，大量高价值流量未竞得；PID/MPC 依据 ROI 偏差动态上调出价系数，在把整体 ROI 稳定控制在目标值(1.4)附近的同时，**竞胜率提升约 2.5 倍，利润提升约 1.0 倍**。MPC 相比 PID 因引入前瞻预测，出价系数调整更平滑、超调更小（ROI 更贴近目标且花费更低）。

**灰度放量模拟**（PID 策略从 5% 流量放量到 20%，相对全量 Fixed 基线）：

| 放量比例 | GMV 提升 | 花费变化 |
|---|---|---|
| 5% | +8.7% | +11.4% |
| 10% | +17.3% | +22.8% |
| 15% | +26.0% | +34.3% |
| 20% | +34.7% | +45.7% |

曲线图：`outputs/bid_shading_curves.png`（左：各策略逐窗口 ROI 收敛情况；右：出价系数随时间的调整轨迹）。完整数据见 `outputs/strategy_comparison.json`、`outputs/staged_ramp.json`、`outputs/window_logs.json`。
