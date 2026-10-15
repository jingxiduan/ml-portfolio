# Multi-Task CTR/CVR Recommendation with RankMixer-style Dense Interaction

用公开数据集复现"多任务塔式建模 + 稠密特征交互(RankMixer)"推荐场景：共享底层表征，上层按任务路由到 CTR / CVR 两个塔，CVR 分支通过 **sample mask** 只在点击样本上回传梯度，复现多任务推荐系统中"曝光-点击-转化"链路依赖问题的经典解法（ESMM 思路）。

> 说明：公开数据集 MovieLens 本身没有真实的"曝光/点击/转化"三段式广告日志，因此这里用业界常见的代理标签方式构造任务：**评分过的影片 = 点击(CTR=1)，未评分的影片做负采样 = 未点击(CTR=0)；点击样本中 rating≥4 视为"转化/满意"(CVR=1)**。这是公开数据集上复现工业级多任务 CTR/CVR 场景时的标准做法，所有评测指标均基于该代理标签计算。

## 对应简历工作

- 多任务塔式建模与特征工程：共享底层 Embedding + 分支任务塔 + sample mask 梯度隔离
- RankMixer 稠密特征交互：用 MLP-Mixer 风格的 token-mixing / channel-mixing 替代传统 FM/DCN 显式交叉，把预估模型从稀疏查表主导改造为稠密计算主导
- 离线评估：GAUC（分组 AUC）、AUC、LogLoss、分桶校准误差（AbsBias）

## 数据集

[MovieLens-1M](https://grouplens.org/datasets/movielens/1m/)，官方直链下载，无需注册/登录，约 100 万条评分记录、6040 个用户、3952 部影片。

```bash
python src/download_data.py
```

## 方法

1. **样本构造**（`src/preprocess.py`）：正样本=用户真实评分记录，负样本=按用户未交互影片负采样（比例 1:2）；特征包括用户画像（性别/年龄桶/职业）、用户历史行为统计（历史均分/评分数）、影片画像（类型 one-hot/热度统计）。
2. **模型**（`src/model.py`，`RankMixerMTL`）：
   - 每个稀疏特征独立 Embedding，稠密特征通过线性层投影为同维 token，拼成 `(B, T, D)` 的 token 序列；
   - 堆叠 `MixerBlock`：LayerNorm → token-mixing MLP（跨特征维度交互）→ 残差 → LayerNorm → channel-mixing MLP（跨 embedding 通道交互）→ 残差；
   - 共享表征展平后分别接 CTR Tower 与 CVR Tower（两层 MLP + sigmoid）。
3. **损失**：`loss = BCE(ctr_logit, click) + mask * BCE(cvr_logit, convert)`，`mask` 来自未点击样本的 `label_convert=-1` 占位，仅点击样本参与 CVR 梯度回传。

## 运行

```bash
pip install -r requirements.txt
python src/download_data.py
python src/preprocess.py
python src/train.py --epochs 4 --batch_size 4096
```

## 实测结果（4 epoch, CPU, ~3 分钟）

| 任务 | AUC | GAUC | LogLoss | AbsBias(校准误差) |
|---|---|---|---|---|
| CTR（点击） | 0.8805 | 0.8796 | 0.4035 | 0.0046 |
| CVR（点击样本内的转化） | 0.7960 | - | 0.5376 | 0.0305 |

训练曲线见 `outputs/training_curve.png`，完整指标见 `outputs/metrics.json`。

```
[epoch 1] train_loss=1.0003 val_ctr_auc=0.8686 val_cvr_auc=0.7874
[epoch 2] train_loss=0.9643 val_ctr_auc=0.8733 val_cvr_auc=0.7909
[epoch 3] train_loss=0.9523 val_ctr_auc=0.8777 val_cvr_auc=0.7940
[epoch 4] train_loss=0.9414 val_ctr_auc=0.8818 val_cvr_auc=0.7962
```

## 目录结构

```
01-multitask-ctr-cvr-rec/
├── src/
│   ├── download_data.py   # 下载 MovieLens-1M
│   ├── preprocess.py      # 构造 CTR/CVR 多任务样本
│   ├── dataset.py         # PyTorch Dataset
│   ├── model.py           # RankMixer 风格多任务模型
│   ├── metrics.py         # GAUC / AUC / 校准误差
│   └── train.py           # 训练 + 评估 + 画图
├── outputs/                # 训练产出的 metrics.json / training_curve.png
└── requirements.txt
```
