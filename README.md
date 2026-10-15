# 算法工程师作品集：公开数据集复现版

简历中的工程项目涉及内部数据与商业机密，原仓库已丢失。这里用**完全公开的数据集**对简历中最具代表性的三类算法工作做了可运行、可验证的复现，重点还原**方法论与系统设计**，而非照搬原始业务数字。每个子项目都可独立 `pip install -r requirements.txt` 后一键跑通，README 中的指标均为本次实测结果（非编造）。

## 子项目一览

| 项目 | 对应简历方向 | 公开数据集 | 核心技术点 |
|---|---|---|---|
| [`01-multitask-ctr-cvr-rec`](01-multitask-ctr-cvr-rec) | 多任务塔式推荐建模 / RankMixer 稠密特征交互 | [MovieLens-1M](https://grouplens.org/datasets/movielens/1m/) | 共享 Embedding + MLP-Mixer 风格 token/channel-mixing、CTR/CVR 双塔、sample mask 梯度隔离、GAUC/校准误差评估 |
| [`02-rtb-bid-shading-control`](02-rtb-bid-shading-control) | 竞胜率模型 `omni_pwrm` / Bid Shading 反馈控制 | [Avazu CTR Dataset](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/avazu-app.bz2)（LIBSVM 镜像）+ 拍卖仿真 | 真实 pCTR 建模、二价拍卖仿真、在线竞胜率模型、贪心候选出价搜索、PID/MPC 反馈控制对比 |
| [`03-reward-model-dpo-pipeline`](03-reward-model-dpo-pipeline) | 偏好数据构建 / Reward Model / DPO 训练数据产出 | [Anthropic/hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf)（helpful-base） | LLM-as-a-Judge 二次判别、置信度筛选金标数据、Bradley-Terry Reward Model、`trl.DPOTrainer` 端到端打通 |

## 如何看待"复现"与"原项目"的差异

公开数据集天然不包含真实广告竞价日志、数字人视频偏好标注、内部多模态表征等商业机密数据，因此每个子项目 README 开头都如实说明了：

1. 用什么公开数据集替代了什么原始数据；
2. 哪些是业界通用的标准处理手法（如用对数正态分布仿真拍卖市场价格、用 ESMM 式代理标签构造多任务样本）；
3. 实测指标与原简历数字的差异及原因。

目的是**诚实地展示方法论的工程实现能力**，而不是伪造一份"看起来和原简历一样"的结果。

## 快速开始

```bash
git clone <this-repo>
cd algo-portfolio
python3 -m venv .venv && source .venv/bin/activate

cd 01-multitask-ctr-cvr-rec && pip install -r requirements.txt && python src/download_data.py && python src/preprocess.py && python src/train.py --epochs 4

cd ../02-rtb-bid-shading-control && pip install -r requirements.txt && python src/download_data.py && python src/prepare_pctr.py && python src/run_experiment.py

cd ../03-reward-model-dpo-pipeline && pip install -r requirements.txt && python src/download_data.py && python src/build_preference_dataset.py && python src/reward_model.py && python src/build_dpo_dataset.py
```

## License

MIT，详见 [LICENSE](LICENSE)。
