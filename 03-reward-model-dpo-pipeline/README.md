# 偏好数据构建 + Reward Model + DPO Pipeline

用公开偏好数据集复现"人类偏好数据构建 → LLM-as-a-Judge 二次判别 → Reward Model 蒸馏 → DPO 训练数据产出"的完整链路。

> 说明：简历原项目面向数字人视频生成场景，使用 Qwen-VL 多模态表征 + 付费 GPT-4o API 作为 LLM-as-a-Judge。公开 GitHub 项目不便获取数字人视频偏好数据，也不便内置付费 API Key，因此：
> - 数据集替换为 [Anthropic/hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf)（`helpful-base` 子集，Hugging Face 官方数据集，无需登录），结构上与原场景完全一致：同一上下文下，人类标注的 chosen/rejected 两条候选回复；
> - Qwen-VL 多模态表征替换为 `sentence-transformers`（all-MiniLM-L6-v2）句向量 + 手工质量特征；
> - LLM-as-a-Judge 默认使用免 API Key 的本地代理裁判（`LocalProxyJudge`），同时保留真实 rubric prompt 模板与 `OpenAIJudge` 接口，配置 `OPENAI_API_KEY` 后可无缝切换为真实 GPT 系列模型。

## 对应简历工作

1. **偏好数据构建**：融合特征置信度与 LLM-as-a-Judge，对模糊样本做 RLAIF 二次判别，结合置信度筛选提纯金标数据
2. **Reward Model**：以语义表征 + 结构化质量特征为输入，拟合 Bradley-Terry 偏好模型，产出可独立部署、远低于直接调用 LLM Judge 延迟的 Reward Signal
3. **DPO 训练数据产出**：导出标准 `{prompt, chosen, rejected}` 格式，可直接喂给 `trl.DPOTrainer`

## 代码结构

```
03-reward-model-dpo-pipeline/
├── src/
│   ├── download_data.py          # 下载 Anthropic/hh-rlhf(helpful-base), 无需登录
│   ├── features.py                # 句向量 + 结构化质量特征(相关度/多样性/重复率/交叉特征)
│   ├── llm_judge.py                # RubricPromptTemplate + LocalProxyJudge + OpenAIJudge
│   ├── build_preference_dataset.py # 模糊样本检测 -> judge 二次判别 -> 置信度筛选金标集
│   ├── reward_model.py             # 线性 Bradley-Terry Reward Model, 5折CV + 延迟基准
│   ├── build_dpo_dataset.py        # 导出 DPO 标准格式数据 + 数据漏斗统计
│   └── dpo_train_demo.py           # trl.DPOTrainer 端到端冒烟测试(默认 tiny-gpt2, CPU 可跑)
└── outputs/
```

## 运行

```bash
pip install -r requirements.txt
python src/download_data.py
python src/build_preference_dataset.py
python src/reward_model.py
python src/build_dpo_dataset.py
python src/dpo_train_demo.py   # 可选: 验证 DPO 训练链路端到端可跑通
```

## 实测结果

**偏好数据提纯漏斗**（train 集 7,980 条原始 chosen/rejected 对）：

| 阶段 | 数量 | 占比 |
|---|---|---|
| 原始偏好对 | 7,980 | 100% |
| 其中"模糊样本"（语义相关度难以区分，送入 LLM-as-a-Judge 二次判别） | 1,029 | 12.9% |
| 经 Judge 判别为噪声标签（与人工标注方向相反，剔除） | 482 | 6.0% |
| 最终金标数据（按置信度排序取 Top 35%，用于 DPO 训练） | 2,624 | 32.9% |

**Reward Model**（句向量 PCA-40 维 + 10 维结构化特征，线性 Bradley-Terry 模型，5 折交叉验证）：

| 指标 | 数值 |
|---|---|
| 5折 AUC（均值±标准差） | 0.6072 ± 0.0029 |
| 5折 Pairwise Accuracy | 0.6267 |
| 独立测试集 AUC | 0.6150 |
| 独立测试集 Pairwise Accuracy | 0.6317 |

> 说明：生产环境用 Qwen-VL 多模态表征在数字人视频质量任务上报告 AUC=0.76；本项目用通用文本句向量在 hh-rlhf 这类"措辞细微差异"偏好数据集上复现，AUC 明显偏低是预期之中的——这类数据集上，人类标注者之间的一致性本身就有限，用轻量特征拟合更是进一步限制了上限。**更有工程价值的结论是下面的推理延迟对比**。

**推理延迟对比**（300 条样本，Reward Model vs 逐条调用 LLM Judge）：

| 方案 | 单条耗时 | 总耗时(300条) |
|---|---|---|
| LLM-as-a-Judge（句向量裁判，含 encode） | 16.17 ms | 4.85 s |
| Reward Model（预计算特征上的线性前向） | 0.00097 ms | 0.00029 s |
| **加速比** | **≈16,700x** | - |

与简历中"相较直接调用 LLM Judge，推理延迟降低到 1/15、成本降低 60%"方向一致：把"每条样本都要调一次 LLM/Judge"的流程，蒸馏成一个轻量级 Reward Model 之后，推理成本可以降低几个数量级，从而让 Reward Model 能够应用于全量数据筛选，而不仅仅是训练阶段用到的那一小部分样本。

**DPO 数据**：`outputs/dpo_train.jsonl`（2,624 条）、`outputs/dpo_test.jsonl`（497 条），标准 `{prompt, chosen, rejected}` 格式。`dpo_train_demo.py` 已验证可直接喂给 `trl.DPOTrainer` 跑通训练循环（默认用 `sshleifer/tiny-gpt2` 做 20 step 的 CPU smoke test，替换为更大模型 + GPU 环境即可用于正式训练）。
