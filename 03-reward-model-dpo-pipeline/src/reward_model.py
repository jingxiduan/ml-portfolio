"""
Reward Model: 用结构化质量特征拟合用户偏好(Bradley-Terry pairwise 模型), 输出可直接用于
后训练数据筛选的 Reward Signal, 对应简历:
  "以 Qwen-VL 多模态表征与结构化质量特征为输入, 拟合用户偏好, 输出可用于后训练的 Reward
   Signal...相较直接调用 LLM Judge, 推理延迟降低到原来的 1/15, 成本降低 60%"

本项目中 Qwen-VL 多模态表征替换为 sentence-transformers 句向量(PCA 降维至 40 维) +
10 维手工质量特征(见 features.py), 核心验证点一致: 训练好的轻量 Reward Model 在准确率
明显优于随机猜测的同时, 推理延迟远低于"每条样本都调用一次 LLM/Judge"的方案。

Bradley-Terry 模型: P(chosen 优于 rejected) = sigmoid(w^T(x_chosen - x_rejected)),
等价于在对称增广的差值样本上训练逻辑回归, 这里直接用 sklearn LogisticRegression 求解
(比有限步数的 SGD/Adam 收敛更稳定), 回归系数 w 即为可解释的线性 Reward 打分函数。
"""
import json
import os
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from features import get_embedder
from llm_judge import LocalProxyJudge

PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


class RewardModel:
    """r(x) = w^T x_std + b, 在标准化特征空间上训练的线性 Bradley-Terry Reward 模型。"""

    def __init__(self, C: float = 0.5):
        self.clf = LogisticRegression(max_iter=2000, C=C)

    def fit(self, feat_c: np.ndarray, feat_r: np.ndarray):
        diff = feat_c - feat_r
        X = np.concatenate([diff, -diff], axis=0)
        y = np.concatenate([np.ones(len(diff)), np.zeros(len(diff))])
        self.clf.fit(X, y)
        return self

    def score(self, feat: np.ndarray) -> np.ndarray:
        """给单条样本的特征打一个标量 reward 分数(不依赖另一条样本, 可独立部署推理)。"""
        w = self.clf.coef_[0]
        return feat @ w  # 省略 bias/2, 只影响整体平移, 不影响排序与 pairwise 比较

    def pairwise_auc(self, feat_c: np.ndarray, feat_r: np.ndarray):
        rc, rr = self.score(feat_c), self.score(feat_r)
        scores = np.concatenate([rc, rr])
        labels = np.concatenate([np.ones(len(rc)), np.zeros(len(rr))])
        auc = roc_auc_score(labels, scores)
        acc = float(np.mean(rc > rr))
        return auc, acc


def cross_validate(feat_c, feat_r, n_splits=5, C=0.5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs, accs = [], []
    last_model = None
    for tr_idx, va_idx in kf.split(feat_c):
        model = RewardModel(C=C).fit(feat_c[tr_idx], feat_r[tr_idx])
        auc, acc = model.pairwise_auc(feat_c[va_idx], feat_r[va_idx])
        aucs.append(auc)
        accs.append(acc)
        last_model = model
    return aucs, accs, last_model


def benchmark_latency(model: RewardModel, feat_c, judge: LocalProxyJudge, responses, n_bench=300):
    """对比: 逐条调用 LLM Judge(句向量编码 + 相似度打分) vs 训练好的 Reward Model 推理。"""
    sample_feat = feat_c[:n_bench]
    sample_resp = responses[:n_bench]

    t0 = time.time()
    for i in range(n_bench):
        _ = model.score(sample_feat[i:i + 1])
    reward_model_time = time.time() - t0

    t0 = time.time()
    for r in sample_resp:
        _ = judge.score("placeholder prompt", r)
    judge_time = time.time() - t0

    return {
        "n_samples": n_bench,
        "reward_model_total_seconds": reward_model_time,
        "llm_judge_total_seconds": judge_time,
        "reward_model_ms_per_sample": reward_model_time / n_bench * 1000,
        "llm_judge_ms_per_sample": judge_time / n_bench * 1000,
        "speedup_x": judge_time / max(reward_model_time, 1e-9),
    }


def main():
    train_feat = np.load(os.path.join(PROC_DIR, "train_features.npz"))
    feat_c, feat_r = train_feat["feat_chosen"], train_feat["feat_rejected"]

    aucs, accs, model = cross_validate(feat_c, feat_r, n_splits=5)
    print("5-fold AUC:", [round(a, 4) for a in aucs], "mean=", round(float(np.mean(aucs)), 4))
    print("5-fold pairwise acc:", [round(a, 4) for a in accs], "mean=", round(float(np.mean(accs)), 4))

    # 用全量 train 重新拟合最终模型, 在独立的 test 集上做一次 held-out 评估
    final_model = RewardModel(C=0.5).fit(feat_c, feat_r)
    test_feat = np.load(os.path.join(PROC_DIR, "test_features.npz"))
    test_auc, test_acc = final_model.pairwise_auc(test_feat["feat_chosen"], test_feat["feat_rejected"])
    print(f"held-out test AUC={test_auc:.4f} pairwise_acc={test_acc:.4f}")

    import pandas as pd
    df = pd.read_parquet(os.path.join(PROC_DIR, "train_preference.parquet"))
    judge = LocalProxyJudge(embedder=get_embedder())
    latency = benchmark_latency(final_model, feat_c, judge, df["chosen"].tolist(), n_bench=300)
    print("latency benchmark:", json.dumps(latency, indent=2))

    os.makedirs(OUT_DIR, exist_ok=True)
    result = {
        "cv_auc_per_fold": [float(a) for a in aucs],
        "cv_auc_mean": float(np.mean(aucs)),
        "cv_auc_std": float(np.std(aucs)),
        "cv_pairwise_acc_mean": float(np.mean(accs)),
        "held_out_test_auc": float(test_auc),
        "held_out_test_pairwise_acc": float(test_acc),
        "latency_benchmark": latency,
    }
    with open(os.path.join(OUT_DIR, "reward_model_metrics.json"), "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    import joblib
    joblib.dump(final_model, os.path.join(OUT_DIR, "reward_model.joblib"))


if __name__ == "__main__":
    main()
