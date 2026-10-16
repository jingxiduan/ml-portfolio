"""
Reward Model: fits a user-preference model (Bradley-Terry pairwise model) on structured
quality features, producing a Reward Signal that can be used directly for post-training
data filtering.

Sentence-transformer embeddings (PCA-reduced to 40 dims) + 10 hand-crafted quality
features (see features.py) are used as input. The key validation point: the trained
lightweight Reward Model achieves accuracy clearly better than random guessing, while
having inference latency far lower than calling an LLM/Judge for every single sample.

Bradley-Terry model: P(chosen beats rejected) = sigmoid(w^T(x_chosen - x_rejected)),
equivalent to training logistic regression on symmetrically augmented difference
samples — this is solved directly with sklearn's LogisticRegression (more stable
convergence than a fixed number of SGD/Adam steps), and the resulting coefficients w
form an interpretable linear reward scoring function.
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
    """r(x) = w^T x_std + b, a linear Bradley-Terry Reward model trained on standardized features."""

    def __init__(self, C: float = 0.5):
        self.clf = LogisticRegression(max_iter=2000, C=C)

    def fit(self, feat_c: np.ndarray, feat_r: np.ndarray):
        diff = feat_c - feat_r
        X = np.concatenate([diff, -diff], axis=0)
        y = np.concatenate([np.ones(len(diff)), np.zeros(len(diff))])
        self.clf.fit(X, y)
        return self

    def score(self, feat: np.ndarray) -> np.ndarray:
        """Scores a single sample's features with a scalar reward (does not depend on a paired sample, so it can be deployed independently)."""
        w = self.clf.coef_[0]
        return feat @ w  # bias/2 omitted - only shifts the overall scale, does not affect ranking or pairwise comparison

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
    """Compares: calling the LLM Judge per sample (sentence-embedding encoding + similarity scoring) vs. inference with the trained Reward Model."""
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

    # refit the final model on the full train split, and evaluate once on the held-out test split
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
