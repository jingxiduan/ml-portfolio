import numpy as np
from sklearn.metrics import roc_auc_score, log_loss


def gauc(user_ids, labels, scores):
    """Grouped AUC (grouped by user_id, weighted average by group size) - the GAUC evaluation metric."""
    import pandas as pd
    df = pd.DataFrame({"u": user_ids, "y": labels, "s": scores})
    total_auc, total_w = 0.0, 0.0
    for _, g in df.groupby("u"):
        if g["y"].nunique() < 2:
            continue
        try:
            auc = roc_auc_score(g["y"], g["s"])
        except ValueError:
            continue
        w = len(g)
        total_auc += auc * w
        total_w += w
    return total_auc / total_w if total_w > 0 else float("nan")


def calibration_abs_bias(labels, scores, n_bins: int = 10):
    """Bucketed calibration error (AbsBias): mean absolute difference between predicted and true means per bucket."""
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    order = np.argsort(scores)
    bins = np.array_split(order, n_bins)
    biases = []
    for b in bins:
        if len(b) == 0:
            continue
        biases.append(abs(scores[b].mean() - labels[b].mean()))
    return float(np.mean(biases))


def binary_metrics(labels, scores):
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    eps = 1e-7
    scores_clip = np.clip(scores, eps, 1 - eps)
    return {
        "auc": float(roc_auc_score(labels, scores)) if len(np.unique(labels)) > 1 else float("nan"),
        "logloss": float(log_loss(labels, scores_clip)),
        "abs_bias": calibration_abs_bias(labels, scores),
        "pos_rate": float(labels.mean()),
        "pred_mean": float(scores.mean()),
    }
