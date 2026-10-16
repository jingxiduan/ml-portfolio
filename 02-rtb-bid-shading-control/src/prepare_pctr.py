"""
Trains a pCTR model on the public Avazu CTR dataset (official LIBSVM mirror, already
feature-hashed), used as the real traffic feature input for the downstream "win-rate
modeling + Bid Shading control" experiment.

Data: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/avazu-app.bz2
First 400K rows (click/no-click binary classification, features already hashed into
high-dimensional sparse one-hot).
"""
import os
import json
import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss
import joblib

RAW_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "avazu_subset.svm")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def main():
    X, y = load_svmlight_file(RAW_PATH)
    print("loaded:", X.shape, "pos_rate=", y.mean())

    X_train, X_rest, y_train, y_rest = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_sim, y_val, y_sim = train_test_split(X_rest, y_rest, test_size=0.5, random_state=42, stratify=y_rest)

    clf = LogisticRegression(max_iter=200, C=1.0, solver="liblinear")
    clf.fit(X_train, y_train)

    val_pred = clf.predict_proba(X_val)[:, 1]
    print("val AUC:", roc_auc_score(y_val, val_pred), "logloss:", log_loss(y_val, val_pred))

    # The "sim" split is reserved exclusively for the auction simulation / bidding strategy
    # experiment, to avoid overlap with the pCTR training/validation samples
    sim_pred = clf.predict_proba(X_sim)[:, 1]
    print("sim AUC:", roc_auc_score(y_sim, sim_pred))

    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez(
        os.path.join(OUT_DIR, "sim_pool.npz"),
        pctr=sim_pred.astype(np.float32),
        click=np.asarray(y_sim).astype(np.int8),
    )
    joblib.dump(clf, os.path.join(OUT_DIR, "pctr_model.joblib"))

    metrics = {
        "n_train": int(X_train.shape[0]),
        "n_val": int(X_val.shape[0]),
        "n_sim": int(X_sim.shape[0]),
        "val_auc": float(roc_auc_score(y_val, val_pred)),
        "val_logloss": float(log_loss(y_val, val_pred)),
        "sim_auc": float(roc_auc_score(y_sim, sim_pred)),
        "sim_pos_rate": float(np.mean(y_sim)),
    }
    with open(os.path.join(OUT_DIR, "pctr_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
