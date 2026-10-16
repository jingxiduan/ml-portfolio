"""
Preference data construction and distillation pipeline:
  "Combines explicit feature-based confidence with LLM-as-a-Judge; re-labels ambiguous
   samples via RLAIF using a rubric prompt; distills raw samples into gold-standard data
   through confidence-based filtering."

Steps:
  1. Extract quality features in batch for the chosen/rejected responses
     (features.build_feature_batch), obtaining relevance (semantic relevance); the
     relevance gap serves as an initial "confidence proxy" (feature_margin).
  2. Samples whose |feature_margin| is below a threshold are flagged as "ambiguous" and
     sent to LLM-as-a-Judge (defaults to the local proxy judge) for re-scoring, yielding
     judge_margin.
  3. If an ambiguous sample's judge re-labeling disagrees with the original human label
     direction, it is flagged as a noisy label and excluded from the gold set.
  4. Rank by combined confidence (|feature_margin| for non-ambiguous samples,
     |judge_margin|/10 for ambiguous samples) and take the top-K ratio as the gold
     dataset; the rest remain usable for Reward Model training but are excluded from
     the DPO gold set.
"""
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from features import build_feature_batch, get_embedder
from llm_judge import LocalProxyJudge

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
AMBIGUOUS_THRESH = 0.03   # feature_margin below this absolute value is flagged as "ambiguous"
GOLD_RATIO = 0.35         # share of gold data (top 35% by confidence among non-noisy samples)
EMB_DIM = 384             # raw all-MiniLM-L6-v2 sentence embedding dimension
PCA_DIM = 40              # PCA-reduced embedding dimension, mitigates overfitting on small samples


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def process_split(split: str, embedder, pca: PCA = None):
    rows = load_jsonl(os.path.join(RAW_DIR, f"{split}.jsonl"))
    prompts = [r["prompt"] for r in rows]
    chosen = [r["chosen"] for r in rows]
    rejected = [r["rejected"] for r in rows]

    t0 = time.time()
    stat_feat_c, rel_c, emb_c = build_feature_batch(prompts, chosen, embedder)
    stat_feat_r, rel_r, emb_r_vec = build_feature_batch(prompts, rejected, embedder)

    if pca is None:
        # fit PCA only on the train split to avoid data leakage; dimensionality reduction
        # mitigates overfitting of the 384-dim sentence embedding on small samples
        pca = PCA(n_components=PCA_DIM, random_state=42)
        pca.fit(np.concatenate([emb_c, emb_r_vec], axis=0))
        joblib.dump(pca, os.path.join(PROC_DIR, "embedding_pca.joblib"))

    emb_c_pca = pca.transform(emb_c)
    emb_r_pca = pca.transform(emb_r_vec)

    # Reward Model input = PCA-reduced sentence embedding (semantic representation) + structured quality features
    feat_c = np.concatenate([emb_c_pca, stat_feat_c], axis=1)
    feat_r = np.concatenate([emb_r_pca, stat_feat_r], axis=1)
    feat_time = time.time() - t0

    feature_margin = rel_c - rel_r
    is_ambiguous = np.abs(feature_margin) < AMBIGUOUS_THRESH

    judge = LocalProxyJudge(embedder=embedder)
    judge_score_c = np.full(len(rows), np.nan)
    judge_score_r = np.full(len(rows), np.nan)
    amb_idx = np.where(is_ambiguous)[0]

    t0 = time.time()
    if len(amb_idx) > 0:
        amb_chosen = [chosen[i] for i in amb_idx]
        amb_rejected = [rejected[i] for i in amb_idx]
        judge_score_c[amb_idx] = judge.batch_score(amb_chosen)
        judge_score_r[amb_idx] = judge.batch_score(amb_rejected)
    judge_time = time.time() - t0

    judge_margin = judge_score_c - judge_score_r
    noisy = is_ambiguous & (judge_margin < 0)  # ambiguous samples where the judge disagrees with the human label direction

    confidence = np.where(
        is_ambiguous,
        np.abs(np.nan_to_num(judge_margin)) / 10.0,
        np.abs(feature_margin),
    )

    valid_idx = np.where(~noisy)[0]
    gold_n = int(len(valid_idx) * GOLD_RATIO)
    gold_order = valid_idx[np.argsort(-confidence[valid_idx])][:gold_n]
    is_gold = np.zeros(len(rows), dtype=bool)
    is_gold[gold_order] = True

    df = pd.DataFrame({
        "prompt": prompts, "chosen": chosen, "rejected": rejected,
        "relevance_chosen": rel_c, "relevance_rejected": rel_r,
        "feature_margin": feature_margin, "is_ambiguous": is_ambiguous,
        "judge_score_chosen": judge_score_c, "judge_score_rejected": judge_score_r,
        "is_noisy": noisy, "confidence": confidence, "is_gold": is_gold,
    })

    os.makedirs(PROC_DIR, exist_ok=True)
    df.to_parquet(os.path.join(PROC_DIR, f"{split}_preference.parquet"))
    np.savez(os.path.join(PROC_DIR, f"{split}_features.npz"), feat_chosen=feat_c, feat_rejected=feat_r)

    stats = {
        "split": split,
        "n_total": len(rows),
        "n_ambiguous": int(is_ambiguous.sum()),
        "ambiguous_ratio": float(is_ambiguous.mean()),
        "n_noisy_discarded": int(noisy.sum()),
        "n_gold": int(is_gold.sum()),
        "gold_ratio_of_total": float(is_gold.sum() / len(rows)),
        "feature_extract_seconds": feat_time,
        "judge_seconds_on_ambiguous": judge_time,
    }
    return stats, pca


def main():
    embedder = get_embedder()
    all_stats = {}
    pca = None
    for split in ["train", "test"]:
        stats, pca = process_split(split, embedder, pca=pca)
        print(json.dumps(stats, indent=2, ensure_ascii=False))
        all_stats[split] = stats

    with open(os.path.join(PROC_DIR, "preference_build_stats.json"), "w") as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
