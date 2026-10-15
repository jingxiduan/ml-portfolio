"""
偏好数据构建与提纯 pipeline, 对应简历:
  "融合显式评分与行为序列构建两级偏好标签; 对模糊样本以 GPT-4o 作为 LLM-as-a-Judge 进行
   RLAIF 二次判别; 结合 Active Learning、去偏采样与置信度筛选, 将原始样本提纯为金标数据"

步骤:
  1. 对 chosen/rejected 两条回复批量提取质量特征(features.build_feature_batch), 得到
     relevance(语义相关度)等, 以 relevance 差值作为初筛"置信度代理"(feature_margin)。
  2. |feature_margin| 低于阈值的样本判定为"模糊样本", 送入 LLM-as-a-Judge(默认本地代理裁判)
     做二次打分, 得到 judge_margin。
  3. 若模糊样本的 judge 二次判别结果与原始人工标注方向相反, 判定为噪声标签, 不进入金标集。
  4. 按综合置信度(非模糊样本用 |feature_margin|, 模糊样本用 |judge_margin|/10)排序, 取
     Top-K 比例作为金标数据集(gold set), 其余作为非金标数据(仍可用于 Reward Model 训练,
     但不进入 DPO 金标集合)。
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
AMBIGUOUS_THRESH = 0.03   # feature_margin 绝对值低于该阈值视为"模糊样本"
GOLD_RATIO = 0.35         # 金标数据占比(非噪声样本中按置信度排序取前 35%)
EMB_DIM = 384             # all-MiniLM-L6-v2 原始句向量维度
PCA_DIM = 40              # 句向量 PCA 降维后的维度, 缓解小样本下的过拟合


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
        # 仅在 train 集上拟合 PCA, 避免信息泄漏; 降维缓解小样本下 384 维句向量的过拟合
        pca = PCA(n_components=PCA_DIM, random_state=42)
        pca.fit(np.concatenate([emb_c, emb_r_vec], axis=0))
        joblib.dump(pca, os.path.join(PROC_DIR, "embedding_pca.joblib"))

    emb_c_pca = pca.transform(emb_c)
    emb_r_pca = pca.transform(emb_r_vec)

    # Reward Model 输入 = 降维后的句向量语义表征(类比简历中的 Qwen-VL 多模态表征) + 结构化质量特征
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
    noisy = is_ambiguous & (judge_margin < 0)  # 模糊样本中 judge 判别与人工标注方向相反

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
