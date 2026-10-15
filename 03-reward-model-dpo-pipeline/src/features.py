"""
多维质量/偏好特征工程, 对应简历"33维质量、用户偏好与交叉特征体系"的小规模复现。

由于公开文本偏好数据集没有"数字人视频"的口型/表情/clone信号, 这里把特征体系迁移到
文本回复质量场景, 保留"语义相关度 + 结构/可读性 + 行为统计 + 交叉特征"的分层设计思路:
  - 语义特征: prompt-response 相关度(embedding 余弦相似度)
  - 结构特征: 长度、句子数、词汇多样性、重复率
  - 可读性特征: 平均词长、平均句长
  - 交叉特征: 长度 × 相关度、多样性 × 相关度 等(对应"克隆方式 × 用户偏好"交叉设计思路)
"""
import re

import numpy as np
from sentence_transformers import SentenceTransformer

_MODEL = None


def get_embedder():
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _MODEL


def _basic_text_stats(text: str):
    words = re.findall(r"\w+", text.lower())
    sentences = re.split(r"[.!?]+", text)
    sentences = [s for s in sentences if s.strip()]
    n_words = len(words)
    n_unique = len(set(words))
    n_sent = max(len(sentences), 1)
    avg_word_len = np.mean([len(w) for w in words]) if words else 0.0
    avg_sent_len = n_words / n_sent

    # 重复率: bigram 重复比例, 越高代表越"水"/套话多
    bigrams = list(zip(words, words[1:]))
    repeat_rate = 0.0
    if bigrams:
        uniq_bigrams = len(set(bigrams))
        repeat_rate = 1 - uniq_bigrams / len(bigrams)

    diversity = n_unique / n_words if n_words > 0 else 0.0
    return {
        "n_words": n_words,
        "n_sent": n_sent,
        "avg_word_len": avg_word_len,
        "avg_sent_len": avg_sent_len,
        "diversity": diversity,
        "repeat_rate": repeat_rate,
    }


def build_feature_vector(prompt: str, response: str, embedder=None):
    embedder = embedder or get_embedder()
    stats = _basic_text_stats(response)

    emb_p = embedder.encode(prompt, normalize_embeddings=True)
    emb_r = embedder.encode(response, normalize_embeddings=True)
    relevance = float(np.dot(emb_p, emb_r))

    feats = _combine(stats, relevance)
    return np.array(feats, dtype=np.float32), relevance


def _combine(stats: dict, relevance: float):
    return [
        stats["n_words"] / 100.0,
        stats["n_sent"] / 10.0,
        stats["avg_word_len"] / 10.0,
        stats["avg_sent_len"] / 20.0,
        stats["diversity"],
        stats["repeat_rate"],
        relevance,
        relevance * stats["diversity"],          # 交叉特征: 相关度 x 多样性
        (stats["n_words"] / 100.0) * relevance,   # 交叉特征: 长度 x 相关度
        relevance * (1 - stats["repeat_rate"]),   # 交叉特征: 相关度 x 低重复
    ]


def build_feature_batch(prompts, responses, embedder=None):
    """批量计算特征, 避免逐条调用 sentence-transformers 带来的性能损耗。"""
    embedder = embedder or get_embedder()
    emb_p = embedder.encode(list(prompts), normalize_embeddings=True, batch_size=256, show_progress_bar=False)
    emb_r = embedder.encode(list(responses), normalize_embeddings=True, batch_size=256, show_progress_bar=False)
    relevances = np.sum(emb_p * emb_r, axis=1)

    feats = []
    for resp, rel in zip(responses, relevances):
        stats = _basic_text_stats(resp)
        feats.append(_combine(stats, float(rel)))
    return np.array(feats, dtype=np.float32), relevances, emb_r
