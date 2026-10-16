"""
Multi-dimensional quality/preference feature engineering.

This builds a feature system for text-response-quality scoring, layered into
"semantic relevance + structural/readability features + behavioral stats + cross
features":
  - Semantic feature: prompt-response relevance (embedding cosine similarity)
  - Structural features: length, sentence count, lexical diversity, repetition rate
  - Readability features: average word length, average sentence length
  - Cross features: length x relevance, diversity x relevance, etc.
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

    # repetition rate: bigram repetition ratio, higher means more filler/boilerplate
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
        relevance * stats["diversity"],          # cross feature: relevance x diversity
        (stats["n_words"] / 100.0) * relevance,   # cross feature: length x relevance
        relevance * (1 - stats["repeat_rate"]),   # cross feature: relevance x low-repetition
    ]


def build_feature_batch(prompts, responses, embedder=None):
    """Computes features in batch, avoiding the performance cost of calling sentence-transformers one at a time."""
    embedder = embedder or get_embedder()
    emb_p = embedder.encode(list(prompts), normalize_embeddings=True, batch_size=256, show_progress_bar=False)
    emb_r = embedder.encode(list(responses), normalize_embeddings=True, batch_size=256, show_progress_bar=False)
    relevances = np.sum(emb_p * emb_r, axis=1)

    feats = []
    for resp, rel in zip(responses, relevances):
        stats = _basic_text_stats(resp)
        feats.append(_combine(stats, float(rel)))
    return np.array(feats, dtype=np.float32), relevances, emb_r
