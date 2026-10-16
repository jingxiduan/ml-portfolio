"""
LLM-as-a-Judge re-labeling: judges ambiguous samples using a rubric prompt covering
scoring dimensions, reasoning, and a structured conclusion, as a form of RLAIF
re-labeling.

Note: since this is a public GitHub project, it is not practical to embed a paid API
key, so:
  - `RubricPromptTemplate` provides a production-style structured rubric-prompt design
    (scoring dimensions / reasoning / structured conclusion, three parts). If the user
    sets `OPENAI_API_KEY` in the environment, `OpenAIJudge` can call a real LLM to score;
  - By default, `LocalProxyJudge` is used: it scores based on sentence-embedding
    similarity to "high-quality/low-quality" anchor statements, serving as a lightweight,
    API-key-free substitute so the full pipeline can run without any LLM service.
"""
import json
import os

import numpy as np

RUBRIC_PROMPT_TEMPLATE = """You are a strict response-quality reviewer. For the question and candidate response below, score it along the following dimensions:
1. Relevance (does it directly answer the question)
2. Completeness and accuracy of information
3. Clarity and structure of expression
4. Whether it contains vague filler or repetition

Question: {prompt}
Candidate response: {response}

Output the following JSON structure (no extra content):
{{
  "relevance": <score 0-10>,
  "completeness": <score 0-10>,
  "clarity": <score 0-10>,
  "reasoning": "<one sentence explaining the score>",
  "final_score": <overall score 0-10>
}}
"""

_GOOD_ANCHORS = [
    "This is a clear, detailed, and directly helpful answer with concrete steps.",
    "The response accurately and thoroughly addresses the user's question.",
]
_BAD_ANCHORS = [
    "This response is vague, repetitive, and does not really answer the question.",
    "The answer is generic, off-topic, or unhelpful.",
]


class LocalProxyJudge:
    """A local proxy judge requiring no paid API: uses the difference in sentence-embedding
    similarity to good/bad anchors as a substitute for an LLM Judge score."""

    def __init__(self, embedder=None):
        from features import get_embedder
        self.embedder = embedder or get_embedder()
        self.good_emb = self.embedder.encode(_GOOD_ANCHORS, normalize_embeddings=True)
        self.bad_emb = self.embedder.encode(_BAD_ANCHORS, normalize_embeddings=True)

    def score(self, prompt: str, response: str) -> dict:
        emb = self.embedder.encode(response, normalize_embeddings=True)
        good_sim = float(np.mean(self.good_emb @ emb))
        bad_sim = float(np.mean(self.bad_emb @ emb))
        final_score = float(np.clip((good_sim - bad_sim) * 10 + 5, 0, 10))
        return {
            "relevance": None, "completeness": None, "clarity": None,
            "reasoning": f"proxy judge: good_sim={good_sim:.3f} bad_sim={bad_sim:.3f}",
            "final_score": final_score,
        }

    def batch_score(self, responses) -> np.ndarray:
        """Scores in batch, returning an array of final_score values (0-10), used for RLAIF re-labeling of ambiguous samples."""
        emb = self.embedder.encode(list(responses), normalize_embeddings=True,
                                    batch_size=256, show_progress_bar=False)
        good_sim = emb @ self.good_emb.T
        bad_sim = emb @ self.bad_emb.T
        score = (good_sim.mean(axis=1) - bad_sim.mean(axis=1)) * 10 + 5
        return np.clip(score, 0, 10)


class OpenAIJudge:
    """Optional: once OPENAI_API_KEY is set, calls a real LLM to produce a rubric score (matching a production setup)."""

    def __init__(self, model: str = "gpt-4o-mini"):
        import openai  # noqa: F401  (only required if this class is explicitly used)
        self.client = openai.OpenAI()
        self.model = model

    def score(self, prompt: str, response: str) -> dict:
        content = RUBRIC_PROMPT_TEMPLATE.format(prompt=prompt, response=response)
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            temperature=0,
        )
        text = resp.choices[0].message.content
        return json.loads(text)


def get_default_judge():
    if os.environ.get("OPENAI_API_KEY"):
        return OpenAIJudge()
    return LocalProxyJudge()
