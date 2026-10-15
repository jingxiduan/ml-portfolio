"""
LLM-as-a-Judge 二次判别, 对应简历"对模糊样本以 GPT-4o 作为 LLM-as-a-Judge, 采用包含
评分维度、推理过程与结构化结论的 rubric prompt 进行 RLAIF 二次判别"。

说明: 公开 GitHub 项目不便内置付费 API Key, 因此:
  - `RubricPromptTemplate` 给出与生产环境一致的结构化 rubric prompt 设计(评分维度 /
    推理过程 / 结构化结论三段式), 若用户在环境变量中配置 OPENAI_API_KEY, 可通过
    `OpenAIJudge` 调用真实 LLM 完成打分;
  - 默认使用 `LocalProxyJudge`: 基于句向量与"高质量/低质量"锚点语句的相似度打分,
    作为免 API Key 的轻量替代, 用于在没有 LLM 服务的环境下跑通完整 pipeline。
"""
import json
import os

import numpy as np

RUBRIC_PROMPT_TEMPLATE = """你是一个严格的回复质量评审员。请针对下面的问题与候选回复, 按以下维度打分:
1. 相关性(是否直接回答了问题)
2. 信息完整度与准确性
3. 表达清晰度与结构
4. 是否存在空洞套话/重复

问题: {prompt}
候选回复: {response}

请输出如下 JSON 结构(不要输出多余内容):
{{
  "relevance": <0-10 打分>,
  "completeness": <0-10 打分>,
  "clarity": <0-10 打分>,
  "reasoning": "<一句话说明打分依据>",
  "final_score": <0-10 综合打分>
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
    """无需付费 API 的本地代理裁判: 用句向量到好/坏锚点的相似度差作为 LLM Judge 的替代打分。"""

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
        """批量打分, 返回 final_score 数组(0-10), 用于对"模糊样本"做 RLAIF 二次判别。"""
        emb = self.embedder.encode(list(responses), normalize_embeddings=True,
                                    batch_size=256, show_progress_bar=False)
        good_sim = emb @ self.good_emb.T
        bad_sim = emb @ self.bad_emb.T
        score = (good_sim.mean(axis=1) - bad_sim.mean(axis=1)) * 10 + 5
        return np.clip(score, 0, 10)


class OpenAIJudge:
    """可选: 配置 OPENAI_API_KEY 后调用真实 LLM 完成 rubric 打分(与生产环境一致)。"""

    def __init__(self, model: str = "gpt-4o-mini"):
        import openai  # noqa: F401  (仅在显式使用该类时才需要安装)
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
