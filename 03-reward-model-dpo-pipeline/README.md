# Preference Data Construction + Reward Model + DPO Pipeline

Reproduces the full pipeline of "human preference data construction -> LLM-as-a-Judge re-labeling -> Reward Model distillation -> DPO training data generation" on a public preference dataset.

> Note: this reproduction targets a general text-response-quality setting (rather than a proprietary multimodal scenario), so:
> - the dataset used is [Anthropic/hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf) (`helpful-base` subset, official Hugging Face dataset, no login required), which structurally matches the target scenario: a human-labeled chosen/rejected pair of candidate responses under the same context;
> - multimodal representations are replaced with `sentence-transformers` (all-MiniLM-L6-v2) sentence embeddings + hand-crafted quality features;
> - LLM-as-a-Judge defaults to a local proxy judge (`LocalProxyJudge`) that requires no API key, while keeping a realistic rubric-prompt template and an `OpenAIJudge` interface — set `OPENAI_API_KEY` to switch seamlessly to a real GPT-series model.

## Key Techniques

1. **Preference data construction**: combines feature-based confidence with LLM-as-a-Judge to re-label ambiguous samples via RLAIF, then filters by confidence to distill gold-standard data
2. **Reward Model**: takes semantic embeddings + structured quality features as input, fits a Bradley-Terry preference model, and produces an independently deployable Reward Signal with far lower latency than calling an LLM Judge directly
3. **DPO training data generation**: exports the standard `{prompt, chosen, rejected}` format, ready to feed directly into `trl.DPOTrainer`

## Code Structure

```
03-reward-model-dpo-pipeline/
├── src/
│   ├── download_data.py          # download Anthropic/hh-rlhf (helpful-base), no login required
│   ├── features.py                # sentence embeddings + structured quality features (relevance/diversity/repetition/cross features)
│   ├── llm_judge.py                # RubricPromptTemplate + LocalProxyJudge + OpenAIJudge
│   ├── build_preference_dataset.py # ambiguous-sample detection -> judge re-labeling -> confidence-filtered gold set
│   ├── reward_model.py             # linear Bradley-Terry Reward Model, 5-fold CV + latency benchmark
│   ├── build_dpo_dataset.py        # export standard DPO-format data + data funnel stats
│   └── dpo_train_demo.py           # end-to-end trl.DPOTrainer smoke test (default tiny-gpt2, runs on CPU)
└── outputs/
```

## Run

```bash
pip install -r requirements.txt
python src/download_data.py
python src/build_preference_dataset.py
python src/reward_model.py
python src/build_dpo_dataset.py
python src/dpo_train_demo.py   # optional: verifies the DPO training pipeline runs end-to-end
```

## Results

**Preference data distillation funnel** (train split, 7,980 raw chosen/rejected pairs):

| Stage | Count | Share |
|---|---|---|
| Raw preference pairs | 7,980 | 100% |
| "Ambiguous" samples (semantic relevance hard to distinguish, sent to LLM-as-a-Judge for re-labeling) | 1,029 | 12.9% |
| Judged as noisy labels (judge disagrees with the human label direction, discarded) | 482 | 6.0% |
| Final gold data (top 35% by confidence, used for DPO training) | 2,624 | 32.9% |

**Reward Model** (40-dim PCA-reduced sentence embedding + 10-dim structured features, linear Bradley-Terry model, 5-fold cross-validation):

| Metric | Value |
|---|---|
| 5-fold AUC (mean ± std) | 0.6072 ± 0.0029 |
| 5-fold pairwise accuracy | 0.6267 |
| Held-out test AUC | 0.6150 |
| Held-out test pairwise accuracy | 0.6317 |

> Note: the AUC here is modest, which is expected — on datasets like hh-rlhf where chosen/rejected differ only in subtle wording, human-annotator agreement itself is limited, and fitting lightweight features caps the achievable ceiling further. **The more practically valuable result is the inference-latency comparison below.**

**Inference latency comparison** (300 samples, Reward Model vs. calling the LLM Judge per sample):

| Approach | Per-sample latency | Total (300 samples) |
|---|---|---|
| LLM-as-a-Judge (sentence-embedding judge, including encoding) | 16.17 ms | 4.85 s |
| Reward Model (linear forward pass on precomputed features) | 0.00097 ms | 0.00029 s |
| **Speedup** | **≈16,700x** | - |

This confirms the core value proposition: distilling the "call an LLM/Judge for every sample" workflow into a lightweight Reward Model cuts inference cost by several orders of magnitude, making it feasible to apply the Reward Model across the full dataset rather than only the small subset used during training.

**DPO data**: `outputs/dpo_train.jsonl` (2,624 rows), `outputs/dpo_test.jsonl` (497 rows), standard `{prompt, chosen, rejected}` format. `dpo_train_demo.py` verifies this data can be fed directly into `trl.DPOTrainer` to run a training loop (defaults to `sshleifer/tiny-gpt2` for a 20-step CPU smoke test; swap in a larger model and a GPU environment for real training).
