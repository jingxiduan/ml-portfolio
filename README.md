# algo-portfolio

## Projects

| Project | Focus | Dataset | Techniques |
|---|---|---|---|
| [`01-multitask-ctr-cvr-rec`](01-multitask-ctr-cvr-rec) | Multi-task tower recommendation / RankMixer dense feature interaction | [MovieLens-1M](https://grouplens.org/datasets/movielens/1m/) | Shared embedding + MLP-Mixer token/channel mixing, CTR/CVR dual-tower, sample-mask gradient isolation, GAUC/calibration evaluation |
| [`02-rtb-bid-shading-control`](02-rtb-bid-shading-control) | Win-rate model `omni_pwrm` / bid shading feedback control | [Avazu CTR Dataset](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/avazu-app.bz2) + auction simulation | pCTR modeling, second-price auction simulation, online win-rate model, greedy bid search, PID/MPC feedback control comparison |
| [`03-reward-model-dpo-pipeline`](03-reward-model-dpo-pipeline) | Preference data construction / Reward Model / DPO data pipeline | [Anthropic/hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf) (helpful-base) | LLM-as-a-Judge re-labeling, confidence-based gold data filtering, Bradley-Terry reward model, `trl.DPOTrainer` end-to-end |

## Structure

```
algo-portfolio/
├── 01-multitask-ctr-cvr-rec/
├── 02-rtb-bid-shading-control/
└── 03-reward-model-dpo-pipeline/
```

## Quickstart

```bash
git clone <this-repo>
cd algo-portfolio
python3 -m venv .venv && source .venv/bin/activate

cd 01-multitask-ctr-cvr-rec && pip install -r requirements.txt && python src/download_data.py && python src/preprocess.py && python src/train.py --epochs 4

cd ../02-rtb-bid-shading-control && pip install -r requirements.txt && python src/download_data.py && python src/prepare_pctr.py && python src/run_experiment.py

cd ../03-reward-model-dpo-pipeline && pip install -r requirements.txt && python src/download_data.py && python src/build_preference_dataset.py && python src/reward_model.py && python src/build_dpo_dataset.py
```
