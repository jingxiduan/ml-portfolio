# algo-portfolio

## Projects

| Project | Focus | Dataset | Techniques |
|---|---|---|---|
| [`01-multitask-ctr-cvr-rec`](01-multitask-ctr-cvr-rec) | Multi-task tower recommendation / RankMixer dense feature interaction | [MovieLens-1M](https://grouplens.org/datasets/movielens/1m/) | Shared embedding + MLP-Mixer token/channel mixing, CTR/CVR dual-tower, sample-mask gradient isolation, GAUC/calibration evaluation |
| [`02-rtb-bid-shading-control`](02-rtb-bid-shading-control) | Win-rate model `omni_pwrm` / bid shading feedback control | [Avazu CTR Dataset](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/avazu-app.bz2) + auction simulation | pCTR modeling, second-price auction simulation, online win-rate model, greedy bid search, PID/MPC feedback control comparison |
| [`03-reward-model-dpo-pipeline`](03-reward-model-dpo-pipeline) | Preference data construction / Reward Model / DPO data pipeline | [Anthropic/hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf) (helpful-base) | LLM-as-a-Judge re-labeling, confidence-based gold data filtering, Bradley-Terry reward model, `trl.DPOTrainer` end-to-end |
| [`04-diffusion-opsd-mps`](04-diffusion-opsd-mps) | On-policy self-distillation RL post-training for diffusion models | [DrawBench prompts](https://github.com/worldbench/DiffusionOPSD) (public subset) | On-policy query sampling, reward-gradient-guided positive/negative target construction, detached distillation fit, EMA-refreshed behavior policy, dual-LoRA on SD-Turbo, differentiable CLIPScore reward, single-machine MPS training |
| [`05-video-opsd-mps`](05-video-opsd-mps) | Extending OPSD RL post-training to text-to-video | Hand-authored motion-prompt set (30 prompts) | Same OPSD loop on AnimateDiff (frozen motion adapter + SD1.5 UNet), LoRA scoped to spatial attention only, frame-averaged CLIPScore + temporal-consistency reward, held-out video reward evaluation |

## Structure

```
algo-portfolio/
├── 01-multitask-ctr-cvr-rec/
├── 02-rtb-bid-shading-control/
├── 03-reward-model-dpo-pipeline/
├── 04-diffusion-opsd-mps/
└── 05-video-opsd-mps/
```

## Quickstart

```bash
git clone <this-repo>
cd algo-portfolio
python3 -m venv .venv && source .venv/bin/activate

cd 01-multitask-ctr-cvr-rec && pip install -r requirements.txt && python src/download_data.py && python src/preprocess.py && python src/train.py --epochs 4

cd ../02-rtb-bid-shading-control && pip install -r requirements.txt && python src/download_data.py && python src/prepare_pctr.py && python src/run_experiment.py

cd ../03-reward-model-dpo-pipeline && pip install -r requirements.txt && python src/download_data.py && python src/build_preference_dataset.py && python src/reward_model.py && python src/build_dpo_dataset.py

cd ../04-diffusion-opsd-mps && pip install -r requirements.txt && python src/download_data.py && python src/opsd_train.py --rounds 400 --batch-size 1 && python src/generate_samples.py && python src/evaluate.py

cd ../05-video-opsd-mps && pip install -r requirements.txt && python src/prepare_data.py && python src/opsd_train_video.py --rounds 400 --batch-size 1 --num-frames 6 --resolution 256 && python src/generate_samples_video.py && python src/evaluate_video.py
```
