# Multi-Task CTR/CVR Recommendation with RankMixer-style Dense Interaction

Reproduces a "multi-task tower modeling + dense feature interaction (RankMixer)" recommendation scenario on a public dataset: a shared bottom representation feeds two task-specific towers (CTR / CVR), where the CVR branch backpropagates gradients only on clicked samples via a **sample mask** — the classic solution to the "impression-click-conversion" dependency chain in multi-task recommendation (ESMM-style).

> Note: MovieLens does not natively contain real "impression/click/conversion" three-stage ad logs, so this project follows the common industry practice of constructing proxy labels: **rated movies = clicked (CTR=1), unrated movies via negative sampling = not clicked (CTR=0); among clicked samples, rating>=4 is treated as "conversion/satisfied" (CVR=1)**. This is the standard approach for reproducing industrial-grade multi-task CTR/CVR scenarios on public datasets, and all metrics below are computed on these proxy labels.

## Key Techniques

- Multi-task tower modeling with shared feature engineering: shared bottom embeddings + task-specific towers + sample-mask gradient isolation
- RankMixer dense feature interaction: MLP-Mixer-style token-mixing / channel-mixing replacing traditional FM/DCN explicit crossing, shifting the model from sparse-lookup-dominant to dense-computation-dominant
- Offline evaluation: GAUC (grouped AUC), AUC, LogLoss, bucketed calibration error (AbsBias)

## Dataset

[MovieLens-1M](https://grouplens.org/datasets/movielens/1m/), official direct download, no registration required, ~1M ratings, 6040 users, 3952 movies.

```bash
python src/download_data.py
```

## Method

1. **Sample construction** (`src/preprocess.py`): positive samples = real user ratings; negative samples = negative sampling over movies the user never interacted with (ratio 1:2); features include user profile (gender/age bucket/occupation), user historical behavior stats (average rating/rating count), and movie profile (genre one-hot/popularity stats).
2. **Model** (`src/model.py`, `RankMixerMTL`):
   - Each sparse feature has its own embedding; dense features are projected via a linear layer into tokens of the same dimension, concatenated into a `(B, T, D)` token sequence;
   - Stacked `MixerBlock`s: LayerNorm -> token-mixing MLP (cross-feature interaction) -> residual -> LayerNorm -> channel-mixing MLP (cross-channel interaction) -> residual;
   - The flattened shared representation feeds separately into a CTR tower and a CVR tower (two-layer MLP + sigmoid).
3. **Loss**: `loss = BCE(ctr_logit, click) + mask * BCE(cvr_logit, convert)`, where `mask` comes from the `label_convert=-1` placeholder on non-clicked samples, so only clicked samples contribute to the CVR gradient.

## Run

```bash
pip install -r requirements.txt
python src/download_data.py
python src/preprocess.py
python src/train.py --epochs 4 --batch_size 4096
```

## Results (4 epochs, CPU, ~3 min)

| Task | AUC | GAUC | LogLoss | AbsBias (calibration error) |
|---|---|---|---|---|
| CTR (click) | 0.8805 | 0.8796 | 0.4035 | 0.0046 |
| CVR (conversion within clicks) | 0.7960 | - | 0.5376 | 0.0305 |

Training curve: `outputs/training_curve.png`; full metrics: `outputs/metrics.json`.

```
[epoch 1] train_loss=1.0003 val_ctr_auc=0.8686 val_cvr_auc=0.7874
[epoch 2] train_loss=0.9643 val_ctr_auc=0.8733 val_cvr_auc=0.7909
[epoch 3] train_loss=0.9523 val_ctr_auc=0.8777 val_cvr_auc=0.7940
[epoch 4] train_loss=0.9414 val_ctr_auc=0.8818 val_cvr_auc=0.7962
```

## Structure

```
01-multitask-ctr-cvr-rec/
├── src/
│   ├── download_data.py   # download MovieLens-1M
│   ├── preprocess.py      # build CTR/CVR multi-task samples
│   ├── dataset.py         # PyTorch Dataset
│   ├── model.py           # RankMixer-style multi-task model
│   ├── metrics.py         # GAUC / AUC / calibration error
│   └── train.py           # training + evaluation + plotting
├── outputs/                # metrics.json / training_curve.png produced by training
└── requirements.txt
```
