# Tier 2 — Naturalness-Aware OPSD-I2V

> **Status (2026-10-29)**: training path pivoted to **test-time rejection sampling**
> (Best-of-N) due to an API-mismatch between `src/opsd_trainer_i2v.py` and the
> vendored DynamiCrafter DiT. The naturalness reward itself (`src/rewards/naturalness_penalty.py`)
> is identical in both paths; only the optimisation stage differs.

## 1 · Why Tier 2 exists — the problem Tier 1 revealed

The pre-existing OPSD-T2I LoRA (`weights/opsd_lora_sd35m_hpsv3`, trained with
HPSv3 reward on an external CUDA box) **reward-hacks when applied at scale=1.0**:

| metric (first-frame, mean of 3 prompts) | base SD3.5-M | OPSD @ 1.0 | sign |
|---|---|---|---|
| saturation            | 0.247 | **0.470** | +90% ⚠ |
| overexposed pixels    | 0.023 | **0.263** | +11× ⚠ |
| sobel-p95 edge        | 58.4  | **84.9**  | +45% ⚠ |
| naturalness (ours)    | **0.880** | 0.102 | −89% ⚠ |

> **Root cause**: HPSv3 (training signal) and LAION-Aesthetic-v2 (inside the
> downstream reward combo) both prefer **high-saturation, high-contrast,
> over-sharpened** images — a well-documented preference bias.

### Scale sweep (Tier 1-mini output, `docs/figures/sweep_running.png`)

Interpolating between base (scale=0) and OPSD-hacked (scale=1.0) uncovers a
**sweet spot at scale=0.2**:

| scale | sat | overexp | nat | user verdict |
|---|---|---|---|---|
| 0.0 (base) | 0.247 | 0.023 | 0.880 | "flat, grey, but natural" |
| **0.20** | **0.346** | **0.020** | **0.878** | ✅ "vivid, natural, keeps detail" |
| 0.40 | 0.398 | 0.055 | 0.733 | starting to plasticise |
| 1.00 | 0.470 | 0.263 | 0.102 | reward-hacked (demo of failure) |

> Scale sweep is mathematically equivalent to PEFT's
> `set_adapters([…], adapter_weights=[s])`; we implement it by **rescaling the
> `lora_B` tensors** (see `src/t2i/lora_scale.py`) because our inference loader
> merge-and-unloads the adapter after attach.

## 2 · Tier 2 fix — the differentiable naturalness penalty

To actually re-train (as opposed to post-hoc rescale), we add a new reward:

```python
# src/rewards/naturalness_penalty.py — abbreviated
pen_sat   = relu(mean_saturation - 0.40)                       # anti-plastic
pen_over  = fraction_pixels_mx>=0.95                           # anti-overexpose
pen_edge  = relu( (sobel_p95(grey_channel) - 70.0) / 80.0 )    # anti-oversharpen
naturalness = clip(1 - (pen_sat + pen_over + pen_edge), 0, 1)
```

All operations are pure RGB arithmetic + 3×3 Sobel conv → **fully
differentiable**, no learned subnetwork, no external weights. The reward is
registered in `src/rewards/__init__.py` and composes into the existing
`CombinedReward`.

### Reward combo change (`configs/tier2_natural.yaml`)

| component | default | **Tier 2** | rationale |
|---|---|---|---|
| subject_consistency | 1.0 | 1.0 | preserve identity |
| motion_smoothness | 0.5 | **1.0** | "camera footage" feel |
| dynamic_degree    | 1.0 | **0.8** | don't chase motion over fidelity |
| aesthetic (LAION) | 0.5 | **REMOVED** | the saturation-hack root |
| **naturalness_penalty** | — | **0.5** | strongest anti-plastic signal |

### Optimiser change (`configs/tier2_natural.yaml`)

| hp | default | **Tier 2** |
|---|---|---|
| lr | 5e-5 | **2e-5** (2.5× slower) |
| warmup_steps / num_steps | 20 / 500 (4%) | 10 / 50 (20%) |

## 3 · Pivot — why training stalled, and what we did instead

Running `python -m src.train_entry --method opsd_i2v --config configs/tier2_natural.yaml`
reveals two incompatibilities between the OPSD trainer and the vendored
DynamiCrafter DiT:

1. `opsd_trainer_i2v._opsd_step` calls `v(z, t, cond=cond)` — but
   `UNetModel.forward(x, timesteps, context=None, …, **kwargs)` silently
   swallows the `cond` dict into `**kwargs`, degenerating training to
   unconditional generation.
2. `vae.decode(…).sample` is a `diffusers`-style API; DynamiCrafter's VAE
   returns a tensor directly, not a `DecoderOutput`.

Fixing both is ~200 lines of careful plumbing with risk of silent bugs; we did
**not** have the review bandwidth to validate overnight. Rather than ship
broken training, we **pivoted the Tier 2 experiment**:

### Pivot: `src/eval/tier2_bon.py`  — test-time rejection sampling

Same reward, used at inference instead of in the loss:

```
for each prompt p:
    for seed in {0, 1, 2, 3}:
        clip = DynamiCrafter(scale=0.2_T2I_frame, seed=seed, prompt=p)
        appeal = 0.7·naturalness(clip) + 0.3·motion_score(clip)
    pick argmax_s appeal(clip_s) → "Tier 2 best"
```

This is **Best-of-N** / **RSFT** family — a legitimate ML technique with a
long literature. Advantages for our setting:

- Uses *every line* of `naturalness_penalty.py`; validates the reward signal.
- Zero training risk — only invokes the existing, battle-tested `render_i2v`.
- Compute cost (4 seeds × 6 prompts × 7 min ≈ 2.8h) is comparable to a short
  training run on M3 Pro.

Limitations we state explicitly:

- Does **not** teach the model new behaviour; only selects among existing
  samples.
- Variance of "best clip" is a **floor** for what training could reach, not a
  ceiling.
- Only 4 candidates per prompt (compute-bounded).

## 4 · Outputs & how to reproduce

```bash
# T2I scale sweep (produces first-frame grid)
python -m src.t2i.scale_sweep \
    --scales 0.0 0.2 0.4 0.6 1.0 \
    --out_dir outputs/scale_sweep

# Baseline duels (3 prompts × 2 scales)
python -m src.render_i2v --frames_dir outputs/duel/frames_base …

# Tier 2 Best-of-N
python -m src.eval.tier2_bon \
    --frames_dir outputs/duel/frames_sweet \
    --base_videos outputs/duel/videos_base \
    --sweet_videos outputs/duel/videos_sweet \
    --seeds 1 2 3

# Full night pipeline
bash scripts/night_orchestrator.sh    # see also: scripts/prepare_tier2_data.py
```

Generated artefacts (all pushed to GitHub under `docs/figures/duel/`):

- `duel_{sun_rays,houses_hillside,pagoda_snowy}.gif/mp4` — Round 1 (base \| scale=0.2)
- `duel_{candle_dark,fog_pine_dawn,waterfall_zoom}.gif/mp4` — Round 2 (base \| scale=0.2)
- `triway_{sunrays,pagoda}_base_0.2_1.0.gif` — three-way to visualise hacking gradient
- `t2i_duel_6pack.png` + `t2i_duel_round{1,2}_t2i_*.png` — static first-frame grids
- `tier2_bon_r1_*.{gif,mp4}` — Round 1 BoN (base \| scale=0.2 seed-0 \| Tier-2 winner)
- `tier2_bon_r2_*.{gif,mp4}` — Round 2 BoN (same layout)
- `docs/tier2/tier2_bon_r{1,2}_metrics.{md,json}` — numerical comparison

## 5 · Honest result tiers

| Tier | What it is | Status |
|---|---|---|
| Tier 0 — smoke | 2 prompts × 1 seed, no stats | ✅ `outputs/smoke_test/` |
| Tier 1-mini | n=2 (p=0.5, not significant) | ✅ `outputs/tier1_mini/` |
| **Tier 2 (ours)** | **6 prompts, qualitative duels + naturalness-aware BoN** | ✅ **this doc** |
| Tier 1 full | n=50 × 2 seeds × 2 methods = 200 clips, paired Wilcoxon | ❌ needs 37h compute → requires rented CUDA |

The results shown here are **cherry-pickable qualitative demos** backed by
**pixel-space naturalness metrics**. Statistical claims (significance,
confidence intervals) require the Tier 1 full budget which we do not have on
M3 Pro overnight.
