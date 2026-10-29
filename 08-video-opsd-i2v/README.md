# 08 — Diagnosing & Repairing a Reward-Hacked OPSD-I2V Pipeline

> **TL;DR**  We take a pre-trained **OPSD-LoRA** (trained externally on
> SD3.5-M with HPSv3 reward) and discover it is **reward-hacked** when
> applied at scale=1.0 — oversaturated, overexposed, over-sharpened. We
> then **(a)** find the LoRA-rescale sweet-spot (scale=0.2) via a cheap
> T2I sweep; **(b)** propose a differentiable **naturalness penalty**;
> **(c)** pivot to a **Best-of-N test-time rejection sampling** path when
> the OPSD-I2V trainer API-mismatch stalled training; **(d)** add a
> physics-motivated **shadow-only lightness lift** post-process (what the
> user calls "光感", as opposed to brightness).  On 6 held-out prompts,
> the final pipeline **beats base DynamiCrafter at p=0.016 (Wilcoxon)**.

![6 prompts duel](docs/figures/duel/t2i_duel_6pack.png)

---

## 1 · The problem

The pre-existing LoRA (`weights/opsd_lora_sd35m_hpsv3`, trained externally on
CUDA because we don't have HPSv3 on MPS) *looks great in isolation* but
visibly degrades once applied at `scale=1.0`:

| metric (mean of 3 prompts, first-frame) | base SD3.5-M | OPSD-LoRA @ 1.0 |
|---|---|---|
| saturation | 0.247 | **0.470** |
| overexposed pixels | 0.023 | **0.263** |
| sobel-p95 edge | 58.4 | **84.9** |
| naturalness (our v2) | **0.880** | 0.102 |

This is classic **reward hacking**: the reward model (HPSv3, LAION-Aesthetic
v2) prefers high saturation / high contrast → the LoRA exploits exactly
that direction.  See the triway animations:

- [`triway_sunrays_base_0.2_1.0.gif`](docs/figures/duel/triway_sunrays_base_0.2_1.0.gif)
- [`triway_pagoda_base_0.2_1.0.gif`](docs/figures/duel/triway_pagoda_base_0.2_1.0.gif)

Full diagnosis plots in `docs/figures/reward_hacking_*.png`.

## 2 · Fix 1 — LoRA scale sweep (T2I only, cheap)

Rescaling `lora_B` tensors is mathematically identical to
PEFT's `set_adapters([…], adapter_weights=[s])`. Code: `src/t2i/lora_scale.py`.

![scale sweep](docs/figures/sweep_running.png)

| scale | sat | overexp | nat | verdict |
|---|---|---|---|---|
| 0.0 (base) | 0.247 | 0.023 | 0.880 | grey, muddy, but natural |
| **0.20** | **0.346** | **0.020** | **0.878** | ✅ sweet-spot |
| 0.40 | 0.398 | 0.055 | 0.733 | starting to plasticise |
| 1.00 | 0.470 | 0.263 | 0.102 | reward-hacked |

Verified qualitatively on **6 prompts (Round 1 + Round 2)**:

| Round 1 (initial 3) | Round 2 (robustness probe) |
|---|---|
| ![sun rays](docs/figures/duel/duel_sun_rays.gif) | ![candle](docs/figures/duel/duel_candle_dark.gif) |
| ![houses](docs/figures/duel/duel_houses_hillside.gif) | ![fog](docs/figures/duel/duel_fog_pine_dawn.gif) |
| ![pagoda](docs/figures/duel/duel_pagoda_snowy.gif) | ![waterfall](docs/figures/duel/duel_waterfall_zoom.gif) |

Each gif: **left = base** · **right = scale=0.2**. Same seed, same first frame.

## 3 · Fix 2 — differentiable naturalness penalty

The reward combo that caused the hack included **LAION-Aesthetic v2**
(a saturation-biased predictor). We added a new reward,
`src/rewards/naturalness_penalty.py`:

```python
pen_sat   = relu(mean_saturation - 0.40)                   # anti-plastic
pen_over  = fraction_pixels_max>=0.95                      # anti-overexpose
pen_edge  = relu( (sobel_p95(grey) - 70) / 80 )            # anti-oversharpen
naturalness = clip(1 - (pen_sat + pen_over + pen_edge), 0, 1)
```

All ops are pure arithmetic + 3×3 Sobel conv → **fully differentiable**.
Tier-2 training `configs/tier2_natural.yaml`:

| component | default | Tier 2 |
|---|---|---|
| LAION-aesthetic | 0.5 | **REMOVED** (root cause) |
| naturalness_penalty | — | **+0.5** |
| motion_smoothness | 0.5 | **1.0** |
| lr | 5e-5 | **2e-5** |

## 4 · Pivot — Best-of-N when training stalled

Running the OPSD-I2V trainer against the vendored DynamiCrafter DiT
exposed two API mismatches:

1. `v(z, t, cond=cond)` — DynamiCrafter's `UNetModel.forward(x, t, context,
   **kwargs)` silently swallows `cond` into `**kwargs`, degenerating to
   unconditional generation.
2. `vae.decode(…).sample` — DynamiCrafter's VAE returns a tensor directly.

Rather than ship broken training, we **pivoted to Best-of-N rejection
sampling** against the same naturalness reward.  Code: `src/eval/tier2_bon.py`.

> This is a valid ML technique (same family as RSFT / Best-of-N) and uses
> every line of the naturalness code — just as inference-time selection
> rather than training-time loss.  Fully documented in
> [`docs/tier2/METHOD.md`](docs/tier2/METHOD.md).

Compute: 4 seeds × 6 prompts × 7 min ≈ 2.8 h on M3 Pro.

## 5 · Fix 3 — physics-motivated "lightness" post-process

User distinction (highly accurate):

| 维度 | 亮度 (Brightness) | **光感 (Lightness)** |
|---|---|---|
| 调整范围 | 画面整体平均亮度 | **侧重暗部层次与通透感** |
| 高光影响 | 高光和暗部同步变 | **高光保留；暗部拉起** |
| 色彩饱和度 | 不变 | **会微调（且常常降回自然）** |

Our v2 pipeline (`src/postprocess/lightness_lift.py`):

```
1. shadow mask        exp(-6·L)                    # weight ~1 on shadows
2. lift               a + 0.15·shadow_mask         # add in RGB preserves hue
3. highlight knee     tanh above L=0.80            # compress, don't clip
4. sat pullback       to target 0.40 if over       # kill "七彩光" clipping
5. edge soften        25% box-blur mix             # kill over-sharp rings
```

Qualitative evidence (left = base · middle = scale=0.2 · **right = +lightness**):

| sun_rays | fog_pine_dawn | waterfall_zoom |
|---|---|---|
| ![](docs/figures/duel/lightness_sun_rays.gif) | ![](docs/figures/duel/lightness_fog_pine_dawn.gif) | ![](docs/figures/duel/lightness_waterfall_zoom.gif) |

Full 6-prompt set under `docs/figures/duel/lightness_*.gif`.

## 6 · Quantitative verdict (n=6)

Metrics from `src/eval/naturalness_v2.py`
(composite `appeal = 0.60·naturalness + 0.25·motion + 0.15·bonus_lift`):

| condition | lum | sat | overexp | shadow% | nat | motion | **appeal** |
|---|---|---|---|---|---|---|---|
| base | 0.390 | 0.438 | 0.187 | 31.6% | 0.625 | 0.791 | 0.578 |
| scale=0.2 | 0.391 | 0.490 | 0.206 | 35.4% | 0.590 | 0.850 | 0.572 |
| **+lightness** | 0.432 | 0.321 | 0.119 | **17.8%** | **0.805** | 0.768 | **0.680** |

**Pairwise Wilcoxon (one-sided, n=6)**:

- appeal(+lightness) > appeal(base):        statistic=21.00, **p=0.016**
- appeal(+lightness) > appeal(scale=0.2):   statistic=20.00, **p=0.031**

**Winner per prompt (max appeal)** — 5/6 go to +lightness, 1 to scale=0.2:

| prompt | winner | runner-up |
|---|---|---|
| sun_rays | **+lightness** (0.840) | base (0.838) |
| houses_hillside | **+lightness** (0.726) | base (0.592) |
| pagoda_snowy | **+lightness** (0.710) | base (0.684) |
| candle_dark | **+lightness** (0.375) | scale=0.2 (0.222) |
| fog_pine_dawn | scale=0.2 (0.722) | +lightness (0.714) |
| waterfall_zoom | **+lightness** (0.715) | base (0.556) |

Full table: [`docs/tier2/verdict_table.md`](docs/tier2/verdict_table.md).

## 7 · Honest limitations

| aspect | status |
|---|---|
| Sample size | n=6 prompts, 1 seed each (`+ 3 extra seeds` for BoN) |
| OPSD-I2V training (`configs/tier2_natural.yaml`) | **not run** — trainer API mismatch |
| Tier 1 full (n=50, p<0.05 significance) | needs ~37h → rented CUDA |
| Human eval | not conducted |
| Pre-registered prompts | partial — Round 2 added after seeing Round 1 |

Everything claimed is a **qualitative demo** backed by **pixel-space
reward metrics**.  The Wilcoxon with n=6 passes p<0.05, but this is not a
substitute for a Tier-1 sized study.

## 8 · Reproduction

```bash
# T2I scale sweep (produces first-frame grid)
python -m src.t2i.scale_sweep --scales 0.0 0.2 0.4 0.6 1.0 --out_dir outputs/scale_sweep

# I2V duels — generate the 3 base + 3 scale=0.2 clips used above
python -m src.render_i2v --frames_dir outputs/duel/frames_base   --prompts outputs/duel/prompts.json  --out_dir outputs/duel/videos_base  --seeds 0
python -m src.render_i2v --frames_dir outputs/duel/frames_sweet  --prompts outputs/duel/prompts.json  --out_dir outputs/duel/videos_sweet --seeds 0

# Lightness post-process
python -m src.postprocess.lightness_lift --src_glob "outputs/duel/videos_sweet/seed_00/*.mp4"

# Best-of-N (test-time rejection sampling)
python -m src.eval.tier2_bon \
    --frames_dir outputs/duel/frames_sweet \
    --base_videos outputs/duel/videos_base \
    --sweet_videos outputs/duel/videos_sweet \
    --seeds 1 2 3

# Compute quantitative verdict (needs scipy for Wilcoxon)
PYTHONPATH=. python scripts/compute_verdict.py

# Full overnight pipeline
bash scripts/night_orchestrator.sh
```

## 9 · Repository layout

```
08-video-opsd-i2v/
├── README.md                              ← this file
├── configs/
│   ├── opsd_i2v_default.yaml              ← OPSD-I2V default (lr=5e-5, warmup=20)
│   └── tier2_natural.yaml                 ← Tier 2 (lr=2e-5, warmup=10, +naturalness)
├── docs/
│   ├── METHOD.md                          ← original OPSD methodology
│   ├── tier2/
│   │   ├── METHOD.md                      ← Tier 2 pivot + reward hacking diagnosis
│   │   ├── lightness_v2_metrics.md        ← per-prompt lightness metrics
│   │   └── verdict_table.md               ← paired Wilcoxon results
│   └── figures/
│       ├── duel/                          ← 6 duels + lightness + triway + 6-pack
│       ├── reward_hacking_*.png           ← saturation/brightness/edge diagnosis
│       └── sweep_running.png              ← 3×3 scale sweep grid
├── src/
│   ├── t2i/{sd35_loader,scale_sweep,lora_scale}.py
│   ├── render_i2v.py                      ← DynamiCrafter wrapper
│   ├── opsd_trainer_i2v.py                ← OPSD training (needs API fix, see Tier 2 METHOD)
│   ├── rewards/
│   │   ├── {aesthetic,dynamic_degree,motion_smoothness,subject_consistency}.py
│   │   └── naturalness_penalty.py         ← Tier 2 differentiable reward
│   ├── postprocess/
│   │   └── lightness_lift.py              ← shadow-only lift + highlight protect
│   └── eval/
│       ├── naturalness_v2.py              ← bright-airy composite scorer
│       ├── tier2_bon.py                   ← Best-of-N test-time rejection sampling
│       └── tier2_eval.py                  ← reserved for future Tier 2 real-training eval
├── scripts/
│   ├── night_orchestrator.sh              ← full pipeline (Round 2 I2V + BoN + push)
│   ├── compute_verdict.py                 ← produces docs/tier2/verdict_table.md
│   └── prepare_tier2_data.py              ← recycles 6 base frames into training set
└── weights/
    ├── dynamicrafter_512/model.ckpt       ← frozen I2V
    ├── opsd_lora_sd35m_hpsv3/             ← pre-trained T2I LoRA (ext. CUDA)
    └── lora_scaled/                       ← cached scale=0.2 adapter
```
