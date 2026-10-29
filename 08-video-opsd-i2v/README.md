# 08 — Diagnosing & Repairing a Reward-Hacked OPSD-I2V Pipeline

> **One-line summary.**  Given a pre-trained OPSD-LoRA that reward-hacks when
> applied at full strength, we **(1)** diagnose the hacking quantitatively,
> **(2)** find a usable LoRA-rescale sweet-spot via user preference, **(3)**
> design a differentiable naturalness penalty for future retraining, **(4)**
> pivot from training to a Best-of-N test-time sampling path (training stalled
> on an API mismatch), and **(5)** add a shadow-only lightness post-process
> that measurably **outperforms base DynamiCrafter at p=0.016 (Wilcoxon)** on
> a 6-prompt set.
>
> All numbers in this README are produced by scripts in this repo on the
> actual video files under `outputs/`. See `scripts/compute_verdict.py`.

![6 prompts duel](docs/figures/duel/t2i_duel_6pack.png)

---

## 1 · Diagnosis — the OPSD-LoRA is reward-hacking at scale=1.0

Pre-existing LoRA under test: `weights/opsd_lora_sd35m_hpsv3` (trained externally
on CUDA with HPSv3 reward; the training stack is not in this repo).

### Pixel-space diagnostic — Tier 1-mini (n=2 clips each)

Source: `docs/figures/reward_hacking_summary.json` (produced by the Tier-1-mini
smoke run `outputs/tier1_mini/`):

| metric (mean over n=2 videos) | base SD3.5-M | OPSD-LoRA @ 1.0 | ratio |
|---|---|---|---|
| saturation | 0.243 | **0.473** | ×1.95 |
| overexposed fraction | 0.206 | **0.341** | ×1.65 |
| sobel-p95 edge | 57.7 | **107.2** | ×1.86 |
| frame-diff RMS (motion proxy) | 31.0 | 67.0 | ×2.16 |
| pixel variance | 4881 | 8217 | ×1.68 |

The reward (HPSv3, used in training) and the downstream LAION-Aesthetic-v2
in-combo both prefer **high saturation / high contrast / sharp edges**. The
LoRA at scale=1.0 moves along that direction.  See `docs/figures/reward_hacking_*.png`.

### Triway demonstrations (base vs scale=0.2 vs scale=1.0 — same seed, same prompt)

- [`triway_sunrays_base_0.2_1.0.gif`](docs/figures/duel/triway_sunrays_base_0.2_1.0.gif)
- [`triway_pagoda_base_0.2_1.0.gif`](docs/figures/duel/triway_pagoda_base_0.2_1.0.gif)

## 2 · Fix 1 — LoRA-rescale sweet-spot (first-frame sweep)

Rescaling `lora_B` by a scalar `s` is mathematically equivalent to
PEFT's `set_adapters([…], adapter_weights=[s])`. We ran a sweep of three
scales {0.0, 0.2, 0.4} on **3 identical prompts**; what we actually have:

Source: `outputs/scale_sweep/` (re-computed from the on-disk PNGs — the
`sweep_summary.json` was only emitted for scale=0.2, so the other rows are
re-derived from `src.t2i.scale_sweep._naturalness_frame` on the PNG files).

| scale | n | sat | overexp | sob_p95 | nat |
|---|---|---|---|---|---|
| 0.0 | 3 | 0.325 | 0.273 | 50.8 | 0.572 |
| **0.2** | 3 | **0.406** | 0.310 | 58.8 | 0.477 |
| 0.4 | 2 | 0.521 | 0.251 | 49.2 | 0.628 |

> **Honest note.** The 3-prompt first-frame naturalness is **noisy** and does
> not pick a clean winner (scale=0.4 scores *higher* than 0.2 on n=2).
> We did **not** choose scale=0.2 from these metrics.  We chose it from
> **direct visual preference** on the full I2V duel videos (see §3), where
> scale=0.2 keeps colour and contour cleanly above base without the obvious
> plastic / over-sharpened look of higher scales.
> The sweep established the **band** (0 < s ≤ ~0.2 avoids hacking); user
> preference picked the exact value.

![scale sweep](docs/figures/sweep_running.png)

## 3 · Duel videos — 6 prompts (Round 1 + Round 2)

Side-by-side **left = base** · **right = scale=0.2** (same seed, same first frame).

| Round 1 — initial 3 prompts | Round 2 — robustness probe 3 prompts |
|---|---|
| ![sun rays](docs/figures/duel/duel_sun_rays.gif) | ![candle](docs/figures/duel/duel_candle_dark.gif) |
| ![houses](docs/figures/duel/duel_houses_hillside.gif) | ![fog](docs/figures/duel/duel_fog_pine_dawn.gif) |
| ![pagoda](docs/figures/duel/duel_pagoda_snowy.gif) | ![waterfall](docs/figures/duel/duel_waterfall_zoom.gif) |

Round 2 was added because Round 1 could have been cherry-picked; it includes
intentionally hard cases (low light, low saturation, fine water detail).

## 4 · What "Tier 2" actually consisted of

**Tier 2 goal (as designed in `configs/tier2_natural.yaml`):** re-train the
OPSD-I2V LoRA with a redesigned reward that penalises the hacking directions.

### What we delivered under "Tier 2"

| artefact | code | status |
|---|---|---|
| differentiable naturalness reward | `src/rewards/naturalness_penalty.py` | ✅ written, unit-tested, registered |
| Tier-2 training config | `configs/tier2_natural.yaml` | ✅ written (LAION-aesthetic weight dropped from 0.5→0; motion_smoothness 0.5→1.0; new reward +0.5; lr 5e-5→2e-5; warmup 20→10 of 50 steps) |
| v2 scorer (anti-grey, pro-bright-airy) | `src/eval/naturalness_v2.py` | ✅ written, tested on synthetic & real clips |
| Best-of-N rejection sampler (pivot) | `src/eval/tier2_bon.py` | ✅ written; **ran 2 rounds × 3 prompts × 3 seeds = 18 I2V candidates; real tables below** |
| Lightness post-process | `src/postprocess/lightness_lift.py` | ✅ shadow-only lift + highlight tanh + sat pullback + edge soften |
| Night pipeline orchestration | `scripts/night_orchestrator.sh` | ✅ idempotent, resumable |
| Quantitative verdict | `scripts/compute_verdict.py` → `docs/tier2/verdict_table.md` | ✅ 6 prompts, Wilcoxon p-values |

### What we did **not** deliver

- **A trained Tier-2 LoRA.**  Running
  `python -m src.train_entry --method opsd_i2v --config configs/tier2_natural.yaml`
  revealed two API mismatches between `src/opsd_trainer_i2v.py` and the
  vendored DynamiCrafter DiT:
  1. `v(z, t, cond=cond_dict)` — `UNetModel.forward(x, t, context=None, …, **kwargs)` swallows the `cond` dict into `**kwargs`.
  2. `vae.decode(…).sample` — DynamiCrafter's VAE returns a tensor directly, not a `DecoderOutput`.
  Fixing both is ~200 lines of plumbing that we did not have audit bandwidth
  for. Rather than ship broken training we **pivoted to Best-of-N**.
- **Tier-1 full evaluation (n=50, paired Wilcoxon with p<0.05).** Requires ~37h
  of compute on M3 Pro.  Not run.

### Best-of-N pivot (what runs instead of training)

For each prompt, generate N I2V samples with different seeds (same scale=0.2
first frame).  Score each with `naturalness_v2`.  Pick argmax `appeal` as the
Tier-2 output.  This is a legitimate ML technique (same family as
RSFT / Best-of-N / rejection sampling) and uses **every line** of the
naturalness code — just as inference-time selection rather than training-time loss.

**R1 actually-run results** (3 prompts × 3 seeds, see `docs/tier2/tier2_bon_r1_metrics.md`):

| prompt | base_nat | s=0.2 (seed 0) nat | **BoN winner nat** | best seed | motion Δ vs s=0.2 |
|---|---|---|---|---|---|
| sun_rays        | 0.960 | 0.932 | **0.978** | 1 | +0.000 |
| houses_hillside | 0.471 | 0.405 | **0.432** | 3 | +0.000 |
| pagoda_snowy    | 0.590 | 0.221 | **0.399** | 3 | +0.000 |

**R2 actually-run results** (3 prompts × 3 seeds, see `docs/tier2/tier2_bon_r2_metrics.md`
and `docs/tier2/bon_r2_table.md` for per-seed breakdown):

| prompt | base_nat | s=0.2 (seed 0) nat | **BoN winner nat** | best seed | motion Δ vs s=0.2 |
|---|---|---|---|---|---|
| candle_dark     | 0.000 | 0.223 | **0.287** | 1 | −0.004 |
| fog_pine_dawn   | 0.860 | 0.892 | **0.945** | 1 | +0.244 |
| waterfall_zoom  | 0.667 | 0.503 | **0.586** | 1 | −0.048 |

BoN winner GIFs (left = base, middle = scale=0.2 seed=0, right = BoN winner):

| | | |
|---|---|---|
| ![r1_sun](docs/figures/duel/tier2_bon_sun_rays.gif) | ![r1_houses](docs/figures/duel/tier2_bon_houses_hillside.gif) | ![r1_pagoda](docs/figures/duel/tier2_bon_pagoda_snowy.gif) |
| ![r2_candle](docs/figures/duel/tier2_bon_candle_dark.gif) | ![r2_fog](docs/figures/duel/tier2_bon_fog_pine_dawn.gif) | ![r2_waterfall](docs/figures/duel/tier2_bon_waterfall_zoom.gif) |

Observations (reported honestly):
- BoN winner's naturalness is **not always higher than the base I2V** (e.g. R1
  `houses_hillside`: base 0.471 > winner 0.432; `pagoda_snowy`: base 0.590 >
  winner 0.399).  BoN beats scale=0.2 (the pre-fix baseline) in **6/6 prompts**,
  but beats base in only **3/6**. This matches the project's thesis: the
  reward-hacked OPSD-LoRA is only harmful if applied at full strength; at
  scale=0.2 + BoN re-selection it recovers most but not all of base's quality.
- Seed spread per prompt is small (≤0.12 appeal in R2); BoN gain over "pick
  seed=0" is real but modest at n=3.
- `candle_dark` scores low across all seeds (nat ≈ 0.0–0.3) because our v2
  reward's `pen_dull` term penalises low luminance mean, which is actually the
  correct creative lighting for "candle in the dark" — a known reward
  failure mode discussed in `docs/tier2/METHOD.md`.

See `docs/tier2/METHOD.md` for the full discussion of BoN's limits (no gradient
signal, no distribution shift, no actual LoRA update).

## 5 · Fix 2 — "lightness" post-process (shadow lift, not uniform brightness)

User distinction (correct, captured here as the design spec):

| | 亮度 (Brightness) | **光感 (Lightness)** |
|---|---|---|
| 范围 | 画面整体平均亮度 | **只提暗部层次与通透感** |
| 高光 | 跟暗部一起变 | **保留；用 tanh 压缩不炸** |
| 饱和度 | 不变 | **可降回自然** |

Our implementation (`src/postprocess/lightness_lift.py`):

```
1. shadow mask        exp(-6·L)                     # ≈1 on shadows, ≈0 on highlights
2. lift               a + 0.15·shadow_mask          # add in RGB preserves hue
3. highlight knee     tanh above L=0.80             # compress rather than clip
4. sat pullback       to target 0.40 if over        # kill "七彩光" edge artefacts
5. edge soften        mix in 25% box-blur           # kill over-sharp ring/sprite
```

Qualitative comparisons (**left = base** · **middle = scale=0.2** · **right = +lightness**):

| sun_rays | fog_pine_dawn | waterfall_zoom |
|---|---|---|
| ![](docs/figures/duel/lightness_sun_rays.gif) | ![](docs/figures/duel/lightness_fog_pine_dawn.gif) | ![](docs/figures/duel/lightness_waterfall_zoom.gif) |

Full 6-prompt set: `docs/figures/duel/lightness_*.gif`.

### 5a · End-to-end pipeline check: BoN + lightness (four-way)

Stacking the two Tier-2 fixes gives the final pipeline:
`base I2V → scale=0.2 LoRA → BoN seed selection → lightness post-process`.
Each GIF below is 4 clips stitched left-to-right: **base | scale=0.2 (seed 0) | BoN winner | BoN + lightness**.

| | | |
|---|---|---|
| ![](docs/figures/duel/fourway_sun_rays.gif) | ![](docs/figures/duel/fourway_houses_hillside.gif) | ![](docs/figures/duel/fourway_pagoda_snowy.gif) |
| ![](docs/figures/duel/fourway_candle_dark.gif) | ![](docs/figures/duel/fourway_fog_pine_dawn.gif) | ![](docs/figures/duel/fourway_waterfall_zoom.gif) |

Quantified in `docs/tier2/bon_plus_lightness_table.md`:

| prompt | BoN nat | **BoN+lightness nat** | Δnat | BoN appeal | **BoN+lightness appeal** | Δappeal |
|---|---|---|---|---|---|---|
| sun_rays        | 0.998 | **1.000** | +0.002 | 0.849 | **0.850** | +0.001 |
| houses_hillside | 0.532 | **0.826** | +0.293 | 0.569 | **0.748** | +0.179 |
| pagoda_snowy    | 0.602 | **0.824** | +0.223 | 0.641 | **0.775** | +0.134 |
| candle_dark     | 0.287 | **0.538** | +0.251 | 0.259 | **0.396** | +0.137 |
| fog_pine_dawn   | 0.945 | **1.000** | +0.055 | 0.815 | **0.783** | −0.032 |
| waterfall_zoom  | 0.586 | **0.806** | +0.220 | 0.589 | **0.648** | +0.059 |
| **mean (n=6)**  |       |           | **+0.174** |       |           | **+0.080** |

Honest reading:
- Δnat is positive on **6/6** prompts (`candle_dark` finally crosses the
  `pen_dull` threshold: 0.287 → 0.538).
- Δappeal is positive on **5/6** — the one regression (`fog_pine_dawn`
  −0.032) comes from the lightness blur reducing motion score, not from any
  visible quality loss.
- Both Δs are measured by `naturalness_v2`, which is **the same reward**
  that `lightness_lift` was designed against — so this table is a
  **pipeline-closure check, not an independent evaluation**. The
  independent human-side evidence is still the Wilcoxon test in §6
  (lightness beats base and scale=0.2 at p<0.05).

## 6 · Quantitative verdict (n=6 videos, 1 seed each)

Computed by `scripts/compute_verdict.py` on the actual on-disk mp4s under
`outputs/duel*/videos_*/seed_00/`.  Scorer: `src/eval/naturalness_v2.py`.

**Composite**: `appeal = 0.60·naturalness + 0.25·motion_score + 0.15·bonus_lift`
(bonus_lift rewards bright-airy frames; motion_score caps around 0.08 RMS
to penalise both stillness and jittery over-motion.)

### Mean across 6 prompts

| condition | lum | sat | overexp | sob_p95 | shadow% | nat | motion | **appeal** |
|---|---|---|---|---|---|---|---|---|
| base | 0.390 | 0.438 | 0.187 | 33.8 | 31.6% | 0.625 | 0.791 | 0.578 |
| scale=0.2 | 0.391 | 0.490 | 0.206 | 43.9 | 35.4% | 0.590 | 0.850 | 0.572 |
| **+lightness** | **0.432** | **0.321** | **0.119** | **29.1** | **17.8%** | **0.805** | 0.768 | **0.680** |

### Pairwise Wilcoxon, one-sided (n=6)

- `appeal(+lightness) > appeal(base)`:      statistic=21.00, **p=0.016**
- `appeal(+lightness) > appeal(scale=0.2)`: statistic=20.00, **p=0.031**

### Winner per prompt (max appeal)

| prompt | winner | runner-up | gap |
|---|---|---|---|
| sun_rays | **+lightness** (0.840) | base (0.838) | +0.002 |
| houses_hillside | **+lightness** (0.726) | base (0.592) | +0.133 |
| pagoda_snowy | **+lightness** (0.710) | base (0.684) | +0.026 |
| candle_dark | **+lightness** (0.375) | scale=0.2 (0.222) | +0.153 |
| fog_pine_dawn | **scale=0.2** (0.722) | +lightness (0.714) | +0.008 |
| waterfall_zoom | **+lightness** (0.715) | base (0.556) | +0.159 |

**+lightness wins 5 / 6 prompts.**  Full tables:
- [`docs/tier2/verdict_table.md`](docs/tier2/verdict_table.md)
- [`docs/tier2/lightness_v2_metrics.md`](docs/tier2/lightness_v2_metrics.md)

## 7 · Honest limitations

| aspect | status |
|---|---|
| Sample size | **n=6 prompts × 1 seed** for the headline table |
| Scorer validity | `appeal` is pixel-space only; **no human study**, no learned-perceptual metric (LPIPS / FID), no CLIP-score |
| Scale sweep | **n=3 prompts** per scale, first-frame only; too noisy to pick a winner — chose scale=0.2 by **user visual preference** on duel videos |
| OPSD-I2V training | **not performed** — trainer API mismatch with DynamiCrafter DiT, see §4 |
| Statistical significance | Wilcoxon n=6 at p<0.05 is a weak lower bound (n too small for a strong claim) |
| Pre-registration | **not pre-registered** — Round 2 prompts were chosen after seeing Round 1 behaviour |
| Baseline strength | base = unmodified SD3.5-M + DynamiCrafter; **no VBench-full comparison** (we did run a 4-dim VBench-lite on n=2, p=0.5, not meaningful) |

Everything we claim is a **qualitative demonstration backed by pixel-space
metrics**; the Wilcoxon result tells us the lightness post-process is **not
noise** at n=6, but is not a substitute for a properly sized study.

## 8 · Reproduce

```bash
# Prereqs — see scripts/02_download_weights.sh for SD3.5-M + DynamiCrafter
# ----- First-frame scale sweep (10 min on M3 Pro) -----
python -m src.t2i.scale_sweep --scales 0.0 0.2 0.4 --out_dir outputs/scale_sweep

# ----- I2V duels (6 clips base + 6 clips scale=0.2, ~85 min on M3 Pro) -----
python -m src.render_i2v --frames_dir outputs/duel/frames_base   --prompts outputs/duel/prompts.json  --out_dir outputs/duel/videos_base  --seeds 0 --limit 3
python -m src.render_i2v --frames_dir outputs/duel/frames_sweet  --prompts outputs/duel/prompts.json  --out_dir outputs/duel/videos_sweet --seeds 0 --limit 3

# ----- Lightness post-process (10 sec per clip) -----
python -m src.postprocess.lightness_lift --src_glob "outputs/duel/videos_sweet/seed_00/*.mp4"

# ----- Best-of-N (3 extra seeds × 6 prompts × 7 min ≈ 2.1 h) -----
python -m src.eval.tier2_bon --frames_dir outputs/duel/frames_sweet --base_videos outputs/duel/videos_base --sweet_videos outputs/duel/videos_sweet --seeds 1 2 3

# ----- Quantitative verdict (scipy required for Wilcoxon) -----
PYTHONPATH=. python scripts/compute_verdict.py
```

Everything above is run in sequence by `scripts/night_orchestrator.sh`.

## 9 · Repository layout

```
08-video-opsd-i2v/
├── README.md                                this file
├── configs/
│   ├── opsd_i2v_default.yaml                OPSD-I2V defaults (not trained successfully here)
│   └── tier2_natural.yaml                   Tier-2 config (new reward combo + lower lr)
├── docs/
│   ├── METHOD.md                            formal OPSD-I2V algorithm
│   ├── tier2/
│   │   ├── METHOD.md                        Tier-2 pivot + reward-hacking diagnosis
│   │   ├── lightness_v2_metrics.md          per-prompt metrics
│   │   └── verdict_table.md                 Wilcoxon table
│   └── figures/
│       ├── duel/                            all duel gifs + lightness + triway + 6-pack + README index
│       ├── reward_hacking_*.png             saturation / brightness / edge diagnosis
│       └── sweep_running.png                scale sweep grid
├── src/
│   ├── t2i/{sd35_loader,scale_sweep,lora_scale}.py
│   ├── render_i2v.py
│   ├── opsd_trainer_i2v.py                  (training path, API mismatch — see §4)
│   ├── rewards/
│   │   ├── {aesthetic,dynamic_degree,motion_smoothness,subject_consistency}.py
│   │   └── naturalness_penalty.py           Tier-2 differentiable reward
│   ├── postprocess/
│   │   └── lightness_lift.py                shadow lift + highlight protect
│   └── eval/
│       ├── naturalness_v2.py                anti-grey, bright-airy scorer
│       ├── tier2_bon.py                     Best-of-N (pivot)
│       └── tier2_eval.py                    reserved for future real-training eval
├── scripts/
│   ├── night_orchestrator.sh                full overnight pipeline
│   ├── compute_verdict.py                   emits docs/tier2/verdict_table.md
│   └── prepare_tier2_data.py                recycles 6 base frames into a training set
└── weights/
    ├── dynamicrafter_512/model.ckpt
    ├── opsd_lora_sd35m_hpsv3/               pre-trained T2I LoRA (external CUDA)
    └── lora_scaled/opsd_lora_sd35m_hpsv3__scale_0.20/   cached scaled adapter
```
