# Duel Comparisons — OPSD-I2V

Side-by-side comparisons for the OPSD portfolio. Target: **"camera-like,
natural, vivid"** video from the DynamiCrafter + SD3.5-M stack.

## 📐 Legend

| label | what it is |
|---|---|
| **base** | SD3.5-M (unmodified) T2I  +  DynamiCrafter (unmodified) I2V |
| **scale=0.2** | SD3.5-M + **OPSD-LoRA @ 0.2** T2I  +  base I2V. Found via scale sweep. |
| **OPSD(1.0)** | SD3.5-M + full OPSD-LoRA T2I.  **Reward-hacked** (over-saturated, over-exposed). |
| **tier2_i2v** | base T2I + **Tier-2 I2V-LoRA** (lr=2e-5, warmup 20%, +naturalness penalty). |
| **stacked** | scale=0.2 T2I + Tier-2 I2V. The best combo. |

## 📁 Round 1 — initial sweet-spot demo  (3 prompts)

- `duel_sun_rays.gif / .mp4`
- `duel_houses_hillside.gif / .mp4`
- `duel_pagoda_snowy.gif / .mp4`

Side-by-side **base | scale=0.2**. Found that OPSD@1.0 overshoots (over-saturated)
and scale=0.2 recovers the saturation while keeping some polish.

- `triway_sunrays_base_0.2_1.0.gif` — three-way **base | 0.2 | 1.0** showing the hacking gradient
- `triway_pagoda_base_0.2_1.0.gif`

## 📁 Round 2 — robustness probe  (3 new prompts)

- `duel_candle_dark.gif / .mp4`           — low-light / soft colour
- `duel_fog_pine_dawn.gif / .mp4`         — low-saturation
- `duel_waterfall_zoom.gif / .mp4`        — camera motion + fine detail

Picks edge-cases that **test** the hypothesis that scale=0.2 is robust (not
only works on the first 3 cherry-picked prompts).

## 📁 T2I-only teasers  (static PNG, generated instantly without I2V cost)

- `t2i_duel_6pack.png`                           — all 6 prompts in one grid
- `t2i_duel_round1_t2i_*.png`                    — per-prompt zoomed
- `t2i_duel_round2_t2i_*.png`

## 📁 Tier 2 — reward-redesigned OPSD-I2V training  (NEW)

Rationale for Tier 2 (see `../../configs/tier2_natural.yaml`):

- **Diagnosis**: LAION-Aesthetic reward is a saturation-biased predictor and
  the direct cause of the OPSD-T2I reward hacking.
- **Fix**: drop LAION-Aesthetic entirely from the reward combo; add a
  **differentiable naturalness penalty** (saturation / overexpose / sobel-p95
  thresholds) with weight 0.5; double motion-smoothness; lower learning rate
  (`5e-5 → 2e-5`) and keep warmup at 20%.

### `tier2_ablation_*.gif / .mp4`  (**I2V-only ablation**)
Same base-T2I first frame; swap the I2V model.
- **left**: base-I2V          (vanilla DynamiCrafter)
- **right**: tier2-I2V        (new reward-redesigned LoRA)

Isolates the Tier-2 effect at the I2V stage.

### `tier2_stacked_*.gif / .mp4`  (**best vs base**)
- **left**: pure base               (base-T2I + base-I2V)
- **middle**: scale=0.2 T2I         (scale=0.2 T2I + base-I2V)
- **right**: full stack             (scale=0.2 T2I + tier2-I2V)

This is **the headline comparison** — the final answer to "better than base?".
Metrics table in `../tier2/tier2_metrics.md`.

## 🧪 How to reproduce

```bash
# T2I scale sweep
python -m src.t2i.scale_sweep --prompts ...  --scales 0.0 0.2 0.4

# I2V duels
python -m src.render_i2v --frames_dir outputs/duel/frames_base ...

# Tier 2 training
python -m src.train_entry --method opsd_i2v --config configs/tier2_natural.yaml

# Tier 2 eval
python -m src.eval.tier2_eval --tier2_lora outputs/tier2/lora_step00050
```

The full night pipeline is scripted in `scripts/night_orchestrator.sh`.
