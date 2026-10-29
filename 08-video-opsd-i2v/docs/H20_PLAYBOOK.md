# H20 Playbook — Tier-2 real training in a 10-hour window

**Target:** 2026-10-30 00:00–10:00, 1× NVIDIA H20 (96GB), 1 卡时/h

## Timeline (what should happen when)

| T+ | step | expected duration | checkpoint |
|---|---|---|---|
| 0:00 | ssh in, `nvidia-smi`, `cd /path/to/clone` | 2 min | H20 confirmed |
| 0:02 | `git clone` portfolio repo | 1 min | sources on H20 |
| 0:03 | `bash scripts/h20_setup.sh` | 30–45 min | smoke pass on production shape |
| 0:50 | `bash scripts/h20_run_tier2.sh` (background with `nohup`) | **4–6 h** | 5 LoRA checkpoints saved to `outputs/opsd_tier2_h20/lora_step{010,020,030,040,050}/` |
| 5:00 | train should finish around here | — | 📍 **primary deliverable: a trained Tier-2 LoRA** |
| 5:00 | `bash scripts/h20_infer_compare.sh` | 1–1.5 h | 6 three-way GIFs (base \| scale=0.2 \| tier2_lora) |
| 6:30 | `git add -A && git commit && git push` | 10 min | 📍 **secondary deliverable: results in portfolio** |
| 6:40 | **1h 20min buffer** for debugging / re-runs / cherry-picking the best ckpt step | — | — |

## Pre-flight checklist (before 10/30 00:00)

- [x] This playbook committed to portfolio (`docs/H20_PLAYBOOK.md`)
- [x] `scripts/h20_setup.sh` idempotent, downloads DynamiCrafter-512 ckpt
- [x] `scripts/h20_run_tier2.sh` saves LoRA every 10 steps
- [x] `scripts/h20_infer_compare.sh` renders 6 I2V with new LoRA and builds comparison
- [x] `scripts/smoke_trainer_e2e.py` supports `SMOKE_SHAPE=production` env var
- [x] First-frame PNGs under `outputs/duel*/frames_*/seed_00/` are committed to repo
      (~3MB; required so H20 does not have to re-run SD3.5-M T2I)

## The three commands

```bash
# -------- 1. one-shot setup (30-45 min) --------
cd ~ && git clone git@github.com:jingxiduan/ml-portfolio.git
cd ml-portfolio/08-video-opsd-i2v
bash scripts/h20_setup.sh

# -------- 2. training (background, 4-6 h) --------
nohup bash scripts/h20_run_tier2.sh > logs/h20_tier2.nohup 2>&1 &
tail -f logs/h20_tier2.log

# -------- 3. after training: inference + comparison + metrics (1-1.5 h) --------
bash scripts/h20_infer_compare.sh

# -------- 4. push results --------
git add -A outputs/opsd_tier2_h20/ docs/tier2/tier2_lora_metrics.* docs/figures/duel/tier2_lora_*
git commit -m "08: Tier-2 LoRA trained on H20 (50 steps, 6-prompt eval)"
git push
```

## Risk register

| risk | mitigation |
|---|---|
| CUDA/driver mismatch surfaces an API bug not seen on MPS | setup.sh runs `smoke_trainer_e2e.py` with `SMOKE_SHAPE=production` before training; fails fast |
| Weight download blocked by firewall / HF rate limit | setup.sh retries; manual fallback: use modelscope mirror (`snapshot_download(..., repo_source="modelscope")`) |
| Per-step wall-clock >> 7 min (training won't fit in window) | `run_tier2.sh` prints per-step timing; if first 3 steps avg > 8 min, Ctrl-C and lower `num_steps` to 25 |
| OOM at production shape even on H20 (unlikely at 96GB but possible with activation checkpointing bugs) | lower `inner_fit_steps=1`, enable `torch.utils.checkpoint`, or fall back to `resolution=[256,448]` |
| Lose data on resource reclaim at 10:00 | `run_tier2.sh` saves ckpts every 10 steps; `infer_compare.sh` saves per-prompt immediately; final `git push` lands on remote well before 10:00 |

## What success looks like (what we expect to commit)

```
08-video-opsd-i2v/
  outputs/opsd_tier2_h20/
    lora_step00050/
      adapter_config.json
      adapter_model.safetensors         (~15 MB)
    training.log                        (per-step loss / reward)
  docs/tier2/
    tier2_lora_metrics.md               (6 prompts × base/s=0.2/tier2 appeal scores)
    tier2_lora_metrics.json
  docs/figures/duel/
    tier2_lora_sun_rays.gif             (3-way)
    tier2_lora_houses_hillside.gif
    tier2_lora_pagoda_snowy.gif
    tier2_lora_candle_dark.gif
    tier2_lora_fog_pine_dawn.gif
    tier2_lora_waterfall_zoom.gif
```

Combined with the existing `bon_r2_table.md` + `bon_plus_lightness_table.md`
+ `verdict_table.md`, this gives three independent Tier-2 output tracks:
- **BoN** (inference-time selection) — already shipped
- **lightness** (post-process) — already shipped
- **trained LoRA** (gradient-based) — new, from H20

## Why 50 steps and not more

- OPSD trust-region updates are expensive (one full on-policy rollout per step),
  but strongly data-efficient compared to vanilla PG; 50 steps is the design
  budget written into `configs/tier2_natural.yaml`.
- At ~5 min/step on H20 (production shape, 50 inference steps per rollout),
  50 steps ≈ 4.2 h — comfortably inside the 10h window with room for inference.
- Checkpoints every 10 steps give us 5 Tier-2 LoRAs to pick from if the final
  one overshoots; `infer_compare.sh` defaults to the latest but takes any dir
  as `$1`.
