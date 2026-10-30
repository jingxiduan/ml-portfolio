# 08 -- OPSD-I2V: Project Status and Continuation Guide

> Last updated: 2026-10-30 10:48 -- pre-expiry backup

---

## On Next Boot

```bash
cd ~/ml-portfolio/08-video-opsd-i2v
git pull
source .venv/bin/activate   # rebuild if venv is gone
export PYTHONPATH="$PWD"
```

---

## 1. Data Inventory

### Training Data (in git)
| Data | Count | Path | Notes |
|------|-------|------|-------|
| Train prompts | 160 | `data/prompts/tier2_train_200.json` | 15 scene categories, GPT-generated |
| Eval prompts | 40 | `data/prompts/tier2_eval_40.json` | Held-out, not used in training |
| Train first frames | 200 | `data/first_frames_200/*.png` | SDXL-Turbo 512x320 |
| Eval first frames | 40 | `data/first_frames_eval/*.png` | SDXL-Turbo 512x320 |
| Legacy prompts | 6 / 894 | `data/prompts/tier2_train.json` / `train.json` | From Tier-1 phase |

### LoRA Checkpoints (in git)
| Experiment | Steps | Path | Size |
|------------|-------|------|------|
| V1 (nat_clamp) | 500 | `outputs/opsd_tier2/lora_step00500/` | 13MB |
| V2 (nat_sigmoid) | 500 | `outputs/opsd_tier2_v2/lora_step00500/` | 13MB |
| NoNat (ablation) | 500 | `outputs/opsd_tier2_no_nat/lora_step00500/` | 13MB |

### Training Metrics (in git)
- 500 lines JSON per experiment: `outputs/opsd_tier2{,_v2,_no_nat}/train_metrics.jsonl`
- Fields: step, reward, loss, R_subject_consistency, R_dynamic_degree, R_motion_smoothness, R_naturalness_penalty, grad_norm, lr

### Training Curve Plots (in git)
- Per-experiment: `outputs/opsd_tier2{,_v2,_no_nat}/figures/fig_reward_curves.png` etc (3 each)
- 3-way comparison: `outputs/figures_comparison/comparison_rewards.png`, `naturalness_evolution.png`, `reward_vs_gpu_hours.png`

### Eval Videos (local only, mp4 gitignored -- lost on server expiry)
| Config | Rendered | Path |
|--------|----------|------|
| Base (no LoRA) | 40/40 done | `outputs/eval/videos_base/seed_00/` |
| V1 (nat_clamp) | 10/40 partial | `outputs/eval/videos_v1_nat_clamp/seed_00/` |
| V2 (nat_sigmoid) | 10/40 partial | `outputs/eval/videos_v2_nat_sigmoid/seed_00/` |
| V2 alpha=32 | 10/10 done | `outputs/eval/videos_v2_alpha32/seed_00/` |
| V2 alpha=6 | 10/10 done | `outputs/eval/videos_v2_alpha6/seed_00/` |
| NoNat (ablation) | 10/40 partial | `outputs/eval/videos_ablation_no_nat/seed_00/` |

Note: mp4 files are in `.gitignore` and will be **lost when server expires**. Re-render with commands below.

### GIF Comparisons (in git)
- 32 GIFs in `docs/figures/duel/`: duel x6, fourway x6, lightness x6, tier2_lora x6, tier2_bon x6, triway x2 (6 legacy scenes)
- 10 GIFs in `docs/figures/tier2_new/`: 3-way Base/alpha=6/alpha=32 (10 new eval scenes)
- 10 GIFs in `docs/figures/eval_compare/`: Base vs V2 side-by-side (10 eval scenes)

### Eval Metrics (in git)
- `outputs/eval/metrics_summary.json` -- 4 configs x 3 metrics (based on n=10)
- `outputs/eval/metrics_table.md`

---

## 2. Key Experimental Findings

### 3-Way Ablation Training Results
| Experiment | Final Reward | R_naturalness | Finding |
|------------|-------------|---------------|---------|
| V1 (ReLU+clamp) | +2.16 | **0.000** | Gradient 100% dead, naturalness constraint ineffective |
| **V2 (sigmoid)** | -0.50 | **-0.786** | Gradient alive, naturalness improved -1.47 to -0.79 |
| NoNat (ablation) | +2.36 | N/A | Pure reward hacking baseline |

### Tier-1 LoRA Scale Sweep (T2I)
- Scale=0.2 is the sweet spot for T2I LoRA: improved dynamics without over-saturation
- Scale=1.0 causes reward hacking: saturation x1.95, overexposure x1.65, edge x1.86

### 8 DynamiCrafter API Bug Fixes
| # | Bug | Fix |
|---|-----|-----|
| 1 | conditioning dropped in UNetModel.forward | `_call_unet` helper |
| 2 | `vae.decode(z).sample` crashed | `_vae_decode_video` helper |
| 3 | `encode(x).latent_dist.sample()` crashed | `encode(x).sample()` |
| 4 | PEFT target modules regex mismatch | Runtime enumeration |
| 5 | Missing `c_crossattn` + `c_concat` | `CondStage` dataclass |
| 6 | OpenCLIP hard-coded CUDA | Patch device attribute |
| 7 | `IndexError` in `timestep_embedding` | Auto-broadcast |
| 8 | Shape mismatch in `rollout_until` | Use `c_concat.shape[2]` |

---

## 3. Next Steps (in order)

### Step 1: Re-render eval videos (~30 min GPU)

V1/V2/NoNat only rendered 10/40 each. Complete the remaining 30.

```bash
# Recreate seed_00 symlinks (renderer requires this structure)
mkdir -p data/first_frames_eval/seed_00
for f in data/first_frames_eval/*.png; do
  ln -sf "$(realpath $f)" "data/first_frames_eval/seed_00/$(basename $f)"
done

# Render remaining videos (~8 min per config)
for tag_lora in \
  "videos_v1_nat_clamp outputs/opsd_tier2/lora_step00500" \
  "videos_v2_nat_sigmoid outputs/opsd_tier2_v2/lora_step00500" \
  "videos_ablation_no_nat outputs/opsd_tier2_no_nat/lora_step00500"; do
  tag=$(echo $tag_lora | cut -d' ' -f1)
  lora=$(echo $tag_lora | cut -d' ' -f2)
  python -m src.render_i2v \
    --frames_dir data/first_frames_eval \
    --prompts data/prompts/tier2_eval_40.json \
    --out_dir outputs/eval/$tag \
    --seeds 0 --num_frames 16 --fps 8 --limit 40 \
    --i2v_lora_dir $lora
done
```

### Step 2: Generate more GIF comparisons

```bash
# Existing script for Tier-2 3-way GIFs
python scripts/make_tier2_gifs.py

# For Tier-1 T2I scale=0.2, need T2I LoRA from project 07
# Then: python scripts/generate_first_frames.py --lora_scale 0.2 ...
```

### Step 3: Increase training to 2000+ steps

500-step LoRA L2 norm is only ~27, eval metrics show no significant change. Need 2000+:

```bash
# Edit configs/tier2_v2.yaml: max_train_steps: 500 -> 2000
python -m src.train_entry --method opsd_i2v --config configs/tier2_v2.yaml
```

### Step 4: Recompute eval metrics with n=40

After rendering all 40 videos per config, re-run evaluation.

### Step 5: Update README with new GIFs and metrics

### Step 6 (optional): Tune reward parameters
- `dynamic_degree` cap_tau=0.6 is too low (saturates in 50 steps) -- increase to 0.85+
- `motion_smoothness` is unreliable under high noise -- add sigma_threshold
- Consider adding FVD/FID standard video metrics

---

## 4. Environment Dependencies

```
# H20 GPU environment
- NVIDIA H20 80GB
- Python 3.10 + .venv
- PyTorch 2.x + CUDA
- DynamiCrafter (in third_party/, gitignored)
- RAFT (in third_party/, gitignored)
- Base model weights in weights/ (gitignored, ~4GB)

# To rebuild:
pip install -r requirements.txt
# Clone DynamiCrafter and RAFT into third_party/
# Download base model weights (see README "Reproduce" section)
```

---

## 5. File Structure

```
08-video-opsd-i2v/
  README.md                               -- GitHub front page (GIFs + curves + metrics)
  STATUS.md                               -- this file (progress + continuation guide)
  requirements.txt
  configs/                                training configs (V1/V2/NoNat + baselines)
    tier2.yaml                            V1 (nat_clamp)
    tier2_v2.yaml                         V2 (nat_sigmoid) -- primary
    tier2_ablation_no_nat.yaml            ablation
  src/
    opsd_trainer_i2v.py                   training loop (OPSD RL)
    render_i2v.py                         inference rendering
    rewards/                              4 reward functions
    postprocess/                          lightness post-processing
    eval/                                 naturalness_v2 evaluator
  scripts/
    generate_prompts_200.py               generate 200 prompts
    generate_first_frames.py              SDXL-Turbo first frame generation
    make_tier2_gifs.py                    Tier-2 3-way comparison GIFs
    make_eval_gifs.py                     Base vs V2 comparison GIFs
    plot_training_curves.py               per-experiment training curves
    plot_comparison.py                    3-way comparison curves
    render_eval_all.sh                    batch rendering script
  data/
    prompts/                              5 JSON prompt files
    first_frames_200/                     200 train first frames (35MB)
    first_frames_eval/                    40 eval first frames (8.7MB)
  outputs/
    opsd_tier2/                           V1: metrics + figures + LoRA
    opsd_tier2_v2/                        V2: metrics + figures + LoRA
    opsd_tier2_no_nat/                    NoNat: metrics + figures + LoRA
    figures_comparison/                   3-way comparison plots (3 PNGs)
    eval/                                 eval metrics + videos (mp4 gitignored)
  docs/figures/
    duel/                                 32 GIFs -- Tier-1 comparisons (6 scenes)
    tier2_new/                            10 GIFs -- Tier-2 alpha sweep (10 new scenes)
    eval_compare/                         10 GIFs -- Base vs V2 (10 eval scenes)
```
