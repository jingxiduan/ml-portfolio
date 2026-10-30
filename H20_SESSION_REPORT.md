# H20 GPU Session Report — Oct 29-30, 2026

## Hardware
- **GPU**: NVIDIA H100 80GB HBM3 × 2
- **CPU**: 42 cores, 350GB RAM
- **Disk**: 3.8TB SSD

## Session 1: Tier-2 Rendering (Oct 29)

### 08-video-opsd-i2v
- **Initial training**: 50 steps LoRA, 6 prompts, ~8.5min
- **Rendering**: 6 prompts × 3 variants (base/sweet/tier2_lora) = 18 videos
- **GIFs**: 6 three-way comparison GIFs
- **Metrics**: tier2 appeal 0.540 < base 0.638 (LoRA insufficient)

### 07-talking-head-motion-opsd
- **Rendering**: 20 clips × 6 variants = 120 videos
- **GIFs**: 20 three-way comparison GIFs
- **Issues**: head_jitter=256 (target ~120), jaw_audio_corr=-0.10 (target 0.3-0.5)

## Session 2: CVPR-Quality Training (Oct 29-30)

### Expanded Training Data
- **Prompts**: 6 → 200 (160 train + 40 eval) across 15 visual categories
- **First frames**: 200 SDXL-Turbo generated images (512×320)
- **Categories**: nature, urban, indoor, animals, weather, water, food, aerial, cinematic, macro

### 3-Way Ablation Study

| Config | Steps | Naturalness | Total Reward | Training Time |
|--------|-------|-------------|-------------|---------------|
| V1 (nat_clamp) | 500 | ReLU+clamp → dead gradient (R≈0.0) | +2.36 | 86 min |
| **V2 (nat_sigmoid)** | 500 | **Sigmoid → alive gradient (R: -1.47→-0.79)** | **-0.50** | **86 min** |
| NoNat (ablation) | 500 | None | +2.36 | 86 min |

### Key Technical Contribution
**Naturalness penalty gradient fix**: Replacing `ReLU + clamp(0,1)` with `sigmoid` in the 
saturation/overexposure/edge penalties ensures gradient always flows through the naturalness 
reward. Without this fix, the penalty collapses to 0.0 throughout training (100% of steps), 
providing zero learning signal for visual quality improvement.

### Evaluation (n=10 held-out prompts)
- All configs: appeal≈0.729, nat≈0.951, dynamic≈0.507
- 500 steps + lr=1e-5 insufficient for major eval metric shift (need 2000+ steps)
- Training curves clearly show reward trends and gradient dynamics

### Artifacts Produced
1. **3 LoRA checkpoints** (step 500, 13MB each)
2. **1500 training metrics** (3×500 steps, JSON lines with per-component rewards)
3. **9 training curve figures** (3 experiments × 3 plot types)
4. **6 comparison figures** (3-way ablation comparison, naturalness evolution, GPU-hours)
5. **40 eval videos** (4 configs × 10 prompts)
6. **Quantitative metrics table** (naturalness_v2)

### GPU Time Summary
| Task | GPU-hours |
|------|-----------|
| V1 training | 1.43 |
| V2 training | 1.43 |
| NoNat training | 1.43 |
| Eval rendering | 0.32 |
| First frame generation | 0.01 |
| **Total Session 2** | **4.62** |
| **Total Both Sessions** | **~6** |

## Files Changed
- `src/opsd_trainer_i2v.py` — Added per-step JSON metrics logging
- `src/rewards/naturalness_penalty.py` — **Fixed: sigmoid replaces ReLU+clamp**
- `configs/tier2_cvpr.yaml` — V1 config (500 steps, nat_weight=1.5)
- `configs/tier2_cvpr_v2.yaml` — V2 config (sigmoid naturalness, nat_weight=2.0)
- `configs/tier2_cvpr_ablation_no_nat.yaml` — Ablation (no naturalness)
- `scripts/generate_prompts_200.py` — 200 diverse I2V prompts
- `scripts/generate_first_frames.py` — SDXL-Turbo first frame generation
- `scripts/plot_training_curves.py` — Publication-quality training curve plots
- `scripts/plot_comparison.py` — 3-way ablation comparison figures
- `scripts/eval_full.py` — Full evaluation pipeline
- `scripts/render_eval_all.sh` — Batch eval rendering

## Next Steps (for next GPU session)
1. **Scale training**: 2000-5000 steps for visible eval metric improvement
2. **Better motion reward**: Replace RAFT-based motion_smoothness with VideoMAE or CLIP-temporal
3. **Higher dynamic degree**: Increase cap_tau from 0.6 to 0.8 to prevent saturation
4. **Add FVD/FID**: Standard video generation metrics for CVPR comparison
5. **Full eval**: 40 prompts × 3 seeds = 120 videos per config
6. **Gallery GIFs**: Side-by-side comparison animations for paper
