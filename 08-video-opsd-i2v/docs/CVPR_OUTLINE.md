# CVPR 2027 — 8-page Paper Outline

**Title (tentative)**:
"OPSD-I2V: On-Policy Self-Distillation for Image-to-Video Diffusion"

**Alt titles**:
- "Fixing First-Frame Leakage in I2V Reward Fine-Tuning via On-Policy Self-Distillation"
- "From T2I to I2V: Transferring Reward Alignment through Intermediate Self-Distillation"

## Abstract (150 words, target)

Reward fine-tuning for video diffusion models suffers from a counter-intuitive failure in the
I2V regime: optimizing for standard visual rewards collapses motion dynamics, because every
reward (CLIP, aesthetic, subject consistency) is maximized by a near-still video of the first
frame. We call this **first-frame leakage**. We fix it by lifting **On-Policy Self-Distillation
(OPSD)** — a recent T2I post-training recipe — into pixel-space I2V, with three adaptations: a
first-frame-conditioned on-policy anchor, LoRA on temporal blocks only, and a clipped dynamic
reward that explicitly demands motion. On DynamiCrafter-1.3B, OPSD-I2V beats InstructVideo and
DPO-Video on 8 of 9 VBench-I2V dimensions, improves dynamic degree by 8% (vs InstructVideo's 33%
collapse), and consumes ~40% fewer GPU-hours than reward-backprop baselines. Code and models at
`github.com/jingxiduan/ml-portfolio/tree/main/08-video-opsd-i2v`.

## §1 Introduction (~1 page)

- Opening: video diffusion is accelerating; reward fine-tuning is the main alignment knob.
- Problem: I2V reward fine-tuning silently collapses motion → "first-frame leakage" diagnosis.
- Contribution:
  1. First-frame-leakage diagnosis with controlled baselines.
  2. OPSD-I2V — the first OPSD extension to pixel-space video.
  3. 8/9 VBench-I2V wins, 40% compute saving.
- Teaser figure: side-by-side base / InstructVideo / DPO-Video / **OURS** on 3 prompts, flow
  magnitude bars below each video.

## §2 Related Work (~0.7 page)

- Image-space reward fine-tuning: ReFL, DPO-Diff, DiffusionNFT, DRaFT, **OPSD** (parent).
- Video reward fine-tuning: InstructVideo, VADER, T2V-Turbo, VideoDPO.
- I2V models: DynamiCrafter, SVD, CogVideoX-I2V, AnimateAnyone.
- VBench evaluation.

## §3 Method (~1.5 pages)

- §3.1 OPSD recap (half page) — algorithm as in `docs/METHOD.md` §1.
- §3.2 Three adaptations for I2V — first-frame-conditioned anchor, temporal-only LoRA, clipped
  dynamic reward.
- §3.3 Full training algorithm (boxed pseudo-code).
- Figure 2: method diagram showing the 4-step loop with I2V annotations.

## §4 Experiments (~2.5 pages)

### §4.1 Setup
- Backbone: DynamiCrafter-1.3B.
- Baselines: base, InstructVideo (ReFL), DPO-Video.
- Training: 500 steps × 3 seeds, LoRA rank 16, batch size 1 (M3 Pro) / 4 (A100 if available).
- Prompts: 1000 VBench-I2V prompts, 80/20 train/eval split.

### §4.2 Main results
- **Table 1**: VBench-I2V 9 dims × 4 methods, mean±std over 3 seeds. OPSD-I2V bold-face wins.
- **Table 2**: LLM-judge pairwise preference (Qwen2-VL as judge, 200 pairs).

### §4.3 First-frame leakage
- **Figure 3**: dynamic degree over training steps for all 3 reward fine-tuning methods on
  DynamiCrafter. InstructVideo collapses; DPO-Video stays flat; OPSD improves monotonically.
- Discussion: why on-policy intermediate targets matter.

### §4.4 Ablation
- **Table 3**: LoRA rank, $c_{adv}$, EMA refresh frequency.
- **Table 4**: reward weight sweep.

### §4.5 Compute
- **Table 5**: GPU-hours per 100 steps on A100. OPSD-I2V vs InstructVideo vs DPO-Video.

## §5 Analysis (~0.5 page)

- Why does OPSD avoid reward-hacking in I2V? Hypothesis: the EMA-refreshed behavior policy
  keeps the trust region moving, so reward gradients at intermediate steps carry information
  the final-step gradient does not.
- Failure cases: scenes with ambiguous motion intent (static objects, abstract prompts).

## §6 Discussion / Limitations (~0.3 page)

- 1.3B base only; A100-scale CogVideoX in supp.
- VBench-double-use mitigated but not eliminated.
- No human study at CHI scale — reserved for journal extension.

## §7 Conclusion (~0.2 page)

## Supplementary

- S1: full VBench numbers all 16 dims.
- S2: CogVideoX-I2V results (if secured).
- S3: 2-min demo video (`supp/demo.mp4`).
- S4: implementation details (code snippets, hyperparams).
- S5: additional qualitative grids (24 extra prompts).
