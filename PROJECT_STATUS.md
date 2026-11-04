# Portfolio status

**Updated:** 2026-11-04

The portfolio is organized around one theme: **reward-driven machine-learning systems, from prediction and control to multimodal generative-model post-training**.

## Project matrix

| # | Project | Current evidence | State |
|---|---|---|---|
| 01 | Multi-task CTR/CVR ranking | CTR AUC 0.8805; CVR AUC 0.7960 on documented proxy labels | Complete |
| 02 | RTB bid shading and control | Avazu pCTR AUC 0.7796; PID outperforms fixed policy in the documented simulator | Complete |
| 03 | Reward model and DPO pipeline | RM AUC 0.6150; 2,624 filtered DPO pairs; DPO run is a smoke test | Complete pipeline |
| 04 | Image reward post-training | Scenes: independent CLIP-H delta +0.102, prompt-cluster 95% CI [-0.087, +0.305]; portraits: +0.076, CI [-0.113, +0.263], n=40 × 3 seeds | Rebuilt with leakage audit |
| 06 | Reserved slot | No public project or result claimed | Reserved |
| 07 | Talking-head motion OPSD | Amplitude targets reached; absolute-jitter gate failed; calibrated expression study active | Flagship, active |
| 08 | I2V OPSD | Reward path works; smooth-naturalness and anchored-fit diagnostics complete; gradient-controlled run complete | Flagship, active |

## Active work

### 04 · Image reward post-training

- The previous DrawBench row split was retired after a prompt-leakage audit.
- Current manifests pin separate Pick-a-Pic training prompts and PartiPrompts dev/test prompts; exact and shared 8-token-shingle overlap are zero across train/dev/test/showcase.
- Scene and portrait domains are trained separately with SDXL-Turbo, dual LoRA adapters, bounded reward-gradient targets, and EMA behavior refresh.
- Final test uses 40 prompts per domain × 3 matched seeds and an independent CLIP-H/14 evaluator.
- Portraits show a small positive delta; scenes are statistically inconclusive and are not claimed as improved.

Record: [`04-diffusion-opsd-mps/README.md`](04-diffusion-opsd-mps/README.md).

### 07 · Calibrated expression dynamics

- Same-unit calibration completed on 77 windows from 40 sampled CelebV/HDTF clips.
- Frozen target: 0.3929 normalized expression RMS.
- Frozen tolerance: 0.2071.
- Calibrated seeds 0/1/2 completed 600 steps without a guard trip.
- On n=40 paired held-out clips, mean target error fell from 0.1202 to 0.0229–0.0238; pose high-frequency RMS was 0.75–0.77× base.
- 160 strict four-second renders and n=40 paired SyncNet evaluation are complete; all seeds pass relative non-inferiority, while absolute lip-sync quality remains poor/unresolved.
- Independent perceptual and blind-preference evaluation remain pending.

Record: [`07-talking-head-motion-opsd/docs/RESULTS_R3.md`](07-talking-head-motion-opsd/docs/RESULTS_R3.md).

### 08 · Stabilized I2V OPSD

- The 2,000-step no-naturalness control completed; the smooth-naturalness objective stopped at step 306 and the anchored fit at step 522.
- Effective `1e-3` gradient clipping stabilized the anchored objective and completed 2,000 steps without a guard trip.
- The stability candidate was selected from a frozen 4-checkpoint × 3-scale grid on six development pairs.
- On 40 untouched confirmatory pairs, appeal decreased by 0.0042 (95% CI [-0.0062, -0.0024]), naturalness was tied, and dynamics decreased by 0.0079 (95% CI [-0.0115, -0.0048]).
- The gradient-controlled run establishes long-run numerical stability but rejects held-out quality improvement for this configuration.
- A unified randomized 80-trial blind-study UI and participant-cluster scorer are ready under `human-study/`; real participant responses remain absent.

Record: [`08-video-opsd-i2v/docs/RESULTS_LONG.md`](08-video-opsd-i2v/docs/RESULTS_LONG.md).

## Evidence policy

- Internal training reward is labeled separately from held-out quality.
- Failed gates remain in the headline record.
- Every optimization run uses a unique output directory and `tools/train_guard.py`.
- Exact configuration, command, metrics, checkpoints, and failure reason are recorded in Markdown.
- Small records belong on `main`; large media, checkpoints, and extracted datasets use a backup branch when retained.

## Claim boundary

The repository demonstrates applied research and post-training ability. It does not claim production deployment, foundation-model pretraining, or ownership of the reused base models and datasets. Project 04's previous row-level prompt split and headline metric were retired after a leakage audit; the current report uses pinned, disjoint manifests and an independent evaluator.
