# Portfolio status

**Updated:** 2026-11-03

The portfolio is organized around one theme: **reward-driven machine-learning systems, from prediction and control to multimodal generative-model post-training**.

## Project matrix

| # | Project | Current evidence | State |
|---|---|---|---|
| 01 | Multi-task CTR/CVR ranking | CTR AUC 0.8805; CVR AUC 0.7960 on documented proxy labels | Complete |
| 02 | RTB bid shading and control | Avazu pCTR AUC 0.7796; PID outperforms fixed policy in the documented simulator | Complete |
| 03 | Reward model and DPO pipeline | RM AUC 0.6150; 2,624 filtered DPO pairs; DPO run is a smoke test | Complete pipeline |
| 04 | Image DiffusionOPSD | Held-out CLIPScore +0.542; 60% prompt-level win rate on n=20 | Mechanism reproduction |
| 05 | Text-to-video OPSD | Held-out reward +0.523; 77.8% win rate on n=9; training curve remains noisy | Proof of concept |
| 06 | Reserved slot | No public project or result claimed | Reserved |
| 07 | Talking-head motion OPSD | Amplitude targets reached; absolute-jitter gate failed; R3 expression study active | Flagship, active |
| 08 | I2V OPSD | Reward path works; held-out gain absent; V2 long run failed stability guard; V3 active | Flagship, active |

## Active work

### 07 · R3 expression dynamics

- Same-unit calibration completed on 77 windows from 40 sampled CelebV/HDTF clips.
- Frozen target: 0.3929 normalized expression RMS.
- Frozen tolerance: 0.2071.
- Calibrated seeds 0/1/2 completed 600 steps without a guard trip.
- On n=40 paired held-out clips, mean target error fell from 0.1202 to 0.0229–0.0238; pose high-frequency RMS was 0.75–0.77× base.
- Matched rendering is complete; independent perceptual, lip-sync, and blind-preference evaluation remain pending.

Record: [`07-talking-head-motion-opsd/docs/RESULTS_R3.md`](07-talking-head-motion-opsd/docs/RESULTS_R3.md).

### 08 · V3 stabilization

- The 2,000-step NoNat control completed.
- Naturalness-constrained V2 stopped at step 306 after a 10.95× rolling `loss_plus` increase.
- V3 replaces the unbounded contrastive quadratic with bounded anchored fitting and removes extreme-pixel hard-clamp dead zones.
- Unit tests and the 100-step production preflight pass; `loss_plus` remained at 0.00660→0.00681 with no guard trip.
- The V3 long run failed the fixed parameter-gradient gate at step 522; V4 changes only effective gradient clipping from 1.0 to 1e-3 and is pre-registered.
- Even a completed V4 run cannot produce a quality claim without held-out n≥40 evaluation and blind preference.

Record: [`08-video-opsd-i2v/docs/RESULTS_LONG.md`](08-video-opsd-i2v/docs/RESULTS_LONG.md).

## Evidence policy

- Internal training reward is labeled separately from held-out quality.
- Failed gates remain in the headline record.
- Every optimization run uses a unique output directory and `tools/train_guard.py`.
- Exact configuration, command, metrics, checkpoints, and failure reason are recorded in Markdown.
- Small records belong on `main`; large media, checkpoints, and extracted datasets use a backup branch when retained.

## Claim boundary

The repository demonstrates applied research and post-training ability. It does not claim production deployment, foundation-model pretraining, or ownership of the reused base models and datasets.
