# R3 experiment record: expression dynamics and unfinished gates

**Status:** active; this document separates completed evidence from work that has not yet passed its gate.

## Problem definition

The V4 policies changed head-pose amplitude but left most non-lip expression dimensions without a direct objective. The resulting question is narrower than general talking-head realism:

> Can a measured, bounded expression-dynamics reward increase non-lip facial motion toward real-video statistics without increasing head jitter or degrading lip synchronization and rendered naturalness?

The intervention is `LivelinessRewardV5`: one peaked Gaussian term over the 0.3–3 Hz RMS of non-lip expression coefficients. All V4 reward terms and optimizer settings remain unchanged for attribution.

## Pre-registered gates

| Gate | Success criterion | State |
|---|---|---|
| P0 · calibration | Target and tolerance computed from extracted real-video coefficients in the same normalized units | Passed: 77 windows from 40 sampled clips |
| P1 · optimization signal | `expr_active_frac > 0.05` and finite reward gradients during a 600-step run | Passed on the provisional probe; calibrated run active |
| P2 · coefficient behavior | Held-out expression dynamics move toward the calibrated band without worsening absolute jitter beyond `1.3×` base | Not run |
| P3 · rendered behavior | Matched portrait/audio/seed evaluation shows improvement on independent metrics | Not run |
| P4 · lip preservation | SyncNet/LSE-C or equivalent does not regress relative to base | Not run |
| P5 · robustness | At least three training seeds satisfy P2–P4 with uncertainty intervals | Not run |

A training reward increase alone cannot pass P2 or P3.

## Existing 600-step V5 probe

Artifact: `outputs/opsd_v5_finance_smoke/metrics.jsonl`.

Configuration recorded by the run:

- 600 steps, seed 0, batch size 8;
- LoRA rank/alpha 8/8 on the JoyVASA motion DiT;
- AdamW, peak learning rate `1e-4`, 200-step warmup and cosine decay;
- expression target/tolerance `1.0/1.0` in normalized units;
- no behavior-cloning samples in this probe.

| Metric | First 100-step mean | Last 100-step mean | Interpretation |
|---|---:|---:|---|
| Total training reward | 1.6287 | 2.0943 | Optimization signal increased |
| Expression reward | 0.7523 | 0.8108 | Moved toward the provisional target |
| Expression active fraction | 1.0000 | 1.0000 | P1 gradient-activity gate passed |
| Amplitude reward | 0.8405 | 0.9632 | Pose term also changed |
| Jitter reward | -0.3030 | -0.3157 | No evidence of jitter improvement |
| Audio-motion coupling | 0.0091 | 0.1976 | Increased on the training batches |
| OPSD loss | 0.01022 | 0.01024 | Stable but nearly flat |

### What this probe proves

- The new expression term is differentiable and active under the sampled states.
- A 600-step run is numerically stable under the recorded configuration.
- The training objective can increase its own expression component.

### What this probe does not prove

- The target `1.0` was provisional, not yet regenerated from the checked-in coefficient dataset.
- The final LoRA was recovered from `origin/backup/2026-11-02-full` and its serialized arguments match the 600-step log; rendered held-out evaluation is still pending.
- Reward components are not independent perceptual metrics.
- There is no multi-seed evidence, independent lip-sync metric, or blind preference result.

The probe is therefore reported as an optimization check, not as a quality improvement.

## R3 same-unit calibration

Artifact: `outputs/behaviour_bands_r3.json`.

The reproducible R3 subset samples the first 20 available CelebV-HQ clips and first 20 HDTF clips, yielding 77 non-overlapping four-second coefficient windows. It is adequate for fixing a narrow experiment target but is not a population estimate.

| Statistic | Expression RMS |
|---|---:|
| p10 | 0.2341 |
| p25 | 0.3175 |
| median / frozen target | **0.3929** |
| p75 | 0.5246 |
| p90 | 0.6503 |
| frozen tolerance (IQR) | **0.2071** |

The provisional probe used target/tolerance `1.0/1.0`; its target lies above the R3 p90 and must not be treated as a calibrated human target. Confirmatory seeds 0/1/2 were launched with `0.3928757608/0.2070943713` and separate guarded output directories.

## Calibration and completion protocol

1. Freeze the R3 median target and IQR-based tolerance; do not retune them after viewing confirmatory results.
2. Train seeds 0, 1, and 2 through `tools/train_guard.py`, writing each run to a unique output directory.
3. Preserve each exact command, `train.log`, `guard.log`, `metrics.jsonl`, adapter, and gate decision.
4. Evaluate on held-out speakers and audio, never on the windows used for calibration or behavior cloning.
5. Report base-versus-trained confidence intervals for expression dynamics, absolute jitter, amplitude, EAR, and an independent lip-sync metric.
6. Render matched portrait/audio/seed comparisons and run a blinded human preference study before making a perceptual claim.

## Current blockers

- The 77-window R3 calibration is a deterministic convenience subset rather than a random or demographically balanced sample; broader recalibration is required for a population claim.
- SyncNet/LSE-C is not installed or integrated.
- The available public real-video coefficient set is small and should not be described as evidence for universal industry-specific behavior.

## Reporting decision

Until P2–P5 are complete, the main README describes V5 as an unfinished expression-dynamics experiment. The validated headline remains the earlier amplitude-target result together with its failed absolute-jitter gate.
