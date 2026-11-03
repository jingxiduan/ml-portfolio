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
| P1 · optimization signal | `expr_active_frac > 0.05` and finite reward gradients during a 600-step run | Passed on seeds 0/1/2; no guard trip |
| P2 · coefficient behavior | Held-out expression dynamics move toward the calibrated band without pose high-frequency RMS exceeding `1.3×` base | Passed on seeds 0/1/2, n=40 paired clips |
| P3 · rendered behavior | Matched portrait/audio/seed evaluation shows improvement on independent metrics | Matched four-way render complete; independent perceptual metric pending |
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

The provisional probe used target/tolerance `1.0/1.0`; its target lies above the R3 p90 and must not be treated as a calibrated human target. Confirmatory seeds 0/1/2 used `0.3928757608/0.2070943713` and separate guarded output directories.

## Confirmatory training result

All three 600-step runs completed without a guard trip and wrote final adapters.

| Seed | Expression reward, first→last 100 | Total reward, first→last 100 | Jitter reward, first→last 100 | Decision |
|---:|---:|---:|---:|---|
| 0 | 0.7767 → 0.9835 | 1.6543 → 2.2347 | -0.3052 → -0.3209 | Pass optimization gate |
| 1 | 0.7749 → 0.9816 | 1.6438 → 2.2031 | -0.3086 → -0.3388 | Pass optimization gate |
| 2 | 0.7818 → 0.9808 | 1.6576 → 2.1720 | -0.3108 → -0.3365 | Pass optimization gate |

Training reward remains an internal diagnostic. The confirmatory claim comes from held-out paired coefficients below.

## Held-out coefficient result

Artifact: `outputs/expression_eval_r3.json`. The evaluation uses 40 speaker-disjoint audio clips, identical audio and generation seed for base and each adapter, and paired bootstrap intervals over target-error deltas.

| Variant | Expression RMS mean | Mean absolute target error | Δ error vs base, 95% CI | Pose high-frequency RMS | Ratio vs base |
|---|---:|---:|---:|---:|---:|
| Base | 0.2727 | 0.1202 | reference | 0.2406° | 1.000 |
| Seed 0 | 0.3872 | 0.0229 | -0.0973 [-0.1065, -0.0869] | 0.1801° | 0.749 |
| Seed 1 | 0.3849 | 0.0238 | -0.0964 [-0.1055, -0.0856] | 0.1822° | 0.757 |
| Seed 2 | 0.3860 | 0.0234 | -0.0968 [-0.1068, -0.0856] | 0.1848° | 0.768 |

All seeds pass the registered coefficient target-error and pose high-frequency gates. The expression metric is intentionally the same statistic used by the reward and calibration, so this proves controlled coefficient behavior, not perceptual realism. It also does not retroactively clear the earlier V4 `jitter_hp` failure, which used a different evaluation definition.

## Matched rendering

![base and calibrated seeds](../outputs/expression_demo_r3/expression_r3_fourway.gif)

The four clips use the same portrait, held-out audio, and generation seed. They are a diagnostic artifact, not a blind preference result. Individual MP4 files and `manifest.json` are retained with the experiment backup.

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
