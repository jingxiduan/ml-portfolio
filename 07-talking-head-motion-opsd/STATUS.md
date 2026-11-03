# 07 · Talking-Head Motion OPSD · Project status

**Updated:** 2026-11-03

**Validated milestone:** V4 amplitude-target study

**Active milestone:** R3 expression dynamics

## Current conclusion

The project establishes that reward-guided LoRA post-training can move JoyVASA head-motion amplitude toward measured human target bands. It does not yet establish a general improvement in talking-head naturalness.

The strongest positive result is the held-out amplitude shift: three policies reach 0.85–0.89× their target bands without the earlier 2.4–2.7× overshoot. The main negative result is equally important: the pre-registered absolute-jitter gate failed, and lip synchronization remains unmeasured by an independent metric.

## Evidence matrix

| Question | Evidence | Status |
|---|---|---|
| Can OPSD update motion-space LoRA parameters? | Multiple 600-step runs and saved V1/V2/V4 adapters | Established |
| Can bounded targets avoid amplitude overshoot? | Finance, real-estate, and short-video held-out evaluation | Established |
| Are the learned policies distinguishable? | Four-way identification accuracy 0.831 | Established |
| Do real industries have the same separation? | Real-reference permutation and effect-size analysis | Mostly rejected; finance and real-estate overlap |
| Is absolute jitter controlled? | Registered threshold 18.36; naive OPSD result 28.97 | Failed |
| Is lip synchronization preserved? | `jaw_audio_corr` is unreliable even for base; no SyncNet/LSE-C | Unresolved |
| Does behavior cloning improve perception? | 1,753-window loader and qualitative prototype | Unresolved |
| Does V5 control non-lip expression coefficients? | Three calibrated seeds; n=40 paired held-out clips; target-error CIs exclude zero | Established |
| Does V5 improve perceived naturalness? | Matched render exists; no independent perceptual metric or blind preference | Unresolved |

## R3 expression-dynamics state

### Completed

- Added a peaked 0.3–3 Hz non-lip expression reward.
- Logged expression value and active-gradient fraction.
- Completed a 600-step seed-0 probe.
- The probe's expression reward increased from a first-100 mean of 0.7523 to a last-100 mean of 0.8108.
- `expr_active_frac` remained 1.0, so the optimization-signal gate passed.
- R3 expression calibration is complete on 77 windows: target 0.3929, tolerance 0.2071.
- Calibrated seeds 0/1/2 completed 600 steps each without a guard trip.
- Held-out n=40 paired coefficient evaluation passed target-error and pose high-frequency gates.
- Same-portrait/audio/seed four-way rendering is complete.

### Not completed

- Independent perceptual evaluation of the matched renders.
- Independent lip-sync metric.
- Blind human preference test.

The exact protocol and claim boundary are in [`docs/RESULTS_R3.md`](docs/RESULTS_R3.md).

## Key artifacts

| Artifact | Purpose |
|---|---|
| `docs/TECH_REPORT.md` | Full method and historical results |
| `docs/SEPARATION_ANALYSIS.md` | Statistical and ecological-validity analysis |
| `docs/GATE_LOG.md` | Registered gate outcomes |
| `docs/RESULTS_R3.md` | Current V5 training and completion record |
| `outputs/industry_eval.json` | Held-out industry-policy evaluation |
| `outputs/naturalness_eval_r3.json` | Base versus trained diagnostic summary |
| `outputs/opsd_v5_finance_smoke/metrics.jsonl` | Existing 600-step V5 probe |
| `scripts/calibrate_behaviour_bands.py` | Same-unit real-video expression calibration |

## Artifact policy

Small metrics, configurations, figures, and Markdown records belong on `main`. Large checkpoints, media, and downloaded datasets remain outside `main`; retained copies use the documented backup branch when available. Every new optimization run must use `tools/train_guard.py`, a unique output directory, and a Markdown record that includes the exact command, configuration, gate decision, and failure reason.

## Next gate order

1. Run same-unit expression calibration and freeze the output JSON.
2. Train seeds 0/1/2 without changing the frozen gate.
3. Evaluate coefficient behavior and absolute jitter on held-out audio.
4. Run independent lip-sync and matched rendered evaluation.
5. Run blind preference testing only if the automatic safety gates pass.

No perceptual-improvement claim should be added before steps 3–5 pass.
