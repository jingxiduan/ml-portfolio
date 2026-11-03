# Long-run training record

**Status:** V2 stopped by guard; NoNat control completed; V3 stabilization preflight pending.

## Research question

The project does not ask only whether a differentiable reward can be optimized. It asks:

> Can temporal-LoRA post-training improve held-out I2V motion while preserving subject identity and pixel naturalness, and can that optimization remain stable beyond a short probe?

A training reward increase is insufficient. A successful run must pass numerical-stability, held-out automatic, and human-perception gates.

## Fixed evaluation hierarchy

1. **Optimization signal:** finite reward gradients and non-zero LoRA updates.
2. **Numerical stability:** no loss/gradient/shift guard violation over the full run.
3. **Held-out metrics:** evaluation prompts and first frames are disjoint from training.
4. **Perceptual evidence:** matched-seed blind preference, reported separately from reward metrics.

Only levels 1–2 have partial positive evidence. Current artifacts do not establish a held-out quality improvement.

## 500-step ablation result

| Configuration | Appeal | Naturalness | Dynamic | Decision |
|---|---:|---:|---:|---|
| Base | 0.7295 | 0.9515 | 0.5075 | Reference |
| V1 hard clamp | 0.7294 | 0.9515 | 0.5074 | Dead naturalness gradient |
| V2 sigmoid | 0.7291 | 0.9513 | 0.5069 | Gradient repaired; no held-out gain |
| NoNat | 0.7296 | 0.9512 | 0.5079 | No meaningful held-out gain |

The reported V2 naturalness change from -1.47 to -0.79 is a **training reward component**, not an external naturalness improvement.

## 2,000-step NoNat control

Artifact: `outputs/opsd_long_no_nat/train_metrics.jsonl`.

- Completed all 2,000 steps and wrote the final checkpoint.
- Mean `loss_plus` was 0.00660 over the first 20 steps and 0.00640 over the final 20.
- Mean shift ratio moved from 0.0463 to 0.0415.
- LoRA-B norm reached 0.9875.
- The run demonstrates numerical stability without the naturalness component; it does not demonstrate perceptual quality and remains vulnerable to reward hacking.

## 2,000-step V2 attempt

Artifacts:

- `outputs/opsd_long_v2/train_metrics.jsonl`
- `outputs/opsd_long_v2/GUARD_STOP.json`

The run stopped at step 306 under the pre-registered guard. The 20-step mean of `loss_plus` increased from 0.00660 initially to 0.10858 before termination; the guard recorded a 10.95× increase over its baseline window. LoRA-B norm reached 2.13 and the naturalness component remained active, but this cannot be interpreted as successful learning because the finite-step fit had already destabilized.

### Diagnosis

The historical objective was

\[
\lVert\hat y-y^+\rVert_2^2-\lVert\hat y-y^-\rVert_2^2.
\]

The subtraction of two quadratic distances is not lower-bounded as a function of \(\hat y\). During a long run it can continue rewarding motion away from the negative target without an explicit behavior anchor. The V2 reward path also mapped decoded pixels through a hard clamp before computing naturalness, giving zero pixel-mapping derivatives outside the display range.

These are algorithmic stability problems, not evidence that 2,000 steps merely need to be rerun unchanged.

## V3 stabilization intervention

V3 keeps the V2 dataset, LoRA, learning rate, `c_adv`, EMA schedule, and reward weights fixed. It changes two diagnosed mechanisms:

1. **Bounded anchored fitting**

\[
\mathcal L_{\mathrm{V3}}
=\lVert\hat y-y^+\rVert_2^2
+\lambda_{\mathrm{anchor}}\lVert\hat y-y_0\rVert_2^2,
\qquad \lambda_{\mathrm{anchor}}=0.25.
\]

The negative target is retained as a diagnostic but is no longer an unbounded repulsive objective.

2. **Smooth pixel mapping**

The naturalness reward uses `(tanh(video) + 1) / 2` rather than a hard `[-1,1]` clamp so extreme decoded pixels retain a gradient.

Configurations:

- `configs/probe_v3_stable.yaml`: 100-step preflight.
- `configs/long_v3_stable.yaml`: 2,000-step run, launched only after the preflight passes.

## V3 pre-registered gates

| Gate | Threshold | Action on failure |
|---|---|---|
| Finite metrics | No NaN/Inf in loss, reward, gradient, shift, or LoRA norm | Stop |
| Positive-fit growth | Rolling `loss_plus` mean ≤10× initial baseline | Stop |
| Parameter gradient | Rolling `param_grad_norm` mean ≤20× initial baseline | Stop |
| Target displacement | Rolling `shift_ratio` mean ≤0.25 | Stop |
| Checkpointing | Steps 50 and 100 present in preflight | Do not promote |
| Long-run completion | Step 2,000 checkpoint and metrics present | Otherwise report incomplete |
| Held-out quality | Confidence interval over held-out prompts excludes zero in the favorable direction | Otherwise no quality claim |
| Human preference | Blind matched-seed preference exceeds chance with uncertainty interval | Otherwise no perceptual claim |

## Exact launch commands

```bash
cd 08-video-opsd-i2v
source .venv/bin/activate
export PYTHONPATH="$PWD"

mkdir -p outputs/probe_v3_stable
python -m src.train_entry --method opsd_i2v --config configs/probe_v3_stable.yaml \
  > outputs/probe_v3_stable/train.log 2>&1 &
TRAIN_PID=$!
python ../tools/train_guard.py \
  --kind i2v \
  --log outputs/probe_v3_stable/train_metrics.jsonl \
  --pid "$TRAIN_PID" \
  --poll 5 \
  --stall-sec 1200
wait "$TRAIN_PID"
```

The command, exit state, guard record, metric summary, and checkpoint inventory must be appended here before V3 is described as completed.

## Claim boundary

At present the project establishes a working I2V reward-gradient path, diagnoses dead gradients and reward hacking, and records a genuine long-run instability. It does **not** yet establish superior I2V generation quality.
