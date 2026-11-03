# 08 · Video OPSD-I2V · Project status

**Updated:** 2026-11-03

**Validated milestone:** reward-path and failure diagnosis

**Active milestone:** V4 gradient-controlled long-run stabilization

## Current conclusion

The project implements an end-to-end OPSD adaptation for DynamiCrafter I2V temporal modules and demonstrates that its LoRA parameters receive a differentiable video-reward signal. It also exposes two failures: reward hacking at strong adapter scale and numerical drift in the naturalness-constrained long run.

Current artifacts do **not** establish better held-out I2V quality than the frozen base model.

## Evidence matrix

| Question | Evidence | Status |
|---|---|---|
| Does I2V conditioning reach the model? | Corrected `c_crossattn`/`c_concat` path and API tests | Established |
| Does video reward backpropagate through the VAE? | FP32 reward decode and non-zero gradient probes | Established |
| Does the smooth naturalness term remain active? | V2 training component changes from -1.47 to -0.79 | Established as an internal signal only |
| Does 500-step V2 beat base on held-out prompts? | Appeal 0.7291 vs 0.7295; naturalness 0.9513 vs 0.9515 | No |
| Can the unconstrained control run for 2,000 steps? | `opsd_long_no_nat` final metrics/checkpoint | Yes |
| Can naturalness-constrained V2 run for 2,000 steps? | Guard stop at step 306, 10.95× `loss_plus` growth | No |
| Is V3 stable and better? | 100-step preflight passed, but the long run stopped at step 522 on a >20× parameter-gradient gate | No |
| Does V4 gradient clipping stabilize the same objective? | Single-variable `grad_clip=1e-3` run registered; step 600 first promotion point | Active |
| Do humans prefer the result? | No blind matched-seed evaluation | Unresolved |

## Long-run artifacts

### NoNat control

- Completed 2,000 steps.
- Final LoRA-B norm: 0.9875.
- `loss_plus` remained near its initial scale.
- This is a numerical control, not a naturalness result.

### V2 naturalness run

- Intended length: 2,000 steps.
- Stopped at step 306 by `tools/train_guard.py`.
- Initial 20-step mean `loss_plus`: 0.00660.
- Final 20-step mean before stop: 0.10858.
- Guard record: `outputs/opsd_long_v2/GUARD_STOP.json`.

## V3 intervention

V3 keeps V2's data, LoRA, learning rate, target radius, EMA schedule, and reward weights. It changes only the diagnosed unstable mechanisms:

1. Replaces attraction-minus-repulsion with a bounded positive-target fit plus a behavior-anchor penalty.
2. Replaces hard clipping in the naturalness pixel mapping with a smooth `tanh` mapping so extreme decoded pixels retain gradients.

The 100-step production-resolution preflight completed without a guard trip, but its cosine schedule reached zero learning rate at step 100. The corresponding long run remained near `1e-4`, developed progressively larger fit loss and gradients, and was stopped at step 522. V4 keeps all V3 conditions fixed except effective parameter-gradient clipping at `1e-3`; raw and post-clip norms are logged separately.

Files:

- `configs/probe_v3_stable.yaml`
- `configs/long_v3_stable.yaml`
- `configs/long_v4_gradclip.yaml`
- `tests/test_stability_objective.py`
- `docs/RESULTS_LONG.md`

## Promotion gates

1. The 100-step preflight must finish with finite metrics and checkpoints at steps 50 and 100.
2. `loss_plus` and parameter-gradient rolling means must stay below their registered multipliers.
3. Mean shift ratio must remain below 0.25.
4. Only then may the 2,000-step V3 configuration run.
5. A completed run must still pass held-out n≥40 evaluation and blind preference testing before any quality claim.

## Artifact policy

Every optimization run must use `tools/train_guard.py`, a unique output directory, and a Markdown record containing the exact config, command, metric summary, checkpoint inventory, and gate decision. Small records and metrics belong on `main`; large videos and model artifacts use the documented backup location.

See [`docs/RESULTS_LONG.md`](docs/RESULTS_LONG.md) for the detailed record and equations.
