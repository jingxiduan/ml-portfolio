# Tier 2 — Best-of-N (test-time rejection sampling)

Candidates: scale=0.2 T2I + base I2V, varying seed. Picked per-prompt via
`appeal = 0.7·naturalness + 0.3·motion_score` (same thresholds as training).

## Per-prompt results

| prompt | best_seed | base_nat | s=0.2(s0)_nat | **BoN_nat** | motion_delta_vs_s0 |
|---|---|---|---|---|---|
| sun_rays | 1 | 0.960 | 0.932 | **0.978** | +0.000 |
| houses_hillside | 3 | 0.471 | 0.405 | **0.432** | +0.000 |
| pagoda_snowy | 3 | 0.590 | 0.221 | **0.399** | +0.000 |
