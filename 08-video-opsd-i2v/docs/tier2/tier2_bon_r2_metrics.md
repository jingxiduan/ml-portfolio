# Tier 2 — Best-of-N (test-time rejection sampling)

Candidates: scale=0.2 T2I + base I2V, varying seed. Picked per-prompt via
`appeal = 0.7·naturalness + 0.3·motion_score` (same thresholds as training).

## Per-prompt results

| prompt | best_seed | base_nat | s=0.2(s0)_nat | **BoN_nat** | motion_delta_vs_s0 |
|---|---|---|---|---|---|
| candle_dark | 1 | 0.000 | 0.223 | **0.287** | -0.004 |
| fog_pine_dawn | 1 | 0.860 | 0.892 | **0.945** | +0.244 |
| waterfall_zoom | 1 | 0.667 | 0.503 | **0.586** | -0.048 |
