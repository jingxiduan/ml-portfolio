# Tier 2 — Best-of-N (test-time rejection sampling)

Candidates: scale=0.2 T2I + base I2V, varying seed. Picked per-prompt via
`appeal = 0.7·naturalness + 0.3·motion_score` (same thresholds as training).

## Per-prompt results

| prompt | best_seed | base_nat | s=0.2(s0)_nat | **BoN_nat** | motion_delta_vs_s0 |
|---|---|---|---|---|---|
| candle_dark | 0 | 0.556 | 0.862 | **0.862** | +0.000 |
| fog_pine_dawn | 0 | 1.000 | 0.987 | **0.987** | +0.000 |
| waterfall_zoom | 0 | 0.617 | 0.432 | **0.432** | +0.000 |
