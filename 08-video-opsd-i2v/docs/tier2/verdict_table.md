# Quantitative Verdict — base vs scale=0.2 vs scale=0.2+lightness

## Mean across 6 prompts

| condition | lum | sat | overexp | sob_p95 | shadow% | nat | motion | **appeal** |
|---|---|---|---|---|---|---|---|---|
| base | 0.390 | 0.438 | 0.187 | 33.8 | 31.6% | 0.625 | 0.791 | **0.578** |
| 0.2 | 0.391 | 0.490 | 0.206 | 43.9 | 35.4% | 0.590 | 0.850 | **0.572** |
| +lightness | 0.432 | 0.321 | 0.119 | 29.1 | 17.8% | 0.805 | 0.768 | **0.680** |

## Pairwise comparisons (per-prompt deltas)

| prompt | base | scale=0.2 | +lightness | Δ(0.2−base) | Δ(light−0.2) | Δ(light−base) |
|---|---|---|---|---|---|---|
| sun_rays | 0.838 | 0.812 | **0.840** | -0.026 | +0.028 | **+0.002** |
| houses_hillside | 0.592 | 0.553 | **0.726** | -0.039 | +0.173 | **+0.133** |
| pagoda_snowy | 0.684 | 0.569 | **0.710** | -0.115 | +0.140 | **+0.026** |
| candle_dark | 0.150 | 0.222 | **0.375** | +0.072 | +0.153 | **+0.225** |
| fog_pine_dawn | 0.646 | 0.722 | **0.714** | +0.076 | -0.008 | **+0.068** |
| waterfall_zoom | 0.556 | 0.552 | **0.715** | -0.004 | +0.163 | **+0.159** |

## Paired Wilcoxon (one-sided, light > ref)

- appeal(**+lightness**) > appeal(**base**):     statistic=21.00, p=0.016
- appeal(**+lightness**) > appeal(**scale=0.2**): statistic=20.00, p=0.031

(n=6; p<0.05 → consistent win across prompts. This is a very small sample — see Tier 1 full run for proper significance.)

## Per-prompt winner (max appeal)

| prompt | winner | runner-up | gap |
|---|---|---|---|
| sun_rays | **+lightness** (0.840) | base (0.838) | +0.002 |
| houses_hillside | **+lightness** (0.726) | base (0.592) | +0.133 |
| pagoda_snowy | **+lightness** (0.710) | base (0.684) | +0.026 |
| candle_dark | **+lightness** (0.375) | scale=0.2 (0.222) | +0.153 |
| fog_pine_dawn | **scale=0.2** (0.722) | +lightness (0.714) | +0.008 |
| waterfall_zoom | **+lightness** (0.715) | base (0.556) | +0.159 |
