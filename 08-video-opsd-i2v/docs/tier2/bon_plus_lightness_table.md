# Tier 2 — BoN winner → lightness post-process (6 prompts)

Column key: `nat` = naturalness_v2 score; `appeal` = 0.6·nat + 0.25·motion + 0.15·bonus_lift.

| prompt | BoN nat | **BoN+lightness nat** | Δnat | BoN appeal | **BoN+lightness appeal** | Δappeal |
|---|---|---|---|---|---|---|
| sun_rays | 0.998 | **1.000** | +0.002 | 0.849 | **0.850** | +0.001 |
| houses_hillside | 0.532 | **0.826** | +0.293 | 0.569 | **0.748** | +0.179 |
| pagoda_snowy | 0.602 | **0.824** | +0.223 | 0.641 | **0.775** | +0.134 |
| candle_dark | 0.287 | **0.538** | +0.251 | 0.259 | **0.396** | +0.137 |
| fog_pine_dawn | 0.945 | **1.000** | +0.055 | 0.815 | **0.783** | -0.032 |
| waterfall_zoom | 0.586 | **0.806** | +0.220 | 0.589 | **0.648** | +0.059 |
| **mean (n=6)** |  |  | **+0.174** |  |  | **+0.080** |

## What this means

- BoN selects the best seed on `appeal`; lightness then rebalances luminance/sat/edges.
- Δnat and Δappeal here are **not** independent of the reward (`naturalness_v2` is
  explicitly engineered to reward shadow-lift + non-overexposed highlights).
- The honest reading: if the pipeline is *required* to maximise `naturalness_v2`,
  BoN+lightness is the end-to-end answer. For human preference, see `verdict_table.md`
  (lightness vs base/scale on paired Wilcoxon with n=6).

Companion GIFs: `docs/figures/duel/fourway_<title>.gif` (base | s=0.2 | BoN | BoN+lightness).
