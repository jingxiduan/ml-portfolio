# Lightness v2 — Per-prompt metrics

Scorer: `src/eval/naturalness_v2.py` — penalises **oversaturation**, **overexposure**, **over-sharpening**, **grey/muddy** and **flat dynamic range**; rewards **mid-high luminance** (bonus_lift).


`appeal = 0.60·naturalness + 0.25·motion_score + 0.15·bonus_lift`


| prompt | condition | lum | sat | overexp | sob_p95 | shadow% | nat | motion | **appeal** |
|---|---|---|---|---|---|---|---|---|---|
| sun_rays | base | 0.453 | 0.237 | 0.040 | 15.4 | 11.2% | 0.980 | 1.000 | **0.838** |
| sun_rays | scale=0.2 | 0.389 | 0.447 | 0.021 | 17.8 | 15.7% | 0.936 | 1.000 | **0.812** |
| sun_rays | **+lightness** | 0.411 | 0.360 | 0.009 | 13.0 | 3.4% | 0.983 | 1.000 | **0.840** |

| houses_hillside | base | 0.444 | 0.561 | 0.368 | 44.2 | 21.7% | 0.571 | 1.000 | **0.592** |
| houses_hillside | scale=0.2 | 0.466 | 0.598 | 0.397 | 56.5 | 21.8% | 0.505 | 1.000 | **0.553** |
| houses_hillside | **+lightness** | 0.488 | 0.455 | 0.227 | 35.6 | 1.2% | 0.793 | 1.000 | **0.726** |

| pagoda_snowy | base | 0.612 | 0.230 | 0.347 | 75.1 | 8.5% | 0.673 | 1.000 | **0.684** |
| pagoda_snowy | scale=0.2 | 0.669 | 0.303 | 0.487 | 94.1 | 17.5% | 0.482 | 1.000 | **0.569** |
| pagoda_snowy | **+lightness** | 0.672 | 0.224 | 0.304 | 64.9 | 0.9% | 0.716 | 1.000 | **0.710** |

| candle_dark | base | 0.085 | 0.756 | 0.089 | 4.2 | 86.5% | 0.000 | 0.601 | **0.150** |
| candle_dark | scale=0.2 | 0.053 | 0.523 | 0.015 | 2.9 | 92.4% | 0.223 | 0.354 | **0.222** |
| candle_dark | **+lightness** | 0.175 | 0.187 | 0.010 | 2.3 | 74.5% | 0.504 | 0.292 | **0.375** |

| fog_pine_dawn | base | 0.350 | 0.340 | 0.000 | 23.1 | 37.6% | 0.860 | 0.521 | **0.646** |
| fog_pine_dawn | scale=0.2 | 0.366 | 0.413 | 0.000 | 40.4 | 39.6% | 0.892 | 0.747 | **0.722** |
| fog_pine_dawn | **+lightness** | 0.409 | 0.226 | 0.000 | 25.6 | 13.0% | 0.978 | 0.511 | **0.714** |

| waterfall_zoom | base | 0.395 | 0.502 | 0.281 | 40.5 | 24.4% | 0.667 | 0.622 | **0.556** |
| waterfall_zoom | scale=0.2 | 0.405 | 0.655 | 0.313 | 51.6 | 25.3% | 0.503 | 1.000 | **0.552** |
| waterfall_zoom | **+lightness** | 0.434 | 0.472 | 0.165 | 33.4 | 13.8% | 0.855 | 0.807 | **0.715** |

