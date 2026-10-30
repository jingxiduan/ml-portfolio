# Gate Log — 07-talking-head-motion-opsd

| Time | Gate | Key Values | Result |
|------|------|------------|--------|
| 2026-10-30 18:40 | T2 v3 train 600 steps | reward=1.179, jitter=-0.427 (v2:-0.458), amp=0.781, cap_frac=0.12-0.33, guard=no trip | PASS |
| 2026-10-30 20:47 | T1 jitter diagnosis | cap 1.0 deg NEVER active: frac_above_cap=0.0 for ALL variants; base std_high p90=0.263, max=0.355; opsd p90=0.384; cap 0.2 deg activates on 28-75% of samples | CONFIRMED (hypothesis 1) |
| 2026-10-30 21:15 | T3 v2 vs v3 on 2000 AISHELL segs, 3000 rounds | v3 jitter last50=-0.4576 vs v2 -0.4672 (delta +0.0096); cap_active_frac=0.266; reward v3 1.5117 vs v2 1.5434; ood v3 -0.0175 vs v2 -0.0227; both guards no trip | PASS (narrow) — v3 jitter better but margin +0.0096 is small vs overall jitter degradation -0.17 during training |
