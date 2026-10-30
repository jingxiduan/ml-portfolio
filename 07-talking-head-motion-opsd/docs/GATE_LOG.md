# Gate Log — 07-talking-head-motion-opsd

| Time | Gate | Key Values | Result |
|------|------|------------|--------|
| 2026-10-30 18:40 | T2 v3 train 600 steps | reward=1.179, jitter=-0.427 (v2:-0.458), amp=0.781, cap_frac=0.12-0.33, guard=no trip | PASS |
| 2026-10-30 20:47 | T1 jitter diagnosis | cap 1.0 deg NEVER active: frac_above_cap=0.0 for ALL variants; base std_high p90=0.263, max=0.355; opsd p90=0.384; cap 0.2 deg activates on 28-75% of samples | CONFIRMED (hypothesis 1) |
