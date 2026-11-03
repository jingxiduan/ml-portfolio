# Gate Log — 07-talking-head-motion-opsd

| Time | Gate | Key Values | Result |
|------|------|------------|--------|
| 2026-10-30 18:40 | T2 v3 train 600 steps | reward=1.179, jitter=-0.427 (v2:-0.458), amp=0.781, cap_frac=0.12-0.33, guard=no trip | PASS |
| 2026-10-30 20:47 | T1 jitter diagnosis | cap 1.0 deg NEVER active: frac_above_cap=0.0 for ALL variants; base std_high p90=0.263, max=0.355; opsd p90=0.384; cap 0.2 deg activates on 28-75% of samples | CONFIRMED (hypothesis 1) |
| 2026-10-30 21:15 | T3 v2 vs v3 on 2000 AISHELL segs, 3000 rounds | v3 jitter last50=-0.4576 vs v2 -0.4672 (delta +0.0096); cap_active_frac=0.266; reward v3 1.5117 vs v2 1.5434; ood v3 -0.0175 vs v2 -0.0227; both guards no trip | PASS (narrow) — v3 jitter better but margin +0.0096 is small vs overall jitter degradation -0.17 during training |
| 2026-10-30 21:30 | T4 motion-space eval (n=20) | jitter_hp: base=14.12, v3=28.97, v2=35.70, v3+BoN=26.38, v2+BoN=32.48. blink/min: v3+BoN=33.7 vs v2+BoN=46.9. amp_lp: base=1093 vs OPSD~23000 | FAIL vs section-4 gate (needs <=1.3x base = 18.36; v3 is 2.05x). v3 beats v2 by 19% though |
| 2026-10-30 21:30 | T4 confound found | jitter/amp ratio: base=0.01292, v3=0.00126, v2=0.00124 -> OPSD motion is 20x larger in amplitude; absolute jitter grows with it but RELATIVE jitter is ~10x lower than base | NOTED — absolute head_jitter is amplitude-confounded |
| 2026-11-02 | V5 expression probe, 600 steps | expression reward first/final 100-step mean 0.7523/0.8108; active fraction 1.0; loss 0.01022/0.01024 | PASS optimization-signal gate only |
| 2026-11-03 | V5 artifact recovery | final seed-0 LoRA restored from `origin/backup/2026-11-02-full`; serialized args match the probe log | PASS artifact completeness |
| 2026-11-03 | R3 same-unit calibration | 40 sampled CelebV/HDTF clips, 77 windows; median target 0.3929, IQR tolerance 0.2071 | PASS for narrow confirmatory run; not a population estimate |
| 2026-11-03 | R3 calibrated seeds | seeds 0/1/2 completed 600 steps each, frozen target/tolerance, no guard trip | PASS optimization gate |
| 2026-11-03 | R3 held-out coefficients | n=40 paired clips; target-error improvement CIs exclude zero; pose high-frequency RMS 0.75–0.77× base | PASS coefficient and pose-HF gates |
| 2026-11-03 | R3 matched rendering | 40 held-out clips × base/seed0/seed1/seed2; strict four-second streams | COMPLETE; visible GIF and 160 MP4s |
| 2026-11-03 | R3 SyncNet non-inferiority | all ΔLSE-C lower CIs ≥ -0.070 and all ΔLSE-D upper CIs ≤ +0.231 under frozen ±0.5 margins | PASS relative preservation; absolute quality unresolved |
