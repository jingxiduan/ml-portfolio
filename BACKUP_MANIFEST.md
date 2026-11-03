# Training artifact backup · 2026-11-03

This branch stores reproducibility artifacts that are intentionally excluded from `main` because of size.

## 07 · Calibrated expression R3

- Three 600-step adapters (`seed0`, `seed1`, `seed2`).
- Exact `train.log`, `guard.log`, and `metrics.jsonl` per seed.
- Same-portrait/audio/generation-seed base and three-seed MP4 renders.
- Four-way GIF and render manifest.
- Main-branch result record: `07-talking-head-motion-opsd/docs/RESULTS_R3.md`.

Raw CelebV-HQ/HDTF calibration windows are not uploaded because dataset redistribution rights are separate from model-artifact reproducibility.

## 08 · V3 stability preflight

- 100-step metrics and guard log.
- LoRA checkpoints at steps 50 and 100.
- Main-branch result record: `08-video-opsd-i2v/docs/RESULTS_LONG.md`.

The 2,000-step V3 run was still active when this snapshot was created and is not represented as complete.
