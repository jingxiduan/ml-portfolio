# Tier-2 · video-space metrics (held-out, MediaPipe, n=20 clips)

- N candidates per model: 8
- Track A (BoN): motion-space `naturalness_v2` argmax over N seeds
- Track B (thin): blink-thinning post-process (min 2.5 s between blinks, cosine attenuation on non-lip expression dims)
- Track C (opsd v2): LoRA retrained with reward v2 (gaussian blink target + hard jitter cap)
| metric | base | opsd v1 (naive) | + BoN | + BoN + thin | **opsd v2** (naive) | **opsd v2 + BoN** | human target |
|---|---|---|---|---|---|---|---|
| blink_rate | 0.00 | 50.49 | 34.06 | 24.78 | 42.90 | 25.62 | 10-20 |
| head_jitter | 119.79 | 262.09 | 264.69 | 265.22 | 261.06 | 256.91 | ~120 |
| head_pitch_std | 1.27 | 3.68 | 3.67 | 3.66 | 3.35 | 3.35 | 1-4 |
| head_yaw_std | 1.00 | 3.30 | 3.31 | 3.30 | 3.22 | 3.28 | 1-4 |
| head_speed_mean | 10.05 | 24.35 | 24.60 | 24.57 | 23.75 | 23.70 | 5-15 |
| speech_head_corr | 0.13 | 0.27 | 0.27 | 0.27 | 0.25 | 0.26 | 0.2-0.4 |
| still_ratio | 0.07 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | <0.1 |
| gaze_std | 0.07 | 0.23 | 0.23 | 0.20 | 0.23 | 0.22 | 0.1-0.3 |
| gaze_shift_rate | 1.83 | 2.75 | 2.52 | 2.52 | 2.53 | 2.45 | 1-3 |
| nod_rate | 1.08 | 1.16 | 1.14 | 1.15 | 1.17 | 1.21 | 0.3-1.0 |
| jaw_audio_corr | -0.14 | -0.09 | -0.08 | -0.09 | -0.10 | -0.10 | 0.3-0.5 |
