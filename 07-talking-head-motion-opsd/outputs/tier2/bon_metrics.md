# Tier-2 BoN + blink-thinning · video-space metrics (held-out, MediaPipe)

- clips: 1
- N candidates per model: 8
- Track A (BoN): motion-space `naturalness_v2` argmax over N seeds
- Track B (thin): blink-thinning post-process on BoN winner (min 2.5s between blinks, cosine attenuation on non-lip expression dims)

| metric | base (naive) | opsd (naive) | opsd + BoN | **opsd + BoN + thin** | human target |
|---|---|---|---|---|---|
| blink_rate | 0.00 | 37.50 | 31.25 | **18.75** | 10-20 |
| head_jitter | 148.36 | 235.40 | 244.53 | **246.30** | ~120 |
| head_pitch_std | 1.32 | 3.19 | 3.26 | **3.25** | 1-4 |
| head_yaw_std | 0.98 | 3.71 | 3.69 | **3.70** | 1-4 |
| head_speed_mean | 12.13 | 21.01 | 23.16 | **23.24** | 5-15 |
| speech_head_corr | -0.09 | 0.12 | 0.14 | **0.15** | 0.2-0.4 |
| still_ratio | 0.04 | 0.04 | 0.03 | **0.04** | <0.1 |
| gaze_std | 0.06 | 0.26 | 0.24 | **0.22** | 0.1-0.3 |
| gaze_shift_rate | 2.08 | 2.50 | 2.40 | **2.29** | 1-3 |
| nod_rate | 1.15 | 1.15 | 1.15 | **1.15** | 0.3-1.0 |
| jaw_audio_corr | -0.26 | -0.24 | -0.24 | **-0.23** | 0.3-0.5 |
