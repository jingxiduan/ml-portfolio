# 07 · Talking-Head Motion OPSD · Project status

> Last updated 2026-11-01, cross-industry round
>
> **Latest round: three industry digital humans, finance / real-estate / short-video, see
> [`docs/INDUSTRY_ROUND.md`](docs/INDUSTRY_ROUND.md).**
> Every industry reward target band comes from measurement on real public video, not hand
> tuning; the personas land at 0.84 to 0.90x of target with no overshoot; the demos and the
> same-face cross-industry comparison are rendered.
> The next task list, jitter single point tuning plus more data, is in
> [`docs/NEXT_H20_TASKS.md`](docs/NEXT_H20_TASKS.md); T1 to T3 were finished on 10-30 as
> v3 and v4, and v4 is the common starting point of this three-industry round.

## Done

### 1. OPSD LoRA training, motion space
- **V1**: 600 steps, reward 0.35 to 1.16, win rate 100 percent against base
- **V2**: 600 steps, reward switched to a gaussian blink target plus a hard jitter cap, threshold
  1.0 deg; reading the training logs back, actual jitter is about 0.2 deg, so the cap was most
  likely never triggered, needs diagnosis
- LoRA checkpoints: `outputs/opsd/lora.pt`, `outputs/opsd_v2/lora.pt`
- Training metrics: `outputs/opsd/metrics.jsonl`, `outputs/opsd_v2/metrics.jsonl`, 600 steps each

### 2. Tier-2 · video space eval, H20 GPU, n=20 clips x 6 variants = 120 videos
- 6 configurations: base / opsd_v1_naive / opsd_v1_BoN / opsd_v1_BoN+thin / opsd_v2_naive / opsd_v2_BoN
- MediaPipe video space metrics, blink_rate, head_jitter, gaze_std and others
- Key results:

| metric | base | opsd v1 (naive) | + BoN | + BoN + thin | opsd v2 (naive) | opsd v2 + BoN | human target |
|---|---|---|---|---|---|---|---|
| blink_rate | 0.00 | 50.49 | 34.06 | **24.78** | 42.90 | **25.62** | 10-20 |
| head_jitter | 119.79 | 262.09 | 264.69 | 265.22 | 261.06 | **256.91** | ~120 |
| head_pitch_std | 1.27 | 3.68 | 3.67 | 3.66 | **3.35** | 3.35 | 1-4 |
| speech_head_corr | 0.13 | **0.27** | 0.27 | 0.27 | 0.25 | 0.26 | 0.2-0.4 |

- 20 three-way comparison GIFs: `outputs/tier2/tier2_compare_00..19.gif`
- Training curve figures: `fig_tier2_blink.png`, `fig_tier2_summary.png`, `fig_v1_vs_v2_curve.png`

### 3. Engine routing prototype
- 6 engine profiles: `outputs/profiles/engine_profiles.json`
- Contextual bandit routing: `outputs/routing/routing_results.json` plus `routing_curve.png`

### 4. Documents
- Full `README.md`, 271 lines
- `docs/H20_RENDER_PLAYBOOK.md`
- `docs/TIER2_NOTE.md`

## Output inventory

```
outputs/
├── opsd/lora.pt + metrics.jsonl          # V1 LoRA + 600 step metrics
├── opsd_v2/lora.pt + metrics.jsonl       # V2 LoRA + 600 step metrics
├── tier2/                                # 120 mp4 + 20 GIF + metrics + figures, 751 MB
├── eval/eval_results.json                # motion space eval
├── profiles/engine_profiles.json         # engine profiling
├── routing/routing_results.json + .png   # routing experiment
├── samples/                              # early samples, 11 MB
└── eye_probe/                            # eye debugging, 352 KB
```

## Not done yet, next up

### High priority
1. **Lower blink_rate**: v2+BoN is 25.62 against a human target of 10 to 20. Options:
   - strengthen blink thinning post-processing, raise the minimum spacing from 2.5 s to 3.5 s
   - or lower the blink weight and raise the jitter penalty in the V3 reward
2. **Lower head_jitter**: currently about 260 against a human 120. OPSD moves the head but shakes
   it. Options:
   - raise the jitter penalty weight, currently -0.5, to -1.5
   - or add temporal smoothing post-processing
3. **jaw_audio_corr is negative**: lip motion is out of sync with audio. The lip-sync loss needs
   to be revisited

### Medium priority
4. **More eval clips**: n=20 now, 50 or more is needed for statistical significance
5. **Human evaluation**: everything is automatic metrics today, A/B human scoring is missing
6. **FVD and LPIPS metrics**: standard video generation metrics are not computed

### Low priority
7. **Make routing online**: it is an offline simulation now, it could take real user preference
8. **More portrait and audio pairs**: only a limited set of faces and audio is used
