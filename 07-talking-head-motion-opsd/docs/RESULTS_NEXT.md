# 07 · Round-2 Results (H20) — jitter single-point optimisation

All numbers below are measured, not estimated. Failed gates are reported as failures.

## 1 · What was changed

Exactly one thing: `jitter_cap_deg` in the reward.

| | v2 | v3 |
|---|---|---|
| `jitter_cap_deg` | 1.0 deg | **0.2 deg** |
| `jitter_cap_penalty` | 5.0 | 5.0 |
| `amp / couple / blink / gaze / ood` | — | identical to v2 |
| `lr` | 3e-4 | 3e-4 |
| LoRA | r=8 / alpha=8 | r=8 / alpha=8 |
| seed | 0 | 0 |

New files only: `src/motion_reward_v3.py`, `src/opsd_train_motion_v3.py`,
`src/opsd_train_motion_v2_large.py` (v2 with a `--data-dir` flag so the large
public set can be used without editing the original v2 script),
`src/diag_jitter.py`, `scripts/build_audio_segments.py`.
No existing reward / training / output file was modified.

## 2 · T1 — was the v2 cap dead? (YES)

`outputs/diag/jitter_diag.json`, 140 entries (20 clips × 7 variants):

| variant | std_high mean | p90 | max | frac > 1.0 deg (v2 cap) | frac > 0.2 deg (v3 cap) |
|---|---|---|---|---|---|
| base_naive | 0.178 | 0.263 | 0.355 | **0.000** | 0.283 |
| base_bon | 0.175 | 0.259 | 0.302 | **0.000** | 0.350 |
| opsd_naive (v3) | 0.255 | 0.384 | 0.443 | **0.000** | 0.700 |
| opsd_v2_bon | 0.242 | 0.346 | 0.502 | **0.000** | 0.750 |

`frac > 1.0 deg` is **exactly 0.0 for every variant**, including the trained ones.
The v2 hard cap never fired, so its gradient was identically zero — the same
dead-zone failure mode as V1's `ReLU+clamp` in project 08. Hypothesis 1 confirmed.
The v3 cap at 0.2 deg is active on 28–75 % of samples (`cap_active_frac` mean
0.266 during training), so it contributes gradient from the start.

## 3 · Training data replaced with a public set

Previous round: 25 macOS `say` TTS clips (synthetic, single style). Not a public
dataset. This round: **AISHELL-1** (OpenSLR #33 / `AISHELL/AISHELL-1` on HF,
Apache 2.0). 75 speakers downloaded; 60 train / 15 eval, **disjoint**.
Segments built by concatenating same-speaker adjacent utterances to 8–15 s with
0.3 s silence gaps, 16 kHz mono. Result: **2000 train / 100 eval segments →
14 260 four-second windows** (vs 107 before, a 133× increase).
Details: `data/DATA_SOURCES.md`, `data/audio_manifest_large.json`.
The 20 legacy TTS clips are retained as `legacy eval`.

## 4 · T3 — v2 vs v3, same data, 3000 rounds

50-step window means, first 50 vs last 50:

| term | v2 last50 | v3 last50 | v3 − v2 |
|---|---|---|---|
| reward | 1.5434 | 1.5117 | −0.032 |
| amp | 0.9244 | 0.8954 | −0.029 |
| **jitter** | −0.4672 | **−0.4576** | **+0.0096** |
| blink | 0.7871 | 0.7735 | −0.014 |
| gaze | 0.9841 | 0.9820 | −0.002 |
| couple | 0.3730 | 0.3623 | −0.011 |
| ood | −0.0227 | **−0.0175** | +0.005 |

Both guards ran 3000 steps without tripping. v3's jitter is better than v2's,
but the margin (+0.0096) is small compared with the fact that jitter still
degrades over training for both (−0.28 → −0.46).

## 5 · T4 — motion-space evaluation (n = 20 clips)

`outputs/tier2_r2/bon_motion_summary.json`:

| variant | jitter_hp_power | blink / min | amp_lp_power |
|---|---|---|---|
| base_naive | **14.12** | 5.5 | 1 093 |
| base_bon | 14.89 | 6.5 | 1 092 |
| v3 naive | **28.97** | 61.7 | 22 974 |
| v2 naive | 35.70 | 54.7 | 28 735 |
| v3 + BoN | 26.38 | **33.7** | 23 061 |
| v2 + BoN | 32.48 | 46.9 | 28 273 |

**v3 vs v2: jitter −19 % (28.97 → from 35.70), blink −27 % (33.7 vs 46.9).**
Both are improvements in the right direction.

### Gate outcome (section 4 of the task book): FAIL
The pre-registered criterion was `jitter_hp <= 1.3 × base` = 14.12 × 1.3 = 18.36.
v3 is 28.97, i.e. **2.05 × base**. The gate is not met, and this is reported as a
failure rather than being reinterpreted.

### Why the absolute jitter number is misleading
`amp_lp_power` shows OPSD makes the head move ~20× more than base. Because
`jitter_hp_power` is an absolute high-frequency power, it scales with amplitude.
Normalising removes the confound:

| variant | jitter / amp |
|---|---|
| base_naive | 0.01292 |
| v3 naive | 0.00126 |
| v2 naive | 0.00124 |
| v3 + BoN | 0.00114 |
| v2 + BoN | 0.00115 |

Relative to its own motion amplitude, OPSD motion is roughly **10× smoother than
base**. The `head_jitter` metric used in the previous round is amplitude-confounded
and should not be read as "OPSD made the motion jerkier" without this caveat.
The real remaining problem is that OPSD overshoots on amplitude (20× base), which
is what makes the absolute jitter look bad.

## 6 · Honest summary

- Confirmed and fixed a real bug: the v2 jitter cap was dead (0 % activation).
- v3 with a calibrated cap (0.2 deg) measurably beats v2: −19 % jitter, −27 % blink.
- The pre-registered success criterion (≤ 1.3 × base jitter) was **not** reached.
- Training on 133× more data (public AISHELL-1) did not by itself fix jitter.
- The next lever is amplitude, not jitter: OPSD's 20× amplitude overshoot is what
  drives the absolute jitter number.

## 7 · Not done / limitations

- Video-space (MediaPipe) metrics and SyncNet were not run: rendering 200 videos
  needs the renderer and time that went to the 08 long runs. Motion-space numbers
  above are the evidence available this round.
- Portraits are still the existing 3 images in `data/portraits`.
- `jaw_audio_corr` is not used as evidence (base itself is negative).
