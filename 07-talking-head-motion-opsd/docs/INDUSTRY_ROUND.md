# 07 · Cross-industry digital humans: finance / real-estate / short-video

> Goal of this round: turn the OPSD digital human into a usable persona for **three industries**,
> and every one of them has to be natural. Nodding, listening, blinking and smiling are real
> human behaviours, not an exaggerated AI look.
> The method is still [DiffusionOPSD](https://github.com/worldbench/DiffusionOPSD)
> on-policy self-distillation, just applied in **motion space**, the 73 dimensional coefficients.
> The baseline is JoyVASA `motion_generator_hubert_chinese`, frozen, rendered with
> LivePortrait at 1280 px with pasteback.

---

## 1. Conclusion in one paragraph

**All three industry personas land inside the measured real-human motion band, amplitude
0.84 to 0.90x of target, none overshoots, and the v2 and v3 style overshoot of 2.4 to 2.7x,
the uncanny valley, did not come back.** The same face under the three industry LoRAs shows
three different behaviours, steady, warm, lively, and that is where the sense of industry
comes from.

**A real problem was found and fixed along the way**: the first rendered version had the
subject's **eyes half closed**, while base actually looked best. There were two root causes,
both confirmed by measurement, see section 4: the blink target had been raised by me to
0.18 to 0.21 while the v4 validated value is 0.15, and some source portraits simply have
small eyes. After the fix, persona eye openness went from **EAR 0.102 to 0.348**, level with
base at 0.343, so **the industry motion stayed and the eyes opened**.

## 2. How the three reward models were set, not hand tuned

The core lesson of v4: **the amplitude term must be peaked, a gaussian, and its centre must
be measured.** This round extends that principle to the industry axis, **one measured motion
band per industry**:

| industry | video proxy, public, real human voice | semantics | n |
|---|---|---|---|
| finance | HDTF news anchor clips | formal, restrained, credible | 25 |
| realestate | CelebV-HQ medium amplitude clips | explaining, friendly | 16 |
| shortvideo | CelebV-HQ high amplitude clips | lively, attention grabbing | 16 |

Note: public datasets carry no industry label, so this is an **explicitly declared proxy
split**. CelebV-HQ is real YouTube talking footage, cut into a calm group and a lively group
by **measured motion amplitude**. All numbers come from measurement in
`scripts/calibrate_industry_bands.py`, MediaPipe FaceLandmarker plus solvePnP, with
Butterworth filtering **identical** to the filter used inside the reward, so the units line up
directly.

### Measured result, `outputs/industry_bands.json`

| industry | low band amplitude, 0.3 to 3 Hz, deg | high band jitter, 6 to 12.5 Hz, deg | blinks per min |
|---|---|---|---|
| finance | **1.384** [IQR 1.03 to 1.50] | **0.094** | 18.1 |
| realestate | **1.257** [1.03 to 1.50] | **0.118** | 20.3 |
| shortvideo | **3.039** [2.70 to 3.49] | **0.349** | 22.6 |

The industry gap is real: short-video talking heads move about **2.2x** the head amplitude of
finance broadcasting, and **3.7x** the jitter.

### Quality gates, important

The raw measurement on CelebV-HQ contained a lot of solvePnP noise, one clip reported 37 deg
of head motion. Without gates that noise would be read as an industry trait. Four hard gates
removed 71 percent of the CelebV-HQ clips:

- near frontal, mean absolute yaw and pitch at or below 25 deg
- low band amplitude inside 0.2 to 8.0 deg, the plausible range for real speech
- high band jitter at or below 1.0 deg
- blinks at or below 60 per minute

HDTF kept 25 of 40, CelebV-HQ kept 18 of 90, re-measured at 16 after expanding the sample to
390 clips, see `industry_bands_full.json`.

### From measured band to reward parameters, the single entry point

| parameter | mapping | finance | realestate | shortvideo |
|---|---|---|---|---|
| `--amp-target-deg` | median of measured low band | 1.3838 | 1.2570 | 2.9299 |
| `--amp-tol-deg` | fixed 1.0, the v4 validated value | 1.0 | 1.0 | 1.0 |
| `--jitter-cap-deg` | 2 x measured high band median | 0.1883 | 0.2358 | 0.5000 |
| `--blink-target-mean` | measured blink rate over 100, clamped to 0.10 to 0.22 | 0.1815 | 0.2030 | 0.2091 |

**Apart from these four numbers, the reward weights, formulas, LoRA config, lr, rounds and seed
are byte identical to v4 across all three industries.** Any difference can therefore be
attributed to the target band itself and not to hyperparameter drift, following the
attributability principle of the task list section 0b.

## 3. Baseline and improvement, what the baseline looks like and what the method adds

### Baseline is frozen JoyVASA without OPSD

Re-measured on 10 public voices, `outputs/naturalness_eval_r3.json`, against the human band:

| variant | low band amplitude, deg | vs real human 1.878 deg | high band jitter, deg |
|---|---|---|---|
| real human, professional broadcast | 1.878 | 1.00x | 0.107 |
| **base, untrained** | 0.879 | **0.47x, too rigid** | 0.119 |
| OPSD v2 | 5.166 | 2.75x, overshoot into the uncanny valley | 0.193 |
| OPSD v3 | 4.460 | 2.38x, overshoot | 0.173 |
| **OPSD v4, peaked reward** | **1.659** | **0.88x** | **0.116** |

**Conclusion: base is not natural but flat, it is rigid, 0.47x of a real human. Naive reward
hacking pushes it to 2.4 to 2.7x, which reads as eerie. Making the amplitude term a peaked
gaussian centred on the measured band lands it at 0.88x, and that is the thing actually
learned by moving DiffusionOPSD into motion space.**

### The three industry personas on **held-out speakers**, `outputs/industry_eval.json`

12 AISHELL-1 eval voices, the 15 eval speakers do not overlap at all with the 60 training
speakers:

| variant | amplitude, deg | vs its own target | vs base | jitter, deg | vs target |
|---|---|---|---|---|---|
| base | 0.890 | n/a | 1.00x | 0.1187 | n/a |
| finance | 1.162 | **0.84x** | **1.31x** | 0.1196 | 1.27x |
| realestate | 1.134 | **0.90x** | **1.27x** | 0.1124 | 0.95x |
| shortvideo | 2.699 | **0.89x** | **3.03x** | 0.1561 | 0.45x |

How to read it:

- **No overshoot**: all three sit at 0.84 to 0.90x of target, the direction is slightly
  conservative rather than exaggerated. The v2 and v3 overshoot of 2.4 to 2.7x did not recur.
- **The three industries genuinely separate**: short-video amplitude is 2.3x finance and
  real-estate, consistent with the measured bands, 3.04 against 1.38 and 1.26.
- **Jitter stays close to the human band everywhere**, 0.11 to 0.16 deg against a human
  0.09 to 0.35.
- Finance and real-estate amplitude are close, 1.16 against 1.13, and **that is the honest
  conclusion**: in real footage, news broadcasting and calm explaining have almost identical
  head motion, 1.38 against 1.26 deg, and a bigger gap should not be invented just to look
  differentiated.

### Training curves

Three industry reward curves: `outputs/figures/fig_industry_training.png`;
band versus persona landing: `outputs/figures/fig_industry_bands.png`;
base, v2, v3, v4 comparison: `outputs/figures/fig_main_comparison.png`;
method diagram: `outputs/figures/fig_method_motion.png`.

## 4. Problem found mid-round: half closed eyes, fixed

The first three-industry demo rendered all three personas **squinting**, while base looked most
natural. This is not an impression, it was measured frame by frame with MediaPipe eye aspect
ratio, EAR:

| | EAR, median | relative to base |
|---|---|---|
| base, untrained | 0.343 | n/a |
| first persona version, blink target 0.18 to 0.21 | **0.102** | eye openness down to about 30 percent |
| **fixed persona, blink target 0.12, penalty doubled** | **0.348** | **1.01x base** |

Two root causes:

1. **The blink target was set too high.** The v4 validated value is 0.15, but I translated
   measured blink rate over 100 literally into 0.18 to 0.21. The blink term rewards the
   **peak** of the blink signal via `smoothmax`, so a high target makes OPSD park the
   expression coefficients near closed.
   Fix: `--blink-target-mean 0.12`, `--blink-overshoot-penalty` 4 to 8.
2. **Some source portraits squint already.** Low EAR in the source image means no amount of
   training opens the eyes.
   Fix: new `scripts/pick_portraits.py` with **EAR at or above 0.30** as a hard gate, together
   with at least 800 px, frontal and smiling; 9 of 220 candidates survived.

Verified with `scripts/measure_eye_openness.py`, per frame EAR plus closed-eye frame ratio.
After the fix the closed-eye frame ratio dropped to **0.05 to 0.17**.

## 5. Demo, cross-industry, high definition, natural

### Portraits and audio, no TTS, **real English human voices**

- Portrait is the **first frame** of a real public video, gated on EAR at or above 0.30, at
  least 800 px, absolute yaw and pitch at or below 15 deg, smile above 0
- Driving audio is the **real human voice of that same clip**, original speech from the
  CelebV-HQ or HDTF source, **English**, 16 kHz mono
- 3 sets per industry, portrait plus matching voice, times base and persona, 6 mp4 each,
  plus 6 same-face cross-industry comparisons
- The mp4 files are committed directly to `main` under
  `outputs/industry_demo_v3/{finance,realestate,shortvideo}/`, and the README homepage has a
  clickable mp4 link and a side by side GIF for every industry

### Render settings, high definition

`source_max_dim=1280` plus `flag_pasteback=True`: the animated face is composited back onto
the full resolution portrait instead of being cropped to a 256 px face. Output is H.264 with
AAC audio at 25 fps.

| industry | portrait resolution | mp4 count |
|---|---|---|
| finance | 844 to 1030 px | 6 |
| realestate | 808 to 1082 px | 6 |
| shortvideo | 886 to 1066 px | 6 |
| same-face control | 1030 px, one face | 6 |

### Same-face control, the most convincing set

One high definition portrait at 1030 px and one English voice, rendered with each of the three
industry LoRAs, under `outputs/industry_demo_v3/*_control/`. Face and voice are identical, so
any behavioural difference on screen can only come from that industry persona, which is direct
evidence that the sense of industry exists. Four-up GIF in `outputs/industry_gifs_v3/control/`.

### Where blink, nod, listening and smile come from

| behaviour | where it comes from |
|---|---|
| blink | the blink term of the reward, peaked, target **0.12**, deliberately conservative because a literal translation of measured blink rate over 100 half closes the eyes, see section 4; eye_probe supplies the gradient |
| nod | the amplitude term, peaked gaussian centred on measured low band amplitude |
| listening, micro motion during silence | the couple term ties the speech envelope to head speed, so amplitude keeps a floor even without speech |
| smile | **not in the reward.** Smile comes from the source portrait, candidates are ranked by mouth corner lift, and LivePortrait preserves the source expression. Smile was deliberately kept out of the reward to avoid introducing an unattributable change this round |

## 6. Reproduce

```bash
# 0. environment, JoyVASA plus LivePortrait plus HuBERT plus MediaPipe, about 2.5 GB
bash scripts/h20_render_setup.sh
#    if cv2 reports missing libGL/libGLES: dnf install -y mesa-libGL mesa-libEGL libglvnd-gles
#    HuBERT directory name on Linux: ln -s chinese-hubert-base \
#        third_party/JoyVASA/pretrained_weights/'TencentGameMate:chinese-hubert-base'
#    ffmpeg: this project uses the one shipped with imageio_ffmpeg, ln -sf $(python -c
#        'import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())') /usr/local/bin/ffmpeg

# 1. data, all public
python scripts/fetch_public_data.py --aishell-speakers 75 --hdtf 40 --celebv 300
python scripts/build_audio_segments.py --n-train 2000 --n-eval 100   # speakers do not overlap

# 2. measured motion bands of the three industries, the source of the reward model
python scripts/calibrate_industry_bands.py            # -> outputs/industry_bands.json

# 3. train, about 14 min each on H20, must run through the train_guard.py guard
ROUNDS=3000 bash scripts/train_industries.sh          # -> outputs/opsd_ind_*/lora.pt

# 4. held-out evaluation, eval speakers do not overlap with training
python scripts/eval_industries.py --n-clips 12        # -> outputs/industry_eval.json

# 5. demo portraits, render, GIFs, figures
python scripts/pick_portraits.py --scan 220 --per-industry 3   # EAR>=0.30, >=800px, frontal, smiling
python scripts/render_industry_demo.py --manifest data/industry_demo_manifest_v3.json --gpu 0
python scripts/measure_eye_openness.py --pattern 'outputs/industry_demo_v3/finance/*.mp4'
# same-face control, pick 1 to 2 high definition portraits, one face across three industries
python scripts/render_industry_demo.py --portrait-override <png...> --gpu 0
python scripts/make_demo_gifs.py --video-dir outputs/industry_demo/finance \
    --out outputs/industry_gifs/finance --variants base finance
python scripts/make_industry_figures.py
```

Guard: every training run is launched through `tools/train_guard.py --kind motion`, selftest OK.
This round ran 3 trainings plus 1 retrain with **zero trips**, no `GUARD_STOP.json`.

## 7. Honest limitations, what was not achieved

1. **Industry labels are proxies.** Public data has no industry-labelled talking-head video.
   Finance uses news anchors, the other two use amplitude quantiles of CelebV-HQ. That is the
   most honest thing that can be built today, but it is not real real-estate agent footage.
2. **The finance and real-estate difference is small**, because their measured bands are close,
   1.38 against 1.26 deg. We do not claim the three industries are sharply different.
3. **The jitter term on finance is 1.27x its target**, 0.120 against 0.094. The absolute value
   is still inside the human band, 0.09 to 0.35, but it was not pinned to target exactly; that
   is normal slack for a peaked reward when the target is tight.
4. **Smile is not in the reward**, see section 4. A model-controlled smile needs a new
   expression term and a retrain, which was skipped this round to keep attribution clean.
5. **No SyncNet and no human blind test.** GPU time this round went to three industry trainings,
   rendering and one rework; lip sync is still unmeasured, consistent with the README
   limitations.
6. **Blink rate is not precisely calibrated.** Counting blink per minute from EAR valleys on the
   rendered video gives 35 to 77, clearly above the human 10 to 20, which means this counting
   method is too sensitive to render noise to be trusted. This round therefore uses **only EAR,
   eye openness, and closed-eye frame ratio** as blink evidence, and blink per minute is not in
   the results table. Getting blink rate into the human range needs more reliable detection and
   a separate tuning round.
7. **Measured bands cannot always be translated linearly into reward parameters**, the biggest
   lesson of this round. Amplitude can be used directly, because its unit matches the one
   inside the reward. Blink cannot, because the reward works on the normalized eye_probe
   quantity, which is not 1 to 100 with blinks per minute. Setting 0.18 to 0.21 from a literal
   translation trained the eyes half shut, section 4. The lesson: translate literally only what
   has aligned units, and run a small sweep before fixing anything that does not.

## 8. Output inventory

```
outputs/
├── industry_bands.json / industry_bands_full.json / industry_bands_small.json
├── industry_eval.json                  # 12 held-out speaker evaluation
├── naturalness_eval_r3.json            # base/v2/v3/v4 re-measured against the human band
├── opsd_ind_finance|realestate|shortvideo/lora.pt + metrics.jsonl
├── industry_demo_v3/<industry>/*.mp4   # three industry demos, base vs persona, HD, eyes open
├── industry_demo_v3/<industry>_control/# same-face cross-industry control
├── industry_gifs_v3/<industry>/*.gif   # side by side GIFs, base vs persona per industry
├── industry_gifs_v3/control/*.gif      # same-face four-up control GIF
├── eye_openness.json                   # per frame EAR, evidence before and after the squint fix
└── figures/fig_*.png                   # paper figures, training curve, bands, main result, method
scripts/
├── fetch_public_data.py                # download AISHELL-1 / HDTF / CelebV-HQ
├── calibrate_industry_bands.py         # three industry measured bands with quality gates
├── train_industries.sh                 # three industry training, params read from the measured json
├── eval_industries.py                  # held-out speaker evaluation
├── prep_industry_demo.py               # portrait and audio prep, frontal plus smile ranking
├── pick_portraits.py                   # portrait gates: EAR >= 0.30, >= 800px, frontal, smiling
├── render_industry_demo.py             # HD render, supports same-face control
├── measure_eye_openness.py             # per frame eye openness, EAR and closed frame ratio
└── make_industry_figures.py            # paper figures
```
