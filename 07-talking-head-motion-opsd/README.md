# 07 · Talking-Head Motion OPSD

**Overview.** This work transfers DiffusionOPSD, an on-policy self-distillation algorithm
originally formulated for pixel diffusion, to audio-driven digital-human motion. The
optimisation algorithm is retained without modification; only the definition of a sample
changes, from an image latent to a trajectory of 73 dimensional motion coefficients that
drive a frozen LivePortrait renderer.

**Contributions.** First, the reward targets are estimated from real public human footage
rather than assigned manually, yielding one empirically measured target band per industry.
Second, real video is introduced as an auxiliary supervision channel, so that the policy can
additionally reproduce the motion statistics of real speakers through behaviour cloning.

**Research question.** Whether a single post-training method yields digital humans whose
motion statistics separate across industries, and whether that separation is a property of
the real footage or an artefact of the prescribed targets. The question is decomposed into
separability and ecological validity; the answer, with significance tests, is given in
`docs/SEPARATION_ANALYSIS.md` and summarised below.

**Technical report, paper format: [`docs/TECH_REPORT.md`](docs/TECH_REPORT.md)** ·
**Separability analysis: [`docs/SEPARATION_ANALYSIS.md`](docs/SEPARATION_ANALYSIS.md)** ·
**Handoff and reproduction: [`docs/HANDOFF_2026-11-02.md`](docs/HANDOFF_2026-11-02.md)**

---

## Primary result · policy trained on real human video

### V8 · behaviour cloning from real video combined with OPSD, 600 steps

First result obtained by learning motion directly from public video. **1753 four-second
windows** of real human motion coefficients were extracted from public footage. Half of the
optimisation steps minimise a behaviour-cloning objective against the observed coefficients,
and the remainder execute the OPSD reward loop. Portrait, driving audio and random seed are
held constant across the two conditions; only the checkpoint differs.

![bc_v1](outputs/demos/bc_v1_base_vs_trained.gif)

Left: `BASE`, the frozen pretrained model without post-training. Right: `TRAINED`, after
600 steps of imitation learning on real video combined with OPSD.
[1080-line mp4](outputs/demos/bc_v1_base_vs_trained_1080.mp4)

| metric | value |
|---|---|
| real-video training windows | **1753**, extracted from 684 public clips with the LivePortrait motion extractor |
| behaviour-cloning loss | decreases from 0.047 to a minimum of 0.024, ending at 0.093 at step 600 |
| coefficient difference, base versus trained | **0.540**; a value of zero would indicate that the LoRA was not applied |
| image statistics | mean luminance 108.4, 98th percentile 227, clipped **0.0 percent**, saturation 86, warmth +25 |

> This is a preliminary result obtained after only 600 steps. The behaviour-cloning loss
> increases again beyond step 420, indicating that the learning rate, the proportion of
> cloning steps and the volume of real data all require further tuning. The result
> nevertheless establishes the feasibility of the approach: motion can be learned from real
> human video.

The formulation post-trains a frozen audio-to-motion diffusion model, JoyVASA, with a
differentiable liveliness reward, so that the synthesised portrait blinks, nods and shifts
gaze rather than remaining static. Synthesis is performed by LivePortrait at up to 1280 px
with pasteback.

---

## Qualitative results · ordered by evidential weight

### V1 · three industry personas from the 3000-step round

Two distinct subjects are presented per industry. Left: `BASE`, the frozen pretrained model
without post-training. Right: `PERSONA`, the model post-trained with OPSD against that
industry's target band. Presenting two identities per industry allows within-industry
consistency and cross-subject generalisability to be assessed simultaneously.

| industry | GIF (v8) | mp4 |
|---|---|---|
| **finance** | ![finance](outputs/persona_grids/finance_two_people_v8.gif) | [mp4](outputs/persona_grids/finance_two_people_v8.mp4) |
| **real-estate** | ![realestate](outputs/persona_grids/realestate_two_people_v8.gif) | [mp4](outputs/persona_grids/realestate_two_people_v8.mp4) |
| **short-video** | ![shortvideo](outputs/persona_grids/shortvideo_two_people_v8.gif) | [mp4](outputs/persona_grids/shortvideo_two_people_v8.mp4) |

### V2 · reference condition driven by real human coefficients, **not** a model output

Left: `BASE`. Right: `REAL`, in which the motion coefficients extracted from a real
speaker's video are transferred onto the same portrait. This condition is natural by
construction, since the motion is a direct recording of human behaviour. It is therefore not
a model output but the reference target that the trained policy is optimised to approach.

| industry | GIF (v8) | mp4 |
|---|---|---|
| **finance** | ![finance](outputs/demos/finance_v8_base_vs_real.gif) | [mp4](outputs/demos/finance_v8_base_vs_real_1080.mp4) |
| **real-estate** | ![realestate](outputs/demos/realestate_v8_base_vs_real.gif) | [mp4](outputs/demos/realestate_v8_base_vs_real_1080.mp4) |
| **short-video** | ![shortvideo](outputs/demos/shortvideo_v8_base_vs_real.gif) | [mp4](outputs/demos/shortvideo_v8_base_vs_real_1080.mp4) |

### Interpretation of the perceptual comparison

| perceptual impression | condition | explanation | status |
|---|---|---|---|
| **most natural** | `REAL`, right panel of V2 | recorded human motion transferred directly, i.e. the reference trajectory itself | **not a model output**; constitutes the upper bound and the optimisation target |
| **intermediate, but clearly more animated than base** | `PERSONA`, right panel of V1 | learned behaviour: amplitude rises from 0.89 deg to **1.16 deg**, attaining **0.84x** of the measured human band without overshoot | **the model output of interest** |
| **least animated** | `BASE`, left panel of V1 and V2 | pretrained model without post-training | baseline condition |

Consequently, `REAL` is perceptually the most natural condition but is not generated by the
model; `PERSONA` is the generated output, and its objective is to approach `REAL`. V8 and V1
characterise the trained behaviour, whereas V2 defines the target.

## Asset inventory

Each artefact is listed below with its provenance, post-training status and disposition.

| id | content | count | post-trained | grading | location | disposition |
|---|---|---|---|---|---|---|
| **V1** | two subjects per industry, base and persona in a 2x2 grid | 3 GIF + 3 mp4 | **yes**, preceding round, 3000 steps | **v8** | **primary position** | **retain**, principal result |
| **V2** | base versus real-driven condition, recorded coefficients transferred | 3 GIF + 3 mp4 | no, inference only | **v8** | secondary position | **retain** as the reference condition; may be merged into the description of V1 |
| **V3** | one identity across three industries, control experiment | 1 GIF | yes | earlier, no v8 | cross-industry study section | retain, or relocate to the control study |
| **V4** | single-subject base versus persona, industry_gifs_v3 | 3 GIF | yes | earlier | **archived, reference only** | same rendering batch as V1 with the earlier grading; duplicate |
| **V5** | base versus v3, demo_gifs_frontal | 8 GIF | yes, v3 | earlier | **archived, reference only** | v3 overshoots the real band by 2.4x; superseded |
| **V6** | base versus v4, demo_gifs_v4 | 4 GIF | yes, v4 | earlier | **archived, reference only** | historical release |
| **V7** | source mp4 of every GIF above, industry_demo_v3 and others | 18+ mp4 | n/a | n/a | directory reference only | **retain**, master material for the GIFs; deletion would preclude re-rendering |
| **V8** | base versus the model **trained on real human video**, behaviour cloning plus OPSD, 600 steps | 1 GIF + 1 mp4 | **yes, this round** | v8 | **most recent position** | **retain**, most recent result |
| n/a | three figures, industry band, training curve, method diagram, plus the portrait sheet | 4 PNG | n/a | n/a | embedded in this document | retain, figures rather than video |

**Remarks.** V1 and V4 originate from the same rendering batch; V4 applies the earlier colour
grading and a single-subject layout. V1 and V2 address different questions: V1 is a model
output, whereas the right panel of V2 transfers recorded human motion and is therefore not a
model output. The greater apparent naturalness of V2 is consequently expected.



**Clip screening** (`scripts/screen_clips.py`); every criterion is measured automatically:
language verified by whisper transcription · frontal head pose, MediaPipe with solvePnP ·
speech rather than singing · adequate resolution and luminance.

**Image post-processing pipeline** (`scripts/postprocess_video.py`, then
`scripts/build_demo_sbs.py`): luminance stretch · CLAHE shadow recovery · gamma solved
against a measured target mean · **highlight rolloff above 200** · mild S-curve · modest
saturation lift · **smoothing restricted to skin regions, with eyes, teeth and hair
preserved** · unsharp masking · Lanczos resampling to a 1080-line master.

**Grading v8, 2026-11-01: reduced sharpening and a warmer tone; commencement of the training
stage.**

| parameter | v7 | **v8** | rationale |
|---|---|---|---|
| unsharp | amount 0.55, radius 1.4 | **amount 0.25, radius 1.8** | v7 exhibited over-sharpening, with visible halos around the eyes and teeth |
| clarity, micro-contrast | 0.45 | **0.30** | preserves tonal depth while removing the over-crisp edge |
| tone | neutral | **warm, +3 % R and -3 % B** | warmer and more favourable skin rendition at constant saturation |
| saturation | 1.05 | **1.05, unchanged** | deliberately held constant |

Measured on `finance_v8_base_vs_real_1080.mp4`: mean luminance 104.1, 98th percentile 235,
clipped **0.0 %**, warmth, R minus B, **+54.8**, sharpness proxy 10.0; v7 was
over-sharpened at comparable luminance.

### Training configuration · finance smoke run, 600 steps

In response to a request for an extended warmup and a reduced learning rate, both were
implemented in `src/opsd_train_motion_v5.py`, through `--lr` together with the additional
`--warmup` argument and cosine decay:

| setting | previous | **current** |
|---|---|---|
| lr | 3e-4 | **1e-4** |
| warmup | none | **200 steps linear (5e-7 -> 1e-4)**, then cosine to 0.1x |
| grad clip | 1.0 | 1.0 (unchanged) |

Checkpoint: `outputs/opsd_v5_finance_smoke/lora.pt`, obtained after 600 steps over 2000
AISHELL-1 windows under the finance target band, amplitude target 1.384 deg, jitter cap
0.188 deg and blink target 0.12.

| metric | step 1 | step 600 |
|---|---|---|
| reward | 1.647 | **2.057** (+25 %) |
| amplitude (deg) | 0.869 | **0.963** (moving toward the 1.384 deg measured band) |
| lr | 5e-7 | 1e-5 (end of cosine) |

No divergence and no guard intervention were observed. The remaining work is to render this
checkpoint against `base` on the finance clips so that the trained behaviour becomes
observable, and subsequently to extend the schedule to 3000 steps and to the two remaining
industries.

**Grading v7: global brightness versus local tonal structure.**
v6 corrected the exposure statistics but the output remained perceptually flat, because
increasing mean luminance and compressing highlights leaves *local* contrast unchanged, and
local contrast is the property responsible for the impression of translucency. v7 introduces
the operations that determine local tonal structure while simultaneously reducing saturation:

| operation | implementation | why it affects local tonal structure rather than brightness |
|---|---|---|
| air in the shadows | black point raised to 16 rather than crushed to 0 | shadow regions retain gradation |
| local contrast | CLAHE, clipLimit 1.6 over 8x8 tiles | the dominant contributor to perceived translucency |
| clarity, micro-contrast | `L += 0.45 * (L - blur(L, sigma 12))` | halo-free depth at a large radius, preserving detail without over-sharpening |
| highlights | compressed above 200 by a factor of 0.80 | highlight gradation is retained instead of clipping to flat white |
| colour | saturation **1.02 to 1.05** with vibrance 0.25, raising low-chroma pixels more | increased chroma without oversaturation |
| skin | smoothing 0.35, applied to skin pixels only | prevents local contrast from exaggerating skin pores |

Measured (`clarity` = mean |frame - blur(frame)|, higher = more local structure):

| clip | mean luma | p98 | clipped % | saturation | clarity v6 -> **v7** |
|---|---|---|---|---|---|
| finance | 111.5 | 238 | 0.0 | 105 | 9.6 -> **15.2** |
| real-estate | 118.1 | 226 | 0.0 | 122 | - -> **22.4** |
| short-video | 116.7 | 245 | 0.9 | 103 | - -> **51.8** |
| finance grid (persona) | 109.0 | 229 | 0.0 | 93 | 34.3 -> **37.7** |

All values are measured on the *rendered files* rather than on targets: no clipping occurs,
saturation decreases by 10 to 20 percent relative to v6, and local contrast increases by
40 to 60 percent.

**Three rendering artefacts: diagnosis, correction and measurement**

| artefact | cause | correction | measured |
|---|---|---|---|
| head size fluctuates | the `scale` coefficient, dimension 63, drifts with the driving video | median filter + ±2 % clamp on dim 63, median filter on translation dims 64–66 (`stabilize_scale_translation`) | face-width std/mean **2.04 % → 1.15 %** (finance), 1.08 % / 1.05 % (others) |
| tooth distortion and collapsed mouth region | reference frame selected with the mouth already open | portrait auto-pick scores eyes-open + **mouth-closed** + bright + crop-valid (`EAR·3 − jawOpen·6 + luma·0.5`) | picked-frame score 0.93 / 1.26 / 1.32 |
| flat, desaturated appearance | the S-curve alone darkens any value below mid-grey | gamma lift to a measured target mean before the S-curve | finance 92 → 110, real-estate 134 → 214 contrast range |

> ### Latest round: three industry personas, 2026-11-01
> A single method with three targets, **finance, real-estate and short-video**. The reward
> target of each persona is *estimated from the real public footage of that industry* rather
> than assigned manually. All three settle inside the measured human band
> (**0.84 to 0.90× target, no overshoot**), and the short-video persona exhibits
> **3.0× the head motion of base** while remaining within that band.
> Full write-up: **[`docs/INDUSTRY_ROUND.md`](docs/INDUSTRY_ROUND.md)** ·
> demos on branch `backup/industry-demo`.

---

## Summary of the experimental setup

| | |
|---|---|
| **Trainable parameters** | LoRA on JoyVASA's motion DiT (~179K params), audio → 73-d motion coefficients |
| **Base model** | JoyVASA `motion_generator_hubert_chinese` (frozen) |
| **Renderer** | LivePortrait (implicit-keypoint warping, not diffusion) · **not** trained |
| **Method** | DiffusionOPSD: on-policy query → reward-guided bounded targets → detached finite fit → EMA refresh |
| **Reward** | Differentiable liveliness: `amp + jitter + couple + blink + gaze + ood` |
| **Training data** | AISHELL-1 (Apache 2.0), 2000 segments / 60 speakers, disjoint 15-speaker eval |
| **Principal result** | v3 outperforms v2, with **jitter reduced by 19 %** and **blink rate by 27 %**, in motion space with n = 20 |
| **Known failure** | Absolute jitter is still 2.05× base; the pre-registered ≤1.3× gate **failed** |
| **Hardware** | 2× H20 (98 GB) this round; originally M3 Pro 36 GB |
| **Most recent extension** | **three industry personas** with target bands estimated from real footage; 0.84 to 0.90× target with **no overshoot**; 44 high-definition demonstration videos plus a same-face cross-industry control |

---

## Method · on-policy self-distillation in motion space

DiffusionOPSD builds supervision at *intermediate* denoising states instead of waiting
for the final sample. We keep the algorithm identical and change only what "sample" means:
a trajectory of 73-d motion coefficients rather than an image latent.

At query \(s=(c, z_q, \sigma_q)\) the frozen behaviour policy defines a clean anchor

$$y_0 = z_q - \sigma_q\, v_{\mathrm{old}}(z_q, c, \sigma_q)$$

Normalised reward gradient gives bounded positive / negative targets

$$y_0^{\pm} = y_0 \pm c_{\mathrm{adv}}\,\sigma_t\,\frac{\nabla_y R(y_0)}{\|\nabla_y R(y_0)\|_2}$$

and the trainable policy fits the **detached** targets with a contrastive objective

$$\mathcal{L}_{\mathrm{OPSD}} = \| \hat y_\theta - \bar y_0^{+} \|^2 - \| \hat y_\theta - \bar y_0^{-} \|^2$$

**Minimal pseudocode (motion space):**

```python
set_mode("behavior")
z      = rollout_to(motion_generator, audio_feat, t1, cfg_scale)   # on-policy trajectory
anchor = guided_x0(motion_generator, z, t1, audio_feat, cfg_scale) # clean anchor
r      = liveliness_reward(anchor, env)                            # differentiable
g      = autograd.grad(r.sum(), z);  d = g / g.norm()              # normalised ascent

tgt_pos = guided_x0(mg, z + step_size * d, ...)                    # bounded targets
tgt_neg = guided_x0(mg, z - step_size * d, ...)

set_mode("trainable")
pred = guided_x0(mg, z.detach(), ...)
loss = mse(pred, tgt_pos) + lambda_neg * relu(margin - mse(pred, tgt_neg))
loss.backward(); optimizer.step()
ema_update_behaviour_policy()
```

**Canonical settings** (mirroring `config/opsd_defaults.py` in DiffusionOPSD):

| Setting | DiffusionOPSD (SD3.5-M) | 07 (JoyVASA motion) |
|---|---|---|
| LoRA | PEFT | r=8, alpha=8 (~179K params) |
| `c_adv` (target radius) | 5 (ρ=0.10) | 3.0 (`reward-step-size`) |
| Optimizer | AdamW | AdamW, lr 3e-4 |
| Batch | 48 prompt groups | 8 windows |
| EMA refresh | every outer update | every 10 steps, decay 0.5 |
| Rollout | 10-step CFG-free | 50-step, `cfg_scale=2.0` |

---

## Baseline definition and the role of real footage

This distinction is a prerequisite for interpreting the results reported below.

**`base` = the same pretrained model with no OPSD post-training.** Following
DiffusionOPSD, the comparison is always *same model, with vs without the
on-policy self-distillation stage*. It is **not** real video.

| term | what it is | role |
|---|---|---|
| **base** | JoyVASA `motion_generator_hubert_chinese`, untouched | the baseline the method must beat |
| **OPSD (v2/v3)** | the same model + trained LoRA | the method |
| **real video** (HDTF / CelebV-HQ) | human talking footage | **reference / upper bound only** · shows where natural behaviour sits; not a baseline |

Real footage is used here for two things only: (1) extracting still portraits, and
(2) reading off what natural motion statistics look like, which is how the jitter
cap was calibrated (base p90 = 0.263°). It is never scored against the model.

## Cross-industry study · finance, real-estate, short-video

> Full write-up: **[`docs/INDUSTRY_ROUND.md`](docs/INDUSTRY_ROUND.md)**.
> This is the current round; the generic presets further down are the earlier,
> hand-tuned version that this round replaces with **measured** targets.

### Per-industry reward targets estimated from real footage

A result carried over from v4 is that the amplitude term must be *peaked* and centred on an
empirically **measured** human band. This study applies that principle per industry. Each
band is estimated from real public footage using the *same band-pass filter as the reward
itself*, so the resulting values are used directly as `--amp-target-deg`:

| industry | video proxy (public, real speech) | amplitude (0.3–3 Hz) | jitter (6–12.5 Hz) | blink /min | n |
|---|---|---|---|---|---|
| finance | HDTF news anchors | **1.384°** | **0.094°** | 18.1 | 25 |
| real-estate | CelebV-HQ mid-amplitude clips | **1.257°** | **0.118°** | 20.3 | 16 |
| short-video | CelebV-HQ high-amplitude clips | **3.039°** | **0.349°** | 22.6 | 16 |

Short-video head motion is **2.2× that of finance**; the industry difference is measured
rather than assumed. A quality gate, comprising near-frontal pose, amplitude of at most
8 deg, jitter of at most 1 deg and a blink rate of at most 60 per minute, removes solvePnP
estimation noise; without it, CelebV-HQ yielded values as large as 37 deg of head motion,
and the band would have been defined by measurement failure.

![measured industry bands vs persona](outputs/figures/fig_industry_bands.png)

### Evaluation on held-out speakers, 40 clips, disjoint from the training speakers

| variant | amplitude | vs its target | vs base | jitter | vs its jitter target |
|---|---|---|---|---|---|
| base, untrained | 0.924° | · | 1.00× | 0.121° | · |
| **finance** | 1.213° | **0.88×** | **1.31×** | 0.120° | 1.27× |
| **real-estate** | 1.114° | **0.89×** | **1.21×** | 0.105° | 0.89× |
| **short-video** | 2.597° | **0.85×** | **2.81×** | 0.151° | **0.43×** |

**No variant overshoots**: all three settle at 0.85 to 0.89× their respective amplitude
targets, and the 2.4 to 2.7× overshoot responsible for the uncanny-valley appearance of v2
and v3 does not recur. Jitter is delivered less faithfully; short-video reaches only 0.43× of
its jitter target, which is why jitter cannot serve as a second behavioural axis.

### Is the industry difference provable and separable?

This question is answered separately for the reference footage and for the trained policies,
with effect sizes, confidence intervals and permutation nulls, in
**[`docs/SEPARATION_ANALYSIS.md`](docs/SEPARATION_ANALYSIS.md)**. The verdict, in short:

| question | answer | evidence |
|---|---|---|
| Are the three personas distinguishable? | **yes** | four-way leave-one-out identification accuracy **0.831** against chance 0.25, permutation p = 2e-4; prescribed ordering reproduced in **33 of 40** clips, binomial p = 1.1e-19 |
| Does that prove the industries differ? | **no** | the difference is prescribed by the measured bands; OPSD's contribution is realising a prescribed band without overshoot |
| Is the prescribed difference real in the footage? | **only for short-video versus the formal style** | finance versus real-estate in real footage: AUC **0.632**, p = **0.16**, not separable |
| Is the difference a style or a scalar? | **a scalar** | adding jitter as a second feature reduces identification accuracy from 0.831 to **0.600** |

The model is more self-consistent than the humans it was calibrated on, coefficient of
variation 0.06 to 0.07 against 0.15 to 0.23, which mechanically inflates every separability
statistic. Model-side separability therefore cannot be cited as evidence for a real industry
difference: for finance versus real-estate the policies separate at AUC 0.816 while the
reference footage does not separate at all.

Only four scalar arguments differ across the three runs, namely `--amp-target-deg`,
`--amp-tol-deg`, `--jitter-cap-deg` and `--blink-target-mean`. Reward weights, formula, LoRA config, lr,
rounds and seed are byte-identical to v4, so the difference is attributable to the
target band alone.

![training curves](outputs/figures/fig_industry_training.png)

### Eye-openness regression: diagnosis and correction

The first cross-industry rendering exhibited **partially closed eyes**, already visible in
the initial figure, and the untrained base model appeared more natural than the trained
personas. Two causes were identified, both by measurement:

1. **Excessive blink target.** v4 was configured with `blink_target = 0.15`; the three
   industries had been raised to 0.18 to 0.21 through a naive conversion of the measured
   blinks per minute divided by 100. Since the blink term rewards the *peak* of the blink
   signal, a higher target drives the expression coefficients towards the closed state.
2. **Low eye openness in the source portraits.** Where the reference frame exhibits a low eye
   aspect ratio, no amount of training yields an open-eyed rendering.

Correction: `blink_target = 0.12`, `blink_overshoot_penalty` increased from 4 to 8, and
portraits are additionally gated on **measured eye openness**, `EAR ≥ 0.30`, together with a
minimum dimension of 800 px, frontal pose and a smiling expression.

Eye aspect ratio, median across frames, measured on the rendered videos
(`scripts/measure_eye_openness.py`):

| | EAR | vs base |
|---|---|---|
| base (untrained) | **0.343** | · |
| initial persona, blink 0.18 | **0.102** | eye openness reduced to approximately 30 % |
| **revised persona, blink 0.12** | **0.348** | **1.01× base**, eyes open |

The fraction of frames with closed eyes decreases to **0.05 to 0.17**. The personas retain
their industry-specific motion while matching the untrained model in eye openness.

### Qualitative comparison across industries, high definition, real English speech

Portraits are the first frames of real public footage, gated on measured eye openness, a
minimum dimension of 800 px, frontal pose and a smiling expression. **The driving audio is
the real human speech of each source clip, in English**, taken from CelebV-HQ and HDTF
footage; no synthetic speech is used. Rendering employs `source_max_dim=1280` with
pasteback.

**Same-face cross-industry control, V3.** Portrait and driving audio are identical across
conditions and only the persona checkpoint varies, so any behavioural difference is
attributable to the persona alone:

![same face, three industries](outputs/industry_gifs_v3/control/same_face_p00_-R5aAhsrS9w_3_0__base_vs_three.gif)

`base | finance | real-estate | short-video`

**Archived: single-subject base versus persona comparison**

> **This single-subject set has been superseded by the two-subject grid presented above**; both
> originate from the same rendering batch, this one with the earlier grading.
> Subject p00 of the finance condition was **excluded**: the portrait does not fixate the
> camera, with a gaze offset of minus 0.073 eye widths, which is already visible in the
> source frame.

### Screening criteria and their measurements

**All candidate clips were screened by automated measurement**
(`scripts/check_persona_clips.py` and `scripts/add_gaze_checks.py`; complete values are
recorded in `outputs/persona_clip_checks.json`):

| check | how | result for the 6 selected clips |
|---|---|---|
| speaking, not singing | voiced-run p90 ≤ 0.3 s, harmonic ratio ≤ 0.44 (singing holds long steady voiced runs) | **all pass** |
| English | whisper-tiny transcription of the audio track | **all English** |
| voice and face correspondence | portrait and audio are taken from the **same source clip**, verified in `outputs/industry_demo_v3/manifest.json` as `pXX_<clip>.png` matched to `pXX_<clip>.wav`, so the voice is that of the depicted subject; a fundamental-frequency cross-check is also reported | **consistent** |
| camera fixation | gaze offset between iris and eye corner, combined with head pose by MediaPipe | selected clips within 0.06 eye widths and absolute yaw at most 9°; **p00 excluded**, gaze offset minus 0.073 in an off-camera reference frame |

Selected subjects per industry: finance, p06 and p03; real-estate, p01 and p04;
short-video, p08 and p05. All remaining clips are retained on disk.

> These single-subject pairs constitute an **earlier release** from the same rendering batch
> with the previous grading, superseded by the two-subject grid above:
> [`outputs/industry_gifs_v3/`](outputs/industry_gifs_v3/) · [`outputs/industry_demo_v3/`](outputs/industry_demo_v3/)

More pairs per industry: `outputs/industry_demo_v3/{finance,realestate,shortvideo}/`
(6 mp4 each) and `outputs/industry_gifs_v3/`.

Provenance of the observed behaviours:

| behaviour | source |
|---|---|
| blink | reward blink term, target = that industry's measured blink rate |
| nod / head motion | amplitude term (peaked Gaussian at the measured band) |
| listening (micro-motion while silent) | `couple` term ties speech envelope to head speed; amplitude keeps a floor |
| smile | **absent from the reward**; inherited from the source portrait, which is ranked by mouth-corner elevation. A smile term was deliberately omitted in this round in order to preserve attributability |

![method](outputs/figures/fig_method_motion.png)

### Reproduction of the cross-industry study

```bash
python scripts/fetch_public_data.py --aishell-speakers 75 --hdtf 40 --celebv 300
python scripts/build_audio_segments.py --n-train 2000 --n-eval 100
python scripts/calibrate_industry_bands.py     # -> the three reward targets
ROUNDS=3000 bash scripts/train_industries.sh   # -> outputs/opsd_ind_*/lora.pt
python scripts/eval_industries.py --n-clips 12
python scripts/prep_industry_demo.py --per-industry 8
python scripts/render_industry_demo.py --gpu 0
python scripts/make_industry_figures.py
```

All four training runs were launched under `tools/train_guard.py`; no guard intervention was recorded.

---

## Persona presets · earlier, manually tuned version

> **Status: superseded.** These weights were tuned manually and used with the v3 trainer,
> whose monotone `tanh` amplitude term overshoots to 2.4 to 2.7× the real band. The study
> above replaces them with targets estimated from real footage. They are retained to preserve
> attribution history.

The reward constitutes the control surface. Three personas are trained from a **single**
reward function, differing **only** in the term weights and in the OPSD step size `c_adv`:

| knob | finance | real estate | entertainment hook |
|---|---|---|---|
| `amp` (motion magnitude reward) | **0.35** | 0.6 | **1.4** |
| `jitter` (anti-jitter pressure) | **1.2** | 0.8 | 0.4 |
| `gaze` (eye wandering) | 0.2 | 0.3 | 0.45 |
| `couple` (speech-rhythm coupling) | 0.6 | 0.5 | 0.7 |
| `blink` target | 0.10 | 0.13 | 0.18 |
| `c_adv` (OPSD target radius) | **1.5** | 2.2 | **3.5** |
| `jitter_cap_deg` | 0.15 | 0.20 | 0.35 |
| intended style | formal, restrained, credible | warm, credible | animated, attention-seeking |

**Relation to the uncanny-valley effect.** The uncanny appearance is not attributable to
jitter: once amplitude is normalised out, OPSD motion is approximately 10× *smoother* than
that of base. It arises from **amplitude overshoot**, since `amp` employs `tanh(std_low/3)`,
which OPSD drives to saturation and thereby produces approximately 20× the base amplitude.
Reducing the `amp` weight, together with `c_adv`, is therefore the appropriate control for a
restrained persona, which is why finance is assigned 0.35 and entertainment 1.4.

```bash
ROUNDS=3000 GPU=0 bash scripts/train_styles.sh
# -> outputs/opsd_style_finance|realestate|hook/lora.pt
```

## Reward calibration · v4 amplitude peaked at the measured human band

**This constitutes the principal result of the round.** All values below are measured in the
reward's own units against the band estimated from real professional speech
(`scripts/eval_naturalness.py`, `outputs/naturalness_eval.json`):

| variant | low-band amplitude | vs real | high-band jitter | vs real |
|---|---|---|---|---|
| **real professional speech** | **1.878°** | 1.00× | **0.107°** | 1.00× |
| base, untrained | 1.042° | 0.55×, insufficiently animated | 0.137° | 1.28× |
| v2 | 5.125° | **2.73×, overshoot** | 0.215° | 2.01× |
| v3 | 4.524° | **2.41×, overshoot** | 0.188° | 1.76× |
| **v4 (calibrated)** | **1.651°** | **0.88×** | **0.124°** | **1.16×** |

v2 and v3 overshoot real speech by 2.4 to 2.7×, and that overshoot *is* the uncanny valley.
v4 falls within 12 % of the human band in amplitude and within 16 % in jitter, while
attaining a **higher** reward, R = 1.776 against 1.512 for v3: the reward increases *because*
the motion becomes more human, not through inflation of its magnitude.

The only change is the amplitude term:

```python
# v3, MONOTONE: the reward increases with magnitude without bound, so OPSD amplifies indefinitely
amp = tanh(std_low / 3.0)

# v4, PEAKED at the measured human band; both under- and overshoot are penalised
amp = exp(-0.5 * ((std_low - 1.878) / 1.0) ** 2)
```

| amplitude | v3 reward | v4 reward |
|---|---|---|
| 1.273° (base) | 0.40 | 0.83 |
| **1.878° (real)** | 0.55 | **1.00** |
| 4.446° (v3 result) | 0.90 | 0.04 |
| 6.0° | 0.96 | 0.00 |

```bash
python scripts/calibrate_reward_targets.py   # measure the band from real video
CUDA_VISIBLE_DEVICES=1 python src/opsd_train_motion_v4.py \
    --out outputs/opsd_v4 --data-dir data/audio_public/train \
    --rounds 3000 --amp-target-deg 1.878 --amp-tol-deg 1.0
python scripts/eval_naturalness.py           # evaluate each variant against the band
```

## Results

### Motion-space evaluation (n = 20 clips)

`outputs/tier2_r2/bon_motion_summary.json`. `jitter_hp_power` and `amp_lp_power` are
high-/low-frequency pose power; `blink` is blinks per minute (human range 10–20).

| Variant | jitter_hp_power | blink / min | amp_lp_power |
|---|---|---|---|
| base naive | **14.12** | 5.5 | 1 093 |
| base + BoN | 14.89 | 6.5 | 1 092 |
| v2 naive (cap 1.0°) | 35.70 | 54.7 | 28 735 |
| **v3 naive (cap 0.2°)** | **28.97** | 61.7 | 22 974 |
| v2 + BoN | 32.48 | 46.9 | 28 273 |
| **v3 + BoN** | **26.38** | **33.7** | 23 061 |

**v3 versus v2 under identical data and optimisation budget: jitter reduced by 19 % and blink rate by 27 %.**

### Training reward, 50-step window means

| Run | Data | Rounds | reward first→last | jitter first→last |
|---|---|---|---|---|
| v1 | 25 TTS clips | 600 | +0.334 → +1.263 | −0.309 → −0.456 |
| v2 | 25 TTS clips | 600 | +0.333 → +1.247 | −0.310 → −0.440 |
| v3 | 25 TTS clips | 600 | +0.315 → +1.116 | −0.329 → −0.432 |
| v2 | **2000 AISHELL segs** | 3000 | +0.337 → +1.543 | −0.282 → −0.467 |
| **v3** | **2000 AISHELL segs** | 3000 | +0.330 → **+1.512** | −0.290 → **−0.458** |

### Pre-registered criterion not satisfied

Pre-registered criterion, task book section 4: `jitter_hp ≤ 1.3 × base`, that is
14.12 × 1.3 = **18.36**. v3 attains **28.97 = 2.05 × base**. The criterion is therefore
**not satisfied**, and the outcome is reported as a negative result.

### Amplitude normalisation of the jitter statistic

OPSD increases motion amplitude by approximately 20×. Since `jitter_hp_power` is an
*absolute* high-frequency power, it scales with amplitude; normalising by amplitude removes
this confound:

| Variant | jitter / amp |
|---|---|
| base naive | 0.01292 |
| v3 naive | 0.00126 |
| v2 naive | 0.00124 |
| v3 + BoN | 0.00114 |

Relative to its own amplitude, OPSD motion is approximately **10× smoother than base**. The
dominant remaining deficiency is amplitude overshoot, approximately 20× base, which accounts
for the unfavourable absolute jitter values; amplitude, rather than the jitter term itself, is
therefore the appropriate next control.

---

## Implementation note · LoRA mode dispatch during inference

`DualLoRALinear.forward` branches on a mode string:

| mode | parameters used |
|---|---|
| `"trainable"` | `A_t / B_t`, the trained parameters |
| `"behavior"` | `A_b / B_b`, the EMA copy, with `B_b` initialised to zero |
| any other value | `base(x)` only, **no LoRA contribution** |

Inference must therefore load `A_t/B_t` **and** invoke `set_mode("trainable")`. An earlier
revision of `scripts/render_hd_compare.py` loaded `A_t/B_t` but subsequently invoked
`set_mode("behavior")`, so the sampled trajectory was generated with `A_b/B_b ≈ 0`.

Measured on one clip (mean |Δ| of motion coefficients vs base):

| mode used | diff vs base |
|---|---|
| `"behavior"`, incorrect | **0.000000**; v3 rendered identically to base |
| `"trainable"`, corrected | **0.795556** |

After the correction, base and v3 differ by **20.5 / 255** mean per-frame pixel difference
over 91 frames at 1024×1024. All results reported here were produced by the corrected code.
`B_t` norm in the trained checkpoint is 6.65, confirming the weights are non-zero.

---

## Diagnostic · inactive jitter constraint

v2 introduced a hard jitter cap at 1.0° that was never activated. Measured over 140 entries,
20 clips by 7 variants, in `outputs/diag/jitter_diag.json`:

| Variant | std_high mean | p90 | max | frac > 1.0° (v2 cap) | frac > 0.2° (v3 cap) |
|---|---|---|---|---|---|
| base naive | 0.178 | 0.263 | 0.355 | **0.000** | 0.283 |
| base + BoN | 0.175 | 0.259 | 0.302 | **0.000** | 0.350 |
| v3 naive | 0.255 | 0.384 | 0.443 | **0.000** | 0.700 |
| v2 + BoN | 0.242 | 0.346 | 0.502 | **0.000** | 0.750 |

`frac > 1.0°` is **exactly 0.000 for every variant**, so the gradient of the cap was
identically zero. This reproduces the dead-zone failure previously observed with the
`ReLU+clamp` formulation of V1 in project 08. v3 sets the cap to the base 90th percentile of
0.2°, which is active on 28 to 75 % of samples, with a mean `cap_active_frac` of 0.266.

**Only `jitter_cap_deg` was modified**, from 1.0 to 0.2. The weights and formulations of
`amp / couple / blink / gaze / ood` are identical to those of v2, so the observed difference
is attributable to that single change.

---

## Data

The preceding round employed 25 clips synthesised with the macOS `say` text-to-speech
engine; these were synthetic, stylistically uniform and **not drawn from a public dataset**.
They were replaced in this round as follows:

| Field | Value |
|---|---|
| Dataset | **AISHELL-1** (OpenSLR #33 / `AISHELL/AISHELL-1` on HF) |
| License | Apache 2.0 |
| Downloaded | 75 speaker archives + transcript |
| Split | 60 train / 15 eval speakers, **disjoint** (asserted) |
| Segments | 2000 train, 100 eval, 8–15 s, 16 kHz mono PCM_16 |
| Windows | 14 260 four-second windows (was 107 → **133×**) |
| Construction | adjacent same-speaker utterances concatenated, 0.3 s silence, never cross-speaker |

Further details are recorded in `data/DATA_SOURCES.md`, `data/audio_manifest_large.json`
and `scripts/build_audio_segments.py`. The 20 legacy text-to-speech clips are retained as a
`legacy eval` set. Audio is **not** committed to `main`; it is preserved on a `backup/*`
branch.

---

## Reproduction

```bash
# 1. environment (downloads JoyVASA + LivePortrait + HuBERT-chinese, ~2.5 GB)
bash scripts/h20_render_setup.sh

# 2. build the AISHELL-1 training set (CPU)
python scripts/build_audio_segments.py --n-train 2000 --n-eval 100

# 3. diagnose the jitter cap (CPU)
python src/diag_jitter.py                      # -> outputs/diag/jitter_diag.json

# 4. train v2 and v3 on the same data (each ~25 min on one H20)
CUDA_VISIBLE_DEVICES=0 python src/opsd_train_motion_v3.py \
    --out outputs/opsd_v3_large --data-dir data/audio_public/train --rounds 3000
CUDA_VISIBLE_DEVICES=1 python src/opsd_train_motion_v2_large.py \
    --out outputs/opsd_v2_large --data-dir data/audio_public/train --rounds 3000

# 5. motion-space evaluation (n=20)
python src/tier2_bon.py --motion-only \
    --lora outputs/opsd_v3_large/lora.pt \
    --lora-v2 outputs/opsd_v2_large/lora.pt --out outputs/tier2_r2

# 6. HD qualitative gallery (1280 px + pasteback)
python scripts/render_hd_compare.py --n-clips 4 --variants base v2 v3
```

All training runs must be launched under the guard; `python ../tools/train_guard.py
--selftest` must first print `selftest OK`.

---

## Qualitative gallery

Portraits and speech are taken from real footage. Source portraits are drawn from
**CelebA-HQ**, studio-quality stills, and **CelebV-HQ**, the first frames of real
talking-head videos whose audio tracks drive the animation, with no synthetic speech.
Rendering employs `source_max_dim=1280` and `flag_pasteback=True`, so that the animated face
is composited into the full-resolution portrait. **Verified output: 1024×1024, H.264 High
profile, 25 fps, with the audio track muxed**; earlier releases produced 256 or 512 pixel
face crops only.

**Portrait pool.** Candidates from CelebA-HQ and CelebV-HQ are filtered to **near-frontal
poses**, absolute yaw at most 15° and absolute pitch at most 15°, by MediaPipe with
solvePnP: **42 of 72 retained**. Profile views were excluded because a non-frontal face
occludes precisely the behaviour under measurement.

![portrait sheet](outputs/demo_gifs_frontal/portrait_sheet.png)

> **Archived**: side-by-side GIFs of the earlier v3 and v4 releases with the previous grading;
> referenced here only and no longer displayed on this page:
> [`outputs/demo_gifs_frontal/`](outputs/demo_gifs_frontal/), base versus v3, and
> [`outputs/demo_gifs_v4/`](outputs/demo_gifs_v4/), base versus v4 with the calibrated reward.
> v4 is the restrained variant: v2 and v3 amplify head motion to 2.4x to 2.7x of the real
> band, which is the origin of the uncanny-valley appearance, whereas v4 holds it at 0.88x.

Application of the LoRA is verified quantitatively: the mean per-frame pixel difference
between base and v3 is **20.5 / 255** over 91 frames at 1024×1024. See the implementation
note on LoRA mode dispatch; an earlier revision of this gallery rendered v3 identically to
base.

Full set: `outputs/hd_frontal/` (16 videos), `outputs/demo_gifs_frontal/` (8 GIFs),
`outputs/hd_compare_public/`, `outputs/hd_compare/`. `manifest.json` in each directory
records every (clip, portrait, variant) mapping.

```bash
python scripts/prep_public_portraits.py     # extract portraits + real audio
python scripts/filter_frontal.py            # keep near-frontal faces only
python scripts/render_hd_compare.py --clips <wav...> --portraits <png...> --variants base v3
python scripts/make_demo_gifs.py --video-dir outputs/hd_frontal
# ~8 s per 8 s clip on one H20
```

Observed behaviour and remaining deficiencies:

| Behaviour | Status |
|---|---|
| Blink rate | v3 with best-of-n selection reaches 33.7 per minute, an improvement over 46.9 for v2, but still above the human range of 10 to 20 |
| Head motion | amplitude increases approximately 20× relative to base; animated, but overshooting |
| Gaze shift | the `gaze` term rises from 0.51 to 0.98 over the course of training |
| Lip synchronisation | `jaw_audio_corr` is **not** used as evidence, since base itself yields a negative value of minus 0.14 |
| Skin and texture | inherited from the warping of the source portrait by LivePortrait; not optimised in this work |

---

## Repository layout

```
07-talking-head-motion-opsd/
├── src/
│   ├── motion_reward.py          # v1 reward (band metrics, smoothmax blink)
│   ├── motion_reward_v2.py       # + blink overshoot penalty, jitter hard cap 1.0°
│   ├── motion_reward_v3.py       # ONLY change: jitter cap 0.2°   <-- this round
│   ├── opsd_train_motion.py      # OPSD loop + add_lora / build_windows / guided_x0
│   ├── opsd_train_motion_v2.py   # v2 trainer
│   ├── opsd_train_motion_v3.py   # v3 trainer (+ --data-dir, cap_active_frac)
│   ├── opsd_train_motion_v2_large.py  # v2 + --data-dir (new file, v2 untouched)
│   ├── diag_jitter.py            # T1 cap-activation diagnostic
│   ├── engine.py                 # JoyVASA + LivePortrait wrapper
│   ├── tier2_bon.py              # best-of-n selection + motion/video metrics
│   └── video_metrics.py          # MediaPipe: blink, head_jitter, gaze
├── scripts/
│   ├── build_audio_segments.py   # AISHELL-1 -> 8-15 s segments
│   ├── render_hd_compare.py      # HD gallery (1280 px + pasteback)
│   └── h20_render_setup.sh       # one-shot env setup
├── data/
│   ├── DATA_SOURCES.md           # provenance, license, split
│   ├── audio_manifest_large.json
│   └── audio_public/             # generated wavs (not in main)
├── outputs/
│   ├── opsd_v2_large/ opsd_v3_large/   # 3000-round runs + metrics.jsonl
│   ├── diag/jitter_diag.json
│   ├── tier2_r2/bon_motion_summary.json
│   └── hd_compare/               # HD gallery
└── docs/
    ├── RESULTS_NEXT.md           # full numbers, including failures
    └── GATE_LOG.md               # per-gate pass/fail log
```

---

## Limitations

- **Three portraits only** (`host_f`, `host_f2`, `host_m`, each 1024×1024). No claim of
  identity diversity is made.
- **No SyncNet / LSE-C this round.** `jaw_audio_corr` is deliberately not used as a
  lip-sync proxy because base is already negative. Lip-sync remains unmeasured.
- **Amplitude overshoot**, approximately 20× base, remains the dominant artefact.
- Motion-space metrics are reported; video-space MediaPipe metrics on the AISHELL-1
  evaluation set were not computed in this round, since the available GPU time was allocated
  to the extended runs of project 08.

---

## Acknowledgements

This work builds upon [DiffusionOPSD](https://github.com/worldbench/DiffusionOPSD), whose
method is transferred here, [JoyVASA](https://github.com/jdh-algo/JoyVASA), the motion
generator, [LivePortrait](https://github.com/KwaiVGI/LivePortrait), the renderer, and
[AISHELL-1](https://www.openslr.org/33/), the training speech corpus.
