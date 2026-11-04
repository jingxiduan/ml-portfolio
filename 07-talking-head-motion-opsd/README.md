<div align="center">

# Talking-Head Motion Post-Training

**Reward-guided adaptation of an audio-to-motion diffusion model, with measured human targets and explicit failure gates**

**Jingxi Duan**

[Technical report](docs/TECH_REPORT.md) · [Separability analysis](docs/SEPARATION_ANALYSIS.md) · [Current results](docs/RESULTS_R3.md) · [Gate log](docs/GATE_LOG.md)

</div>

---

## Run the demo first

The complete demo path is documented in [`docs/DEMO.md`](docs/DEMO.md). It includes a dependency-free CPU mock, a CUDA render command with wall-clock and sampled VRAM reporting, pre-generated held-out comparisons, weight download instructions, and failure diagnosis.

```bash
# CPU-only mock: no model weights, no CUDA
python3 scripts/demo.py --mode mock

# Real GPU render: Base + three calibrated adapters, matched inputs
bash scripts/h20_render_setup.sh
python3 scripts/demo.py --mode gpu --seconds 4 --seed 0 --out outputs/demo/gpu_r3
```

The checked-in visual comparison is [female Base vs calibrated GIF](outputs/new_face_demo_r3/female/female_new_face_base_vs_r3.gif) and [male Base vs calibrated GIF](outputs/new_face_demo_r3/male/male_new_face_base_vs_r3.gif). The mock is a pipeline/UI smoke test; only the GPU path loads the model.

## Contribution boundary

This project is an independent domain adaptation built on public methods and open-source components. `DiffusionOPSD` supplies the public OPSD method description; JoyVASA supplies the audio-to-motion backbone; LivePortrait supplies the frozen renderer. The contribution here is the motion-space reward design, bounded target construction, LoRA training integration, failure diagnosis, and held-out evaluation protocol—not ownership of those upstream models or the original OPSD algorithm.

## 📌 ABSTRACT

> **DiffusionOPSD optimizes against a single scalar image-preference reward. Can the same reward-guided fine-tuning mechanism be retargeted to a human-motion domain, driven by motion statistics measured from real speakers instead of visual preference alone, while staying inside explicit, pre-registered failure gates?**

**Core contribution: moving OPSD from a visual-preference reward to measured human-motion targets.** This project retargets DiffusionOPSD's query/target/EMA optimization structure onto a 73-coefficient audio-to-motion trajectory space produced by JoyVASA and rendered by frozen LivePortrait:

1. **New measured reward targets:** per-industry amplitude, jitter, and blink bands estimated from real speaker footage (HDTF news anchors, CelebV-HQ) replace a single hand-tuned visual-preference reward.
2. **A bounded motion reward:** a peaked Gaussian target around the measured band replaces monotone amplitude maximization, removing the earlier **2.4–2.7×** overshoot it caused.
3. **Diagnosed and fixed a blink-target bug:** an excessive blink target drove the eyes toward half-closed; retuning it restored eye-aspect-ratio to untrained-base levels.
4. **A real-video behavior-cloning path:** 1,753 aligned motion windows from real footage supplement the reward signal through an auxiliary supervised objective.
5. **Pre-registered failure gates kept in the main result:** jitter, lip-sync, and ecological-separability gates are evaluated and reported as part of the headline result, not filtered out after the fact.

Three trained policies reach **0.85–0.89×** of their measured amplitude targets on held-out speakers, and the calibrated expression policy reduces mean held-out target error from **0.1202** to **0.0229–0.0238** across three seeds (n=40 paired clips; all three 95% bootstrap intervals exclude zero). Full scope boundaries, failed gates, and open questions are documented in **Limitations** below.

## 🌟 HIGHLIGHTS

### What this project built

- **Domain transfer of OPSD to motion space.** Retained DiffusionOPSD's query/target/EMA structure and retargeted it from pixel latents to a 73-coefficient audio-conditioned motion trajectory.
- **Measured, not hand-tuned, reward targets.** Amplitude, jitter, and blink targets are estimated from public human footage per industry, instead of being selected only by visual preference.
- **Fixed two concrete reward-design bugs.** The monotone amplitude reward caused 2.4–2.7× overshoot; replacing it with a peaked Gaussian target fixed it. An excessive blink target caused half-closed eyes; retuning it restored normal eye-aspect-ratio.
- **Added a real-video supervision path.** 1,753 aligned real-motion windows support an auxiliary behavior-cloning objective alongside the reward.
- **Auditable, pre-registered evaluation.** Jitter, lip-sync non-inferiority, and ecological-separability gates are all reported as part of the main result regardless of outcome.

Failed gates, sample sizes, and open questions are in **Limitations** below.

> This is a personal research implementation based on [DiffusionOPSD](https://github.com/worldbench/DiffusionOPSD), [JoyVASA](https://github.com/jdh-algo/JoyVASA), and [LivePortrait](https://github.com/KwaiVGI/LivePortrait). It is not affiliated with their authors.

## New licensed hold-out face demos

| New face A | New face B |
|---|---|
| ![Pexels face A Base versus calibrated policy](outputs/new_face_demo_r3/female/female_new_face_base_vs_r3.gif) | ![Pexels face B Base versus calibrated policy](outputs/new_face_demo_r3/male/male_new_face_base_vs_r3.gif) |
| [Base MP4](outputs/new_face_demo_r3/female/expression_r3__base.mp4) · [Calibrated policy MP4](outputs/new_face_demo_r3/female/expression_r3__seed0.mp4) | [Base MP4](outputs/new_face_demo_r3/male/expression_r3__base.mp4) · [Calibrated policy MP4](outputs/new_face_demo_r3/male/expression_r3__seed0.mp4) |

Both 1024×1024 inputs were extracted from previously unused 4K Pexels License videos. They were not part of training, behavior calibration, or model selection. Base and the calibrated policy use identical portrait, held-out audio, generation seed, and frozen LivePortrait renderer. The test isolates motion changes; it does not claim improved skin texture, lighting, or color. [Source URLs, authors, timestamps, license and checksums](data/portraits_new_licensed/SOURCES.json).

### Historical evaluation portraits

![portrait sheet](outputs/demo_gifs_frontal/portrait_sheet.png)

*Evaluation portraits were filtered for near-frontal pose, open eyes, and closed-mouth source frames to reduce renderer confounds.*

----------------------------------------

## Method

At query state $s=(c,z_q,\sigma_q)$, the frozen behavior policy defines a clean anchor:

$$
y_0 = z_q - \sigma_q v_{\mathrm{old}}(z_q,c,\sigma_q).
$$

A normalized reward gradient defines bounded positive and negative query states:

$$
d = \frac{\nabla_{z_q} R(y_0)}{\lVert\nabla_{z_q} R(y_0)\rVert_2 + \varepsilon},
\qquad z_q^{\pm}=z_q\pm c_{\mathrm{adv}}d.
$$

The behavior policy maps these perturbed states to detached targets, and the trainable LoRA policy fits the positive target while enforcing a margin from the negative target:

$$
\mathcal{L}_{\mathrm{OPSD}}
= \lVert\hat y_\theta-\operatorname{sg}(y_0^+)\rVert_2^2
+\lambda_-\max\!\left(0,m-\lVert\hat y_\theta-\operatorname{sg}(y_0^-)\rVert_2^2\right).
$$

### Minimal Pseudocode (Motion Space)

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

| Hyperparameter | DiffusionOPSD (SD3.5-M) | Talking-Head Motion OPSD (JoyVASA) |
|---|---|---|
| **LoRA** | PEFT | r=8, alpha=8 (~179K params) on Motion DiT |
| **Target Radius (`c_adv`)** | 5 (ρ=0.10) | 3.0 (`reward-step-size`) |
| **EMA Refresh** | Every outer update | Every 10 steps, decay 0.5 |
| **Optimizer** | AdamW | AdamW, lr 1e-4 with 200-step linear warmup |

----------------------------------------

## 📊 MAIN RESULTS

### 1. Per-Industry Reward Targets (Measured from Real Video)

| Industry | Video Proxy | Amplitude (0.3–3 Hz) | Jitter (6–12.5 Hz) | Blink /min |
|---|---|---|---|---|
| **Finance** | HDTF news anchors | **1.384°** | **0.094°** | 18.1 |
| **Real Estate** | CelebV-HQ mid-amplitude | **1.257°** | **0.118°** | 20.3 |
| **Short Video** | CelebV-HQ high-amplitude | **3.039°** | **0.349°** | 22.6 |

### 2. Evaluation on Held-Out Speakers (Disjoint from Training)

| Variant | Amplitude (Measured) | vs. Target | vs. Untrained Base |
|---|---|---|---|
| **Base (Untrained)** | 0.924° | · | 1.00x |
| **Finance Persona** | 1.213° | **0.88x** | **1.31x** |
| **Real Estate Persona** | 1.114° | **0.89x** | **1.21x** |
| **Short Video Persona** | 2.597° | **0.85x** | **2.81x** |

### 3. Calibrated Expression Dynamics (n=40 Paired Clips)

| Variant | Expression RMS | Mean Target Error | Pose High-Frequency RMS | Gate |
|---|---:|---:|---:|---|
| **Base** | 0.2727 | 0.1202 | 0.2406° | Reference |
| **Seed 0** | 0.3872 | **0.0229** | **0.1801°** | Pass |
| **Seed 1** | 0.3849 | **0.0238** | **0.1822°** | Pass |
| **Seed 2** | 0.3860 | **0.0234** | **0.1848°** | Pass |

All three paired bootstrap intervals for target-error improvement exclude zero. Full interpretation of this result relative to the optimized statistic is in **Limitations** below; see [`docs/RESULTS_R3.md`](docs/RESULTS_R3.md) for the complete protocol.

![Calibrated expression policy: strict four-second matched render](outputs/expression_render_set_r3/clip000_fourway.gif)

----------------------------------------

## 🔬 DIAGNOSTICS & ABLATION STUDIES

### 1. Eye Aspect Ratio (EAR) Corrective Ablation
Early post-trained models exhibited half-closed eyes. We diagnosed this as being driven by an excessive blink target (`blink_target = 0.18`). Correcting this to `0.12` and increasing the overshoot penalty completely resolved the issue, returning EAR to untrained base levels:

| Configuration | Eye Aspect Ratio (EAR) | vs. Untrained Base | Fraction of Closed Frames |
|---|---|---|---|
| **Base (Untrained)** | **0.343** | 1.00x | ~0.05 |
| **Initial Persona (Blink 0.18)** | **0.102** | 0.30x (Partially Closed) | ~0.42 |
| **Revised Persona (Blink 0.12)** | **0.348** | **1.01x (Open)** | ~0.05 |

### 2. Temporal Scale and Translation Stabilization
Raw motion coefficients suffered from head size drift during driving. We resolved this via `stabilize_scale_translation` (applying a median filter and a ±2% clamp on scale dimension 63, and median filtering on translation dimensions 64–66):
- **Face-width std/mean:** **2.04% → 1.15%** (Finance).

----------------------------------------

## 🎬 COMPREHENSIVE VISUAL EVIDENCE GALLERY

The gallery organizes qualitative evidence by research question and control type, with **soft, realistic, low-saturation skin tones** used wherever comparable renders are available.

### **Multi-Subject Persona Generalization**
*Description:* Two distinct subjects are presented per industry to assess within-industry consistency and cross-subject generalizability. The renders use soft, natural color grading without artificial warm saturation.
*   **BASE**: Pretrained model without post-training.
*   **PERSONA**: Model post-trained with OPSD against each industry's measured target band.

| Industry | Qualitative Comparison (BASE vs. PERSONA - Soft Tone) | High-Definition Video |
|---|---|---|
| **Finance** (Formal) | ![finance](outputs/persona_grids/finance_two_people_base_vs_persona.gif) | [mp4](outputs/persona_grids/finance_two_people_base_vs_persona.mp4) |
| **Real Estate** (Warm) | ![realestate](outputs/persona_grids/realestate_two_people_base_vs_persona.gif) | [mp4](outputs/persona_grids/realestate_two_people_base_vs_persona.mp4) |
| **Short Video** (Animated) | ![shortvideo](outputs/persona_grids/shortvideo_two_people_base_vs_persona.gif) | [mp4](outputs/persona_grids/shortvideo_two_people_base_vs_persona.mp4) |

---

### **Real-Motion Reference Condition (Upper Bound)**
*Description:* Direct transfer of recorded human motion coefficients onto the portrait. This represents the reference target that the OPSD policy is optimized to approach.

| Industry | Qualitative Comparison (BASE vs. REAL) | High-Definition Video |
|---|---|---|
| **Finance** | ![finance_real](outputs/demos/finance_v8_base_vs_real.gif) | [mp4](outputs/demos/finance_v8_base_vs_real_1080.mp4) |
| **Real Estate** | ![realestate_real](outputs/demos/realestate_v8_base_vs_real.gif) | [mp4](outputs/demos/realestate_v8_base_vs_real_1080.mp4) |
| **Short Video** | ![shortvideo_real](outputs/demos/shortvideo_v8_base_vs_real.gif) | [mp4](outputs/demos/shortvideo_v8_base_vs_real_1080.mp4) |

---

### **Same-Face Cross-Industry Control**
*Description:* Control experiment holding portrait and driving audio identical. Any behavioral difference is purely due to the target persona checkpoint.

![same face, three industries](outputs/industry_gifs_v3/control/same_face_p00_-R5aAhsrS9w_3_0__base_vs_three.gif)
<div align="center"><code>BASE | FINANCE | REAL ESTATE | SHORT VIDEO</code></div>

---

### **Single-Subject Persona Comparisons (Soft Tone)**
*Description:* Six single-subject comparisons using soft, low-saturation grading. These are displayed directly rather than hidden behind directory links. Slight grey/background flicker remains in these renders.

| Finance | Finance |
|---|---|
| ![finance p00](outputs/industry_gifs_v3/finance/finance__00_p00_-R5aAhsrS9w_3_0__base_vs_finance.gif) | ![finance p03](outputs/industry_gifs_v3/finance/finance__01_p03_0tx4o3yXM64_12__base_vs_finance.gif) |
| Real estate | Real estate |
| ![real estate p01](outputs/industry_gifs_v3/realestate/realestate__00_p01_0wLgQQGob1U_13__base_vs_realestate.gif) | ![real estate p04](outputs/industry_gifs_v3/realestate/realestate__01_p04_0TvKsVxgbF4_2__base_vs_realestate.gif) |
| Short video | Short video |
| ![short video p02](outputs/industry_gifs_v3/shortvideo/shortvideo__00_p02_0Ba6y1Y8JjU_4__base_vs_shortvideo.gif) | ![short video p05](outputs/industry_gifs_v3/shortvideo/shortvideo__01_p05_0wLgQQGob1U_0__base_vs_shortvideo.gif) |

---

### **High-Amplitude Frontal-Portrait Ablation**
*Description:* Eight frontal-portrait comparisons driven by the high-amplitude checkpoint. This checkpoint overshoots the measured human motion band and is retained as an ablation result.

| Clip 00 | Clip 01 |
|---|---|
| ![c00 celebA-HQ 004](outputs/demo_gifs_frontal/c00__celebahq_004__base_vs_v3.gif) | ![c01 celebA-HQ 004](outputs/demo_gifs_frontal/c01__celebahq_004__base_vs_v3.gif) |
| ![c00 celebA-HQ 005](outputs/demo_gifs_frontal/c00__celebahq_005__base_vs_v3.gif) | ![c01 celebA-HQ 005](outputs/demo_gifs_frontal/c01__celebahq_005__base_vs_v3.gif) |
| ![c00 celebA-HQ 007](outputs/demo_gifs_frontal/c00__celebahq_007__base_vs_v3.gif) | ![c01 celebA-HQ 007](outputs/demo_gifs_frontal/c01__celebahq_007__base_vs_v3.gif) |
| ![c00 celebA-HQ 009](outputs/demo_gifs_frontal/c00__celebahq_009__base_vs_v3.gif) | ![c01 celebA-HQ 009](outputs/demo_gifs_frontal/c01__celebahq_009__base_vs_v3.gif) |

---

### **Calibrated Restrained-Motion Comparisons**
*Description:* Six comparisons under the soft-tone grading pipeline. These demonstrate the calibrated, restrained-motion checkpoint across held-out clips.

| Clip 00 | Clip 01 |
|---|---|
| ![c00 HDTF 001](outputs/demo_gifs_v4/c00__hdtf_AdamKinzinger0_001__base_vs_v4.gif) | ![c01 HDTF 001](outputs/demo_gifs_v4/c01__hdtf_AdamKinzinger0_001__base_vs_v4.gif) |
| ![c00 HDTF 002](outputs/demo_gifs_v4/c00__hdtf_AdamKinzinger0_002__base_vs_v4.gif) | ![c01 HDTF 002](outputs/demo_gifs_v4/c01__hdtf_AdamKinzinger0_002__base_vs_v4.gif) |
| ![c00 HDTF 003](outputs/demo_gifs_v4/c00__hdtf_AdamKinzinger0_003__base_vs_v4.gif) | ![c00 HDTF 004](outputs/demo_gifs_v4/c00__hdtf_AdamKinzinger0_004__base_vs_v4.gif) |

---

### **Synchronized High-Definition Deliverables**
*Description:* GitHub README pages do not reliably render repository MP4 files inline. Therefore each master is represented by its visible GIF preview below, with the synchronized high-definition MP4 available beside it.

| Industry master | Visible preview | Synchronized master |
|---|---|---|
| **Finance** | ![finance master preview](outputs/persona_grids/finance_two_people_base_vs_persona.gif) | [Play/download MP4](outputs/persona_grids/finance_two_people_base_vs_persona.mp4) |
| **Real estate** | ![real-estate master preview](outputs/persona_grids/realestate_two_people_base_vs_persona.gif) | [Play/download MP4](outputs/persona_grids/realestate_two_people_base_vs_persona.mp4) |
| **Short video** | ![short-video master preview](outputs/persona_grids/shortvideo_two_people_base_vs_persona.gif) | [Play/download MP4](outputs/persona_grids/shortvideo_two_people_base_vs_persona.mp4) |

---

### **Real-Video Behavior-Cloning + OPSD Prototype (600 Steps)**
*Description:* Alternates OPSD updates with supervised fitting to 1,753 extracted real-video motion windows. This gallery is qualitative evidence only: no blind preference test or independent lip-sync gain has been established. *Rendering note: this sample uses the warm/sharp tone.*

![bc_v1](outputs/demos/bc_v1_base_vs_trained.gif)
<div align="center">Left: <code>BASE</code>. Right: <code>TRAINED</code>. Same portrait, audio and seed. [1080-line mp4](outputs/demos/bc_v1_base_vs_trained_1080.mp4)</div>

----------------------------------------

## ⚠️ LIMITATIONS & REJECTED GATES

| Metric / Aspect | Performance Status |
|---|---|
| **Visual Realism** | **Not solved.** Skin, color, and teeth textures are determined by the frozen LivePortrait warp. |
| **Pre-registered Jitter Gate** | **Failed.** Pre-registered gate of `jitter_hp ≤ 1.3x base` (18.36) was not satisfied; OPSD naive attained **28.97 (2.05x base)**. |
| **Lip Synchronisation** | **Relative gate passed, absolute quality unresolved.** Three seeds pass the frozen paired SyncNet non-inferiority margin, but all variants have low confidence and 7.7–8.8-frame mean offsets. |
| **Expression dynamics** | **Coefficient and lip-preservation gates passed.** Three calibrated seeds and n=40 paired evaluations are complete, but expression RMS is also the optimized statistic, so this establishes coefficient control rather than an independent perceptual-quality claim. |
| **Ecological separability** | Four classes (base + 3 amplitude policies) are separable at **0.831** accuracy, but most of that separation is explained by amplitude alone; the real reference data do not support a general claim that finance and real-estate speakers have distinct motion styles. |
| **Behavior cloning** | Uses 1,753 real-video motion windows, but its perceptual gain has not been established by real participants. |
| **Human preference** | **Not run.** Gallery clips are diagnostic artifacts, not blinded perceptual evidence. A randomized, balanced 40-pair expression study is executable from [`../human-study`](../human-study); no responses are fabricated. |

See [`docs/RESULTS_R3.md`](docs/RESULTS_R3.md) for the completion protocol and exact claim boundary.

----------------------------------------

## Reproduction

```bash
# 1. environment (downloads JoyVASA + LivePortrait + HuBERT-chinese, ~2.5 GB)
bash scripts/h20_render_setup.sh

# 2. data preparation (AISHELL-1 + public clips)
python scripts/fetch_public_data.py --aishell-speakers 75 --hdtf 40 --celebv 300
python scripts/build_audio_segments.py --n-train 2000 --n-eval 100

# 3. measure pose/industry targets and non-lip expression targets
python scripts/calibrate_industry_bands.py
python scripts/calibrate_behaviour_bands.py

# 4. run training only after freezing a config and gate record
python ../tools/train_guard.py --selftest
ROUNDS=3000 bash scripts/train_industries.sh

# Calibrated expression runs use the target/tolerance written above and unique output dirs;
# the exact command and outcome must be appended to docs/RESULTS_R3.md.

# 5. held-out evaluation and matched demo rendering
python scripts/eval_industries.py --n-clips 12
python scripts/render_industry_demo.py --gpu 0
```

----------------------------------------

## 📁 REPOSITORY LAYOUT

```
07-talking-head-motion-opsd/
├── src/                  # reward (motion_reward*.py), OPSD trainers, engine, video metrics
├── scripts/              # data prep, calibration, training scripts, rendering
├── data/                 # manifests and data descriptors
├── outputs/              # figures/, demos/, and checkpoints
└── docs/                 # TECH_REPORT, SEPARATION_ANALYSIS, HANDOFF
```

----------------------------------------

## 🙏 ACKNOWLEDGEMENTS

This repository builds upon [DiffusionOPSD](https://github.com/worldbench/DiffusionOPSD) (methodology), [JoyVASA](https://github.com/jdh-algo/JoyVASA) (motion generator), [LivePortrait](https://github.com/KwaiVGI/LivePortrait) (implicit warping renderer), and [AISHELL-1](https://www.openslr.org/33/) (corpus).
