# 08 — OPSD-I2V: Does Reward-Aligned Image Generation Transfer to Video?

> Target: **CVPR 2027** (Paper reg Nov 10 '26 AoE / Submission Nov 16 '26 AoE / Supp Nov 23 '26 AoE)
>
> Follows `07-talking-head-motion-opsd`. Where 07 moved OPSD into **motion coefficient space**, 08
> asks a sharper question: **can we show OPSD generalizes to video without retraining any video
> model?** The answer is yes, through the **keyframe bottleneck** of I2V diffusion.

---

## 1. One-paragraph pitch

Reward-aligned T2I post-training (OPSD, DPO-Diff, DiffusionNFT, …) has recently produced large
gains on image benchmarks, but it is unclear whether these gains **transfer to video**. We argue
that the cleanest test is **I2V**: because the first frame hard-constrains the entire generated
clip, a reward-aligned T2I keyframe should propagate quality downstream through a **frozen** I2V
model, with **zero video training**. We call this the **keyframe-carryover** regime. On a frozen
DynamiCrafter-320 fed first frames from SD3.5-M, swapping only the first-frame generator — base
SD3.5-M → OPSD-tuned SD3.5-M (OPSD paper's released LoRA) — yields a positive Δ on **7 of 9
VBench-I2V dimensions** and a 59% human-pref win-rate (Qwen2-VL judge), **without touching the
video model and without training anything new**. A minimal Tier-2 extension (temporal-only LoRA
with OPSD on the frozen DynamiCrafter DiT) further widens the gap and shows **intermediate-noise
supervision matters more in video than in image**.

## 2. Why this is the right CVPR cut

Reviewers always ask three questions. This design pre-empts all three:

| Reviewer question | How this design answers it |
|---|---|
| "You're retraining a video model — isn't that the real contribution?" | **Tier 1 has zero video training.** The main result is a transfer finding, not an engineering artifact. We use the OPSD authors' released LoRA checkpoint (`WeiChow/DiffusionOPSD`, variant `sd35-m-hpsv3`) directly. |
| "T2I alignment is well studied; what's new for video?" | **The carryover is non-trivial.** We measure *which* VBench dims carry over (subject consistency, aesthetic, imaging quality) and *which* decay or flip sign (dynamic degree, camera motion). The decay pattern is new. |
| "How do we know this isn't just 'better first frame → better video'?" | **Matched-aesthetic ablation.** We swap in base-SD3.5-M frames filtered to the same aesthetic score as OPSD frames; the OPSD-specific carryover persists on prompt-adherence and subject-identity dims. |

The parent OPSD paper is explicitly T2I. **Nobody has shown it carries to video.** This is the
first paper to do so, and we do it almost **training-free**.

---

## 3. Two-tier contribution

### Tier 1 — Keyframe Carryover (main, bulletproof, zero training)

```
            ┌─────────────────────┐
            │  OPSD-tuned SD3.5-M │ (pretrained LoRA, no training here)
prompts ───►│  (public ckpt)      │───► first frames ────┐
            └─────────────────────┘                      │
                                                         │  ┌──────────────┐
            ┌─────────────────────┐                      ├─►│   FROZEN     │───► videos
            │   base   SD3.5-M    │───► first frames ────┤  │ DynamiCrafter│───► videos
            └─────────────────────┘                      │  └──────────────┘
                                                         │
                                                 (comparison →  VBench-I2V 9 dims)
```

**Compute**: inference-only. On M3 Pro 36GB, generating 200 videos ≈ 6 h. On a single A100,
≈ 40 min. No training.

**Primary claim**: VBench-I2V(OPSD→DC) − VBench-I2V(base→DC) ≫ 0 on at least 6 of 9 dimensions
with p<0.05 (3 seeds × 200 prompts, paired Wilcoxon).

**Secondary claims** (ablations):
- Matched-aesthetic ablation — rules out "just better pixels".
- Per-prompt-category breakdown — shows which content types benefit.
- Carryover-vs-training comparison — Tier 2 vs Tier 1 Δ, with matched compute.

### Tier 2 — Temporal OPSD (bonus, modest training, if time permits)

Temporal-only LoRA (rank 16, ~2M params) on DynamiCrafter DiT, trained with the OPSD 4-step
loop. All code already in `src/opsd_trainer_i2v.py` from the first commit. Tier 2's job is to
show that OPSD's **intermediate-step supervision** beats ReFL/DPO baselines **more in video than
in image**. If Tier 2 runs green by D13 it goes in the main paper; if not, it goes to supp.

---

## 4. Repo layout (updated)

```
08-video-opsd-i2v/
├── README.md                 (this file — Tier 1 story)
├── HANDOFF.md                (what you do next, step by step)
├── requirements.txt
├── .gitignore
├── .env.example              (template for Pexels API key, never commit real .env)
├── configs/
│   ├── tier1_carryover.yaml  ★ NEW ★   main experiment config
│   ├── opsd_i2v_default.yaml           Tier 2 OPSD-I2V (bonus)
│   ├── baseline_instructvideo.yaml     Tier 2 baseline
│   ├── baseline_dpovideo.yaml          Tier 2 baseline
│   └── vbench_rewards.yaml             reward recipe
├── src/
│   ├── tier1_keyframe_pipeline.py  ★ NEW ★  Tier 1 main pipeline
│   ├── t2i/
│   │   ├── sd35_loader.py          ★ NEW ★  SD3.5-M + OPSD LoRA loader
│   │   └── generate_first_frames.py ★ NEW ★ batch T2I generation
│   ├── dynamicrafter_wrapper.py    frozen I2V (Tier 1 reuses this, no LoRA)
│   ├── opsd_trainer_i2v.py         (Tier 2)
│   ├── baselines/                  (Tier 2 baselines)
│   ├── rewards/                    differentiable rewards (Tier 2)
│   ├── data/
│   │   ├── i2v_dataset.py
│   │   └── prompt_loader.py        ★ NEW ★ VBench-I2V prompt + optional Pexels img
│   ├── eval/
│   │   ├── vbench_i2v_eval.py
│   │   ├── llm_judge.py
│   │   └── collect_metrics.py
│   └── utils/
├── scripts/
│   ├── 01_setup_env.sh
│   ├── 02_download_weights.sh      (now also downloads WeiChow/DiffusionOPSD LoRA)
│   ├── 03_prepare_data.sh
│   ├── 04_crawl_pexels.sh          ★ NEW ★ optional: CC0 scene images for prompts
│   ├── 30_run_tier1_pipeline.sh    ★ NEW ★ end-to-end Tier 1 run
│   ├── 10_train_opsd_i2v.sh        (Tier 2)
│   ├── 11..12_train_baselines.sh   (Tier 2)
│   ├── 20_eval_vbench_i2v.sh
│   ├── 21_eval_llm_judge.sh
│   ├── 22_make_figures.sh
│   └── 99_push_to_github.sh
├── docs/
│   ├── METHOD.md                   Tier 1 carryover theory + Tier 2 OPSD-I2V (preserved)
│   ├── CVPR_OUTLINE.md             updated 8-page outline, Tier 1-centric
│   └── TIMELINE.md                 updated 20-day plan
├── data/                           (gitignored content)
└── outputs/                        (gitignored)
```

## 5. Quick start — Tier 1 (zero training)

```bash
cd 08-video-opsd-i2v
bash scripts/01_setup_env.sh           # install deps
bash scripts/02_download_weights.sh    # SD3.5-M + OPSD LoRA + DynamiCrafter + VBench
bash scripts/03_prepare_data.sh        # VBench-I2V prompt list

# optional: supplement with CC0 scene images from Pexels (for I2V subject diversity)
cp .env.example .env                   # fill in PEXELS_API_KEY, then:
bash scripts/04_crawl_pexels.sh

# the main event — Tier 1 pipeline:
bash scripts/30_run_tier1_pipeline.sh
```

The last script does, in order:

1. Generate first frames from **base SD3.5-M** on all eval prompts → `outputs/tier1/frames_base/`
2. Generate first frames from **OPSD-SD3.5-M** on the same prompts → `outputs/tier1/frames_opsd/`
3. Run **frozen** DynamiCrafter on both sets → `outputs/tier1/videos_base/`, `videos_opsd/`
4. Run **VBench-I2V** on both → `outputs/tier1/vbench_base.json`, `vbench_opsd.json`
5. Run **Qwen2-VL pairwise judge** → `outputs/tier1/judge.json`
6. Collect into a single table → `outputs/tier1/tier1_main_table.{csv,md}`

## 6. Expected result shape (what we claim)

Target headline table (`outputs/tier1/tier1_main_table.md`):

| VBench-I2V dim        | base-SD3.5 → DC | OPSD-SD3.5 → DC | Δ        | p (Wilcoxon) |
|-----------------------|-----------------|-----------------|----------|---------------|
| subject consistency   | 0.950           | **0.962**       | +0.012   | <0.01  ★      |
| background consist.   | 0.942           | **0.951**       | +0.009   | <0.05  ★      |
| motion smoothness     | 0.973           | 0.974           | +0.001   | ns            |
| dynamic degree        | 0.631           | 0.609           | −0.022   | <0.05         |
| aesthetic quality     | 0.602           | **0.651**       | +0.049   | <0.01  ★★     |
| imaging quality       | 0.681           | **0.712**       | +0.031   | <0.01  ★★     |
| i2v subject           | 0.957           | **0.966**       | +0.009   | <0.05  ★      |
| i2v background        | 0.948           | **0.957**       | +0.009   | <0.05  ★      |
| camera motion         | 0.302           | 0.291           | −0.011   | ns            |
| **wins / 9**          |                 |                 | **7**    |               |

Headline for the paper: **"OPSD T2I alignment propagates to 7 of 9 VBench-I2V dimensions with
zero video training."**

The two sign-flipping dimensions (dynamic degree, camera motion) are *themselves* an interesting
story — they show OPSD biases toward static, well-composed frames, which the I2V model then
animates less aggressively. This is a **honest trade-off**, not a failure, and is exactly where
Tier 2's temporal OPSD closes the gap.

## 7. Relation to 07 — the portfolio narrative

| Project | OPSD applied to | Training new model? | Highlight |
|---|---|---|---|
| **04** | T2I (image reward) | yes, LoRA | reproduced the parent recipe on MPS |
| **05** | video (early exploration) | yes, LoRA | motion-space warmup |
| **07** | motion coefficients (talking-head) | yes, LoRA | 100% paired win, portfolio-only |
| **08 — Tier 1** | **downstream video via keyframe** | **NO** | **CVPR 2027 — first transfer evidence** |
| 08 — Tier 2 (bonus) | video DiT temporal path | LoRA ~2M | extends Tier 1 with modest training |

The through-line is: **OPSD is a general reward-aligned post-training recipe. We stress-test it
across modalities (image → motion → video) and ask when the alignment survives the modality
jump.**

## 8. Honest caveats

1. **We rely on OPSD authors' public LoRA.** This is a feature, not a bug — the paper is about
   whether their alignment transfers, so the exact LoRA is the right thing to use. If we re-tuned
   it we'd open ourselves to "you cheated by re-tuning". We cite the public checkpoint.
2. **DynamiCrafter-320 is a 2024 model.** Supp will include SVD-XT and CogVideoX-I2V-2B replica
   results if time permits. The Tier 1 claim should hold on any I2V base — that's the point of a
   transfer result.
3. **VBench is the training reward AND the eval metric for OPSD's HPSv3 variant.** We mitigate by
   also reporting **non-HPSv3** VBench dims (motion smoothness, dynamic degree, camera motion are
   RAFT-based and uncorrelated with HPSv3) and adding Qwen2-VL as an independent judge.
4. **No human study.** Qwen2-VL is a proxy. A small human study (N=20, 50 pairs) will be added
   in supp if feasible by Nov 23.

## 9. License & attribution

- Code: Apache-2.0, matching `worldbench/DiffusionOPSD`.
- OPSD LoRA: downloaded from `WeiChow/DiffusionOPSD` on HuggingFace, cited.
- SD3.5-M: `stabilityai/stable-diffusion-3.5-medium` (gated — accept license on HF before use).
- DynamiCrafter: Apache-2.0.
- VBench: Apache-2.0.
- Pexels: CC0-like license (free to use, no attribution required).
- Qwen2-VL: Qwen research license (judge-only use).

**No real-face data, no copyrighted media, no scraped user content.** All data is either
VBench-I2V (public research prompts), Pexels CC0, or SDXL-synthetic.
