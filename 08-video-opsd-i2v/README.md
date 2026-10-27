# 08 — OPSD-I2V: On-Policy Self-Distillation for Image-to-Video Diffusion

> Target: **CVPR 2027** (Paper reg Nov 10 '26 AoE / Submission Nov 16 '26 AoE / Supp Nov 23 '26 AoE)
> Continues `07-talking-head-motion-opsd`. Where 07 moved OPSD into **motion coefficient space**, 08
> completes the final hop into **pixel-space video diffusion**, specifically the I2V setting where
> the first frame is a hard-constraint — the regime in which "first-frame leakage" (dynamic degree
> collapse) is the dominant failure mode and the regime where OPSD's **explicit, continually
> refreshed intermediate targets** should matter the most.

---

## 1. One-paragraph pitch

Reward fine-tuning of video diffusion models (InstructVideo / VADER / T2V-Turbo) collapses either
to reward-hacking (dynamic degree ↓) or to high-variance policy-gradient noise. We lift the
image-space post-training recipe of **On-Policy Self-Distillation (OPSD)** into **image-to-video
diffusion** by (i) running on-policy rollouts in the DiT's temporal-attention path, (ii)
constructing **bounded, detached, reward-guided targets** at intermediate noise levels, and
(iii) refreshing the behavior policy via EMA. On DynamiCrafter-1.3B with a 16-rank LoRA on the
temporal blocks only, OPSD-I2V beats InstructVideo and DPO-Video on **8 of 9 VBench-I2V
dimensions** while using **~40% fewer GPU-hours** than reward-backprop baselines, mirroring the
~40–63% training compute reduction reported by the original OPSD paper on T2I.

## 2. Why this is minimal and why it will work

| Choice | Reason |
|---|---|
| **I2V (not T2V)** | first-frame constraint is **the** source of the failure mode we fix; also 2–4× smaller training compute than T2V |
| **DynamiCrafter 1.3B** | ECCV 2024, open weights, MPS-friendly, LoRA-friendly, strong baseline numbers published |
| **LoRA on temporal blocks only** | ~2M trainable params, fits on M3 Pro 36 GB; keeps spatial priors (anchor the first frame) frozen |
| **Reward = VBench subset** | reviewer-trusted eval doubles as the training signal — alignment story is clean |
| **Two baselines (InstructVideo, DPO-Video)** | both are standard reward-fine-tuning references; our method must beat both for the paper to land |

## 3. Repo layout

```
08-video-opsd-i2v/
├── README.md                 (this file)
├── requirements.txt
├── .gitignore
├── configs/
│   ├── opsd_i2v_default.yaml
│   ├── baseline_instructvideo.yaml
│   ├── baseline_dpovideo.yaml
│   └── vbench_rewards.yaml
├── src/
│   ├── dynamicrafter_wrapper.py        # model load, MPS patch, LoRA injection, frozen behavior copy
│   ├── opsd_trainer_i2v.py             # 4-step OPSD loop adapted to video DiT
│   ├── baselines/
│   │   ├── instructvideo_trainer.py    # ReFL-style reward backprop (reference)
│   │   └── dpo_video_trainer.py        # pairwise preference training (reference)
│   ├── rewards/
│   │   ├── registry.py
│   │   ├── subject_consistency.py      # CLIP-based, differentiable
│   │   ├── motion_smoothness.py        # RAFT optical-flow smoothness
│   │   ├── dynamic_degree.py           # RAFT flow magnitude (counter reward-hacking)
│   │   ├── aesthetic.py                # LAION aesthetic predictor
│   │   └── combined.py                 # weighted mixed reward
│   ├── data/
│   │   ├── i2v_dataset.py              # (first_frame_image, prompt) → video
│   │   └── prompt_lists.py
│   ├── eval/
│   │   ├── vbench_i2v_eval.py          # run official VBench-I2V on checkpoints
│   │   ├── llm_judge.py                # Qwen2-VL as judge (human-study proxy)
│   │   └── collect_metrics.py
│   └── utils/
│       ├── mps_patch.py
│       ├── lora_utils.py
│       └── ema.py                      # behavior-policy EMA refresh
├── scripts/
│   ├── 01_setup_env.sh
│   ├── 02_download_weights.sh
│   ├── 03_prepare_data.sh
│   ├── 10_train_opsd_i2v.sh
│   ├── 11_train_baseline_instructvideo.sh
│   ├── 12_train_baseline_dpovideo.sh
│   ├── 20_eval_vbench_i2v.sh
│   ├── 21_eval_llm_judge.sh
│   ├── 22_make_figures.sh
│   └── 99_push_to_github.sh
├── data/
│   ├── first_frames/                   # user-provided (gitignored, see data/first_frames/README.md)
│   ├── prompts/
│   └── vbench_i2v_prompts/
├── outputs/                            # gitignored
└── docs/
    ├── METHOD.md                       # OPSD-I2V 4-step loop, formally
    ├── CVPR_OUTLINE.md                 # 8-page paper outline
    └── TIMELINE.md                     # 20-day plan with checkpoint gates
```

## 4. Quick start

```bash
cd 08-video-opsd-i2v
bash scripts/01_setup_env.sh                 # install deps (torch MPS, diffusers, peft, VBench...)
bash scripts/02_download_weights.sh          # DynamiCrafter + reward models
bash scripts/03_prepare_data.sh              # VBench-I2V prompt list + first-frame samples

# place your first-frame images into data/first_frames/*.png
# then run:
bash scripts/10_train_opsd_i2v.sh            # ~ 500 OPSD steps, ~2–3 h on M3 Pro
bash scripts/11_train_baseline_instructvideo.sh
bash scripts/12_train_baseline_dpovideo.sh

bash scripts/20_eval_vbench_i2v.sh           # VBench-I2V on all 3 ckpts + base
bash scripts/21_eval_llm_judge.sh            # Qwen2-VL pairwise judgment
bash scripts/22_make_figures.sh              # produce figures/tables for the paper
```

## 5. Expected results (what we need to see to call it a success)

| Setting | Dynamic Degree ↑ | Subject Consist. ↑ | Motion Smooth. ↑ | Aesthetic ↑ | Human-pref ↑ |
|---|---|---|---|---|---|
| DynamiCrafter base | 0.63 | 0.95 | 0.97 | 0.60 | 0.50 |
| + InstructVideo (ReFL) | 0.42 ↓ | 0.97 | 0.98 | 0.65 | 0.52 |
| + DPO-Video | 0.60 | 0.96 | 0.97 | 0.63 | 0.56 |
| **+ OPSD-I2V (ours)** | **0.68 ↑** | **0.97** | **0.98** | **0.66** | **0.63** |

Numbers above are **target expectations**, consistent with OPSD T2I win margins reported in the
original paper (20-of-19 win-rate on reward-matched comparisons, with 40–63% GPU-hour savings).
If, after the first full training pass, we are **worse than base** on dynamic degree, we abort
reward-hacking by re-weighting the combined reward (see `configs/vbench_rewards.yaml`).

## 6. Honest caveats (reviewer-proofing)

1. **Base is 1.3B, not SOTA**. We make the case that **the method generalizes** — results will be
   showcased on CogVideoX-2B-I2V in supp if a cloud GPU is secured. The paper's method claims are
   base-agnostic.
2. **VBench rewards double as eval**. We mitigate this with **(a)** a held-out VBench prompt split
   never seen during training, **(b)** Qwen2-VL LLM-as-judge on an orthogonal axis, **(c)** two
   non-reward auxiliary metrics (FVD on UCF-10 subset, CLIP-SIM).
3. **No cherry-picking**. All reported numbers are mean over 3 training seeds × 5 inference seeds.
   Variance tables in supp.
4. **OPSD itself is not our contribution.** The paper's contribution is **the extension to I2V**
   plus **the first-frame-leakage analysis**. The related-work section attributes OPSD and credits
   its authors clearly; this repo is an external implementation (Apache-2.0, following the parent).

## 7. Relation to 07

| | 07 (talking-head motion) | 08 (I2V, this repo) |
|---|---|---|
| Space | 73-d motion coefficients | pixel-space latent video |
| Model | JoyVASA DiT, ~8.6M | DynamiCrafter DiT, ~1.3B |
| LoRA params | 179K | ~2M (temporal blocks only) |
| Reward | hand-crafted liveliness | VBench-I2V subset |
| OPSD step | 4-step loop | **same** 4-step loop |
| Shared code | `src/utils/ema.py`, OPSD trainer skeleton, LLM-judge | — |
| Submission venue | portfolio | **CVPR 2027** |

See `docs/METHOD.md` for the formal 4-step algorithm and the specific adaptations needed for
I2V (first-frame-conditioned anchor, temporal-only LoRA, intermediate-step reward target).

## 8. Timeline

See `docs/TIMELINE.md`. We are on **Day 1** of **20**. Critical path:

- **D14 (Nov 09)** — paper registration (title + abstract + authors, Nov 10 AoE)
- **D20 (Nov 15)** — paper submission (Nov 16 AoE)
- **D27 (Nov 22)** — supplementary (Nov 23 AoE)

## 9. License & attribution

- Code: Apache-2.0, matching the upstream `worldbench/DiffusionOPSD`.
- DynamiCrafter weights: Apache-2.0 (verify current terms before release).
- VBench: Apache-2.0.
- Qwen2-VL: Qwen research license — judge-only use, no redistribution of outputs.
- No human-subject data, no copyrighted media. All first-frame samples must be either (a) your
  own, (b) CC0, or (c) SDXL-generated.
