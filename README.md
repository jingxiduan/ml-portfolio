<div align="center">

# Reward-Driven Machine Learning Systems

**From ranking and feedback control to preference alignment and multimodal generative-model post-training**

[Jingxi Duan](https://github.com/jingxiduan) · [Project status](PROJECT_STATUS.md) · [Talking-head research](07-talking-head-motion-opsd) · [I2V research](08-video-opsd-i2v)

</div>

---

## Portfolio thesis

This repository studies one recurring problem across different ML systems:

> **How can an offline model be turned into a measurable decision policy, and how do we detect when the optimization target stops matching the real objective?**

The projects progress through two connected tracks:

1. **Prediction, control, and preference learning** — multi-task ranking, auction control, reward modeling, and DPO data construction.
2. **Reward-guided generative post-training** — a sequence of OPSD implementations spanning images, text-to-video, talking-head motion, and image-to-video.

Every reported number is backed by an artifact in the corresponding project. Negative results and rejected gates are kept alongside successful runs.

## Projects 01–08

| # | Project | Problem and contribution | Verified result | Status |
|---|---|---|---|---|
| 01 | [`multitask-ctr-cvr-rec`](01-multitask-ctr-cvr-rec) | Multi-task CTR/CVR ranking with shared RankMixer-style feature interaction and clicked-sample masking for CVR. | CTR AUC **0.8805**, GAUC **0.8796**; CVR AUC **0.7960** on proxy labels derived from MovieLens-1M. | Complete |
| 02 | [`rtb-bid-shading-control`](02-rtb-bid-shading-control) | Combines pCTR, a win-rate model, candidate-bid search, and PID/MPC feedback control in a second-price auction simulator. | Avazu pCTR AUC **0.7796**; PID profit **5,557** vs fixed-policy **2,678** under the documented simulation. | Complete |
| 03 | [`reward-model-dpo-pipeline`](03-reward-model-dpo-pipeline) | Preference filtering, proxy LLM judging, Bradley–Terry reward modeling, DPO-format export, and an end-to-end trainer smoke test. | Held-out RM AUC **0.6150**, pairwise accuracy **0.6317**; **2,624** DPO training pairs. The included DPO run is a 20-step smoke test, not a quality claim. | Complete pipeline |
| 04 | [`diffusion-opsd-mps`](04-diffusion-opsd-mps) | Small-scale from-scratch implementation of the DiffusionOPSD training loop on SD-Turbo with differentiable CLIP reward. | Held-out CLIPScore **+0.542**, **60.0%** per-prompt win rate over base on 20 prompts. | Mechanism reproduction |
| 05 | [`video-opsd-mps`](05-video-opsd-mps) | Extends the same loop to AnimateDiff video latents with frame-averaged CLIP alignment and temporal-consistency reward. | Held-out reward **+0.523**, **77.8%** win rate over base on 9 prompts; the training curve itself remained noisy and flat. | Video proof of concept |
| 06 | **Reserved research slot** | No public project is assigned to this number. It remains visible so the repository history is not presented as a continuous series that never existed. | No result claimed. | Reserved |
| 07 | [`talking-head-motion-opsd`](07-talking-head-motion-opsd) | Transfers OPSD to JoyVASA's 73-dimensional audio-to-motion trajectories; adds measured human target bands, motion rewards, real-video behavior cloning, diagnostics, and rendered controls. | Calibrated expression seeds reduce paired target error from **0.1202** to **0.0229–0.0238** and pass n=40 SyncNet non-inferiority; absolute lip-sync quality and human preference remain unresolved. | Flagship, active |
| 08 | [`video-opsd-i2v`](08-video-opsd-i2v) | Adapts OPSD to DynamiCrafter I2V temporal modules and studies reward hacking, naturalness constraints, gradient flow, and long-run stability. | Held-out metrics remain tied with base; V2 stopped at step **306** and V3 at step **522**. V4 changes only effective gradient clipping and is pre-registered. | Flagship, active |

## The OPSD research thread

```text
04 · image mechanism reproduction
       |
       +-- 05 · first video-latent transfer
       |
       +-- 07 · audio-to-motion post-training
       |        measured human targets + behavior cloning + avatar evaluation
       |
       +-- 08 · image-to-video post-training
                temporal LoRA + video rewards + reward-hacking and stability diagnosis
```

The optimization loop is deliberately held recognizable across projects: on-policy query states, reward-gradient positive/negative targets, detached finite-step fitting, and a slowly refreshed behavior policy. The research contribution of each extension is therefore not presented as a new optimizer. It lies in the domain-specific state representation, reward design, conditioning path, failure diagnosis, and evaluation protocol.

## Flagship evidence

### 07 · Talking-head motion

- **Controlled state space:** frozen JoyVASA and LivePortrait; LoRA updates only the motion DiT.
- **Measured targets:** amplitude, jitter, and blink bands are estimated from public human footage.
- **Real-video supervision:** 1,753 aligned four-second motion windows are available for behavior cloning.
- **Controlled evaluation:** held-out speakers, identical audio/seed comparisons, real-motion reference, same-face cross-industry control, effect sizes, bootstrap intervals, and permutation tests.
- **R3 expression control:** three calibrated seeds pass paired n=40 target-error and pose high-frequency gates; this remains a coefficient-control claim, not an independent perceptual claim.
- **Independent lip preservation:** all three seeds pass the frozen n=40 SyncNet non-inferiority margin, although low absolute confidence means lip synchronization is not solved.
- **What did not work:** the earlier absolute-jitter gate failed; finance and real-estate are not ecologically separable in the real reference footage; human preference evaluation is still missing.

Start with the [technical report](07-talking-head-motion-opsd/docs/TECH_REPORT.md), [separability analysis](07-talking-head-motion-opsd/docs/SEPARATION_ANALYSIS.md), and [current experiment record](07-talking-head-motion-opsd/docs/RESULTS_R3.md).

### 08 · Image-to-video

- **Model path:** DynamiCrafter I2V with LoRA on 136 temporal-attention targets; spatial priors remain frozen.
- **Reward path:** subject consistency, optical-flow dynamics, motion smoothness, and a differentiable pixel-naturalness penalty.
- **Core diagnosis:** full-strength reward-aligned LoRA amplified saturation, overexposure, and edge artifacts; the first naturalness implementation also zeroed its own gradient through clamping.
- **Long-run result:** the unconstrained NoNat control completed 2,000 steps, while the naturalness-constrained V2 run crossed the fixed loss-growth threshold at step 306 and was stopped.
- **What is not claimed:** current artifacts do not establish that OPSD improves held-out I2V quality.

Start with the [method](08-video-opsd-i2v/docs/METHOD.md), [long-run results](08-video-opsd-i2v/docs/RESULTS_LONG.md), and [gate log](08-video-opsd-i2v/docs/GATE_LOG.md).

## Evidence and reporting policy

Each active generative project separates four kinds of evidence:

| Evidence | What it can establish | What it cannot establish |
|---|---|---|
| Training reward and gradient diagnostics | Whether the intended optimization signal exists and changes | Perceptual quality or generalization |
| Held-out automatic metrics | Repeatable changes under a fixed evaluator | Human preference when evaluator and reward are related |
| Controlled qualitative comparisons | Visible failure modes under matched inputs and seeds | Population-level significance |
| Blind human evaluation | Perceived preference under the stated protocol | Production reliability without deployment evidence |

Training optimizations are launched through `tools/train_guard.py`. Each run records its configuration, logs, checkpoint state, gate result, and failure reason in Markdown before the next experiment begins. Small artifacts are committed to `main`; large media are retained on the documented backup branch when available.

## Reproduction

Clone the actual repository name, then use the project-local environment and instructions:

```bash
git clone https://github.com/jingxiduan/ml-portfolio.git
cd ml-portfolio

# Example: CPU-scale ranking project
cd 01-multitask-ctr-cvr-rec
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/download_data.py
python src/preprocess.py
python src/train.py --epochs 4 --batch_size 4096
```

The projects intentionally do not share one environment: `01–03` are lightweight CPU workflows, `04–05` target Apple MPS, and `07–08` require separate model weights and GPU-specific setup. Use each project's `README.md` rather than installing all dependencies together.

## Repository map

```text
ml-portfolio/
├── 01-multitask-ctr-cvr-rec/       ranking and multi-task learning
├── 02-rtb-bid-shading-control/     auction modeling and feedback control
├── 03-reward-model-dpo-pipeline/   preference and alignment pipeline
├── 04-diffusion-opsd-mps/          image OPSD reproduction
├── 05-video-opsd-mps/              text-to-video OPSD proof of concept
├── 06                              reserved; no public project
├── 07-talking-head-motion-opsd/    audio-driven avatar motion
├── 08-video-opsd-i2v/              image-to-video reward post-training
├── tools/                           shared training guard and backup helpers
├── AGENTS.md                        repository operating rules
└── PROJECT_STATUS.md                current state and roadmap
```

## Scope

This is a personal research portfolio built from public datasets and open-source model components. It does not claim production deployment, foundation-model pretraining, or ownership of the referenced base models. Dataset and model licensing constraints are documented inside the relevant projects.
