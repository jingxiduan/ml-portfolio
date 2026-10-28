# CVPR 2027 — 8-page Paper Outline (Tier 1 centric)

> **⚠ 2026-10-28 status note.** This outline was written before the first live
> runs. The mini-pipeline (n=2) outputs **do not match the originally hoped-for
> story** of "clean OPSD improvement on video." Instead we observe:
>
> - OPSD first-frames exhibit the textbook **HPSv3 reward-hacking signature**:
>   HSV saturation ×1.95, Sobel-edge p95 ×1.86, overexposed pixel ratio ×1.65
>   vs base SD3.5-M (see `docs/figures/reward_hacking_*.png`).
> - These artifacts **spill through DynamiCrafter**: pixel variance ×1.68
>   → RAFT-flow magnitude ×2.16, which inflates VBench's `dynamic_degree`
>   from 0.309 → 0.489 (**+0.180**) *without* improving perceived motion.
> - `subject_consistency` *decreases* (0.970 → 0.968), consistent with CLIP
>   embedding drift from saturation shift.
> - `motion_smoothness` is unchanged (0.368 → 0.368).
>
> **Revised story (likely the actual CVPR angle):** we present the first
> systematic measurement of **T2I reward-hacking carryover to I2V**, including
> a diagnostic protocol that distinguishes genuine quality transfer from
> metric-fooling artifacts. The old "7 of 9 dims p<0.05, 59% judge win-rate"
> framing below is **retained for revision reference only** and will be
> replaced by the full n=50 Wilcoxon table + Qwen2-VL judge once the overnight
> run completes (ETA 2026-10-30 04:40).

---

**Title (tentative)**
"Does Reward-Aligned Image Generation Transfer to Video? A Keyframe-Carryover Study"

**Alt titles**
- "The Keyframe Bottleneck: Measuring T2I Alignment Carryover to I2V Diffusion"
- "Transfer Without Training: An OPSD Case Study on Image-to-Video Diffusion"

---

## Abstract (150 words, target)

Reward-aligned post-training (OPSD, DPO-Diff, DiffusionNFT, …) has substantially advanced
text-to-image generation, but it remains unclear whether these image-level gains **transfer to
video**. We propose the cleanest possible test: because image-to-video diffusion models
hard-condition on a first frame, we can swap *only* the first-frame generator and ask whether
downstream video quality improves — with **no video training and no gradient updates**. On a
frozen DynamiCrafter-320, feeding first frames from OPSD-tuned SD3.5-M (using the authors'
public LoRA) rather than base SD3.5-M improves 7 of 9 VBench-I2V dimensions with p<0.05 and
achieves a 59% pairwise win-rate under a Qwen2-VL judge. A matched-aesthetic ablation shows the
gains are not merely "better pixels". A bonus Tier-2 extension — temporal-only LoRA on the
DynamiCrafter DiT, trained with OPSD — closes the remaining gaps and shows intermediate-noise
supervision matters more in video than in image. Code and results at
`github.com/jingxiduan/ml-portfolio/tree/main/08-video-opsd-i2v`.

---

## §1 Introduction (~0.9 page)

- Opening: image-space reward alignment is thriving; video alignment is nascent and expensive.
- Core question: **when and why do image-level reward gains transfer to video?**
- Core idea: I2V hard-conditions on the first frame → a frame-level alignment should propagate
  mechanically. Test it with **zero training**.
- Contributions:
  1. **Tier 1 — the keyframe-carryover protocol**: a training-free diagnostic that measures how
     a T2I alignment method's gains survive the I2V bottleneck. 50-line pipeline.
  2. **Empirical regularity**: on OPSD × SD3.5-M → DynamiCrafter, 7 of 9 VBench-I2V dims improve,
     with a structured sign pattern (static gains, motion may lose).
  3. **Matched-aesthetic ablation** proving the carryover is not reducible to "better pixels".
  4. **Tier 2 — temporal OPSD**: a minimal video-training extension showing that the
     intermediate-step-supervision gap between OPSD and ReFL/DPO is **larger in video**.
- Teaser figure: a 2×3 grid of (prompt, base-SD3.5→DC output, OPSD-SD3.5→DC output) with VBench
  Δ bars underneath.

## §2 Related Work (~0.7 page)

- **Image-space reward alignment**: ReFL, DRaFT, DPO-Diff, DiffusionNFT, FlowGRPO, OPSD.
- **Video reward alignment**: InstructVideo, VADER, T2V-Turbo, VideoDPO.
- **I2V diffusion**: SVD, DynamiCrafter, CogVideoX-I2V, AnimateAnyone.
- **Transfer / carryover studies in diffusion**: adapter-sharing (ControlNet variants), prompt
  transfer (prompt-to-prompt), cross-modality transfer (SD3 → video).
- **Evaluation**: VBench, VBench-I2V.
- Positioning: our Tier-1 protocol is new — the only prior "no-training transfer" study we are
  aware of is {cite competitor}, which does X; we differ by Y.

## §3 Method: Keyframe-Carryover Protocol (~1.2 pages)

- §3.1 Formal setup — $G^{T2I}$, $G^{I2V}$, $\mathcal{R}^V$, the carryover delta $\Delta(c)$.
- §3.2 Hypotheses H1 (positive carryover), H2 (selective), H3 (not just pixels).
- §3.3 Protocol pseudocode (half-page figure).
- §3.4 Statistical procedure — paired Wilcoxon with Holm-Bonferroni correction over 9 dims,
  3 seeds × 200 prompts, bootstrap CIs.
- §3.5 Tier 2 — temporal OPSD (compact, pointer to supp for full derivation; Tier 2 is a §4.6
  sub-experiment, not the main pitch).

## §4 Experiments (~2.8 pages)

### §4.1 Setup
- $G_0^{T2I}$: SD3.5-M (base), $G_\star^{T2I}$: SD3.5-M + `WeiChow/DiffusionOPSD/sd35-m-hpsv3`.
- $G^{I2V}$: DynamiCrafter-320, frozen. Supp: SVD-XT and CogVideoX-I2V-2B replicas.
- Prompts: VBench-I2V eval split (200 held out of 1118). Supp: Pexels-augmented long tail.
- Hardware: M3 Pro 36 GB (main), 1× A100 for ablations (supp).

### §4.2 Main result — Tier 1 carryover
- **Table 1**: 9×3 matrix (base, OPSD, Δ with p-values), with wins/losses summary.
- **Figure 3**: forest plot of Δ with 95% bootstrap CIs across the 9 dims.

### §4.3 LLM-judge pairwise
- **Table 2**: Qwen2-VL win-rates on 4 rubrics (motion, identity, prompt-align, overall) across
  200 random pairs.

### §4.4 Matched-aesthetic ablation (addresses H3)
- Filter base-SD3.5 frames to equal aesthetic score as OPSD frames.
- Report Δ on **video-level** dims only; show selective dims (prompt-adherence, i2v subject)
  retain significance even after matching.

### §4.5 Per-category breakdown
- VBench-I2V prompts split by content category (animals / scenes / objects / abstract).
- Show which categories benefit most — hypothesis: scenes gain most, abstract least.

### §4.6 Bonus: Tier 2 temporal OPSD vs ReFL/DPO
- 500-step LoRA runs on DynamiCrafter DiT, matched compute.
- Headline: OPSD-I2V > ReFL-I2V > DPO-I2V on 7/9 dims, confirming the
  intermediate-supervision-helps-more-in-video claim.

### §4.7 Why it (sometimes) fails
- Dimensions where OPSD hurts (dynamic degree, camera motion) analyzed via per-prompt tie-break.
- Hypothesis: OPSD biases toward well-composed static frames → DC animates less. Supported by
  qualitative examples in supp.

## §5 Analysis (~0.6 page)

- Why does the alignment transfer at all? Mechanistic argument: I2V temporal attention anchors
  frames 2..T to frame 1; a high-alignment frame 1 is a high-prior over the rest.
- Why does it fail on motion dims? Alignment and motion are antagonistic under the hard
  first-frame constraint. Tier 2 closes this gap.

## §6 Limitations (~0.3 page)

- Only one T2I alignment method tested at main-table scale. OPSD is representative but a true
  protocol-level claim needs 3+ methods (DPO-Diff, DiffusionNFT) — in supp.
- Only one I2V backbone at main-table scale. SVD-XT and CogVideoX in supp.
- VBench-I2V ↔ HPSv3 non-independence partially mitigated by RAFT-based motion dims and Qwen2-VL
  judge.
- No large-scale human study (N≤20 supp).

## §7 Conclusion (~0.2 page)

Reward-aligned image generation **does** transfer to image-to-video diffusion, selectively,
without any video training. We formalized the carryover protocol, measured it on OPSD × SD3.5-M
× DynamiCrafter, and showed the pattern of gains and losses is structured and interpretable.
A minimal temporal-LoRA extension closes the motion-dim gap. We hope the protocol lets future
T2I alignment papers audit their video-domain usefulness with 50 lines of code.

---

## Supplementary

- S1: full VBench-I2V 16-dim table.
- S2: SVD-XT and CogVideoX-I2V replica tables.
- S3: 2-min demo video (`supp/demo.mp4`).
- S4: Tier 2 full training curves, ablation (LoRA rank, $c_{\mathrm{adv}}$, EMA refresh).
- S5: 50-pair human study results.
- S6: implementation details, hyperparameters, exact prompt lists.
