# OPSD-I2V: Method

> **2026-11-03 status.** Tier 1 is a diagnostic showing that image-reward
> optimization can propagate saturation, overexposure, and edge artifacts into
> I2V. Tier 2 implements temporal-LoRA reward post-training but has not improved
> held-out quality: the 500-step metrics are tied with base, and the V2 long run
> stopped at step 306 on its fixed loss-growth guard. The active question is now
> long-run stability, documented in `docs/RESULTS_LONG.md`.

The two tiers are connected failure analyses, not separate improvement claims.

---

# TIER 1 — Keyframe Carryover (no video training)

## 1.1 Setup and notation

- Let $G^{T2I}: \mathcal{P} \to \mathcal{X}$ be a text-to-image generator mapping prompts to RGB
  frames. We compare two instantiations:
  - $G_0^{T2I}$ = base SD3.5-M (pretrained flow-matching backbone).
  - $G_\star^{T2I}$ = OPSD-tuned SD3.5-M (base + public LoRA from
    `WeiChow/DiffusionOPSD` variant `sd35-m-hpsv3`).
- Let $G^{I2V}: \mathcal{X} \times \mathcal{P} \to \mathcal{V}$ be a frozen I2V model
  (DynamiCrafter-320 in our main runs).
- Let $\mathcal{R}^{V}: \mathcal{V} \to \mathbb{R}^9$ denote VBench-I2V's 9-dim evaluator.

The **carryover delta** for a prompt $c$ is

$$\Delta(c) = \mathcal{R}^{V}\!\big(G^{I2V}(G_\star^{T2I}(c), c)\big) - \mathcal{R}^{V}\!\big(G^{I2V}(G_0^{T2I}(c), c)\big) \in \mathbb{R}^9.$$

## 1.2 Hypothesis

**H1** (positive carryover): $\mathbb{E}_{c \sim \mathcal{P}}[\Delta(c)_d] > 0$ for the majority
of dimensions $d \in \{1, \dots, 9\}$ with $p < 0.05$ under a paired Wilcoxon signed-rank test.

**H2** (selective carryover): The sign and magnitude of $\mathbb{E}[\Delta(c)_d]$ vary across
$d$ in a structured way — static-oriented dims (subject/background/aesthetic/imaging) gain,
motion-oriented dims (dynamic degree, camera motion) may lose.

**H3** (not just better pixels): Even after matching base and OPSD frames on single-frame
aesthetic score (via LAION aesthetic predictor), the per-dim $\Delta$ pattern on *video*-level
dims (motion smoothness, i2v subject, i2v background) persists.

## 1.3 Why this is the right experiment

In I2V, the first frame is a **hard conditioning input** — it is literally pasted as the first
frame of the latent sequence and the model's temporal attention is anchored to it. Thus any
quality trait of the first frame has a mechanical downstream effect on frames 2..T. OPSD claims
to improve first-frame quality on seven image-level axes (HPSv2.1, CLIPScore, PickScore,
Aesthetic, ImageReward, HPSv3, DeQA). The experiment asks: **which of these axes mechanically
transmit through the I2V bottleneck, and which get filtered?** This is a well-posed empirical
question with a clean binary answer per dimension.

## 1.4 Pipeline (fully training-free)

```
for seed in {0, 1, 2}:
  for prompt c in eval_split (|eval_split| = 200):
    frame_base[c, seed] ← G_0^T2I(c; seed)
    frame_opsd[c, seed] ← G_*^T2I(c; seed)                      # same seed → paired
  for prompt c, seed:
    video_base[c, seed] ← G^I2V(frame_base[c, seed], c; seed)   # DC frozen
    video_opsd[c, seed] ← G^I2V(frame_opsd[c, seed], c; seed)
  for prompt c:
    R_base[c] ← VBench(video_base[c, 0..2])                     # per-prompt mean
    R_opsd[c] ← VBench(video_opsd[c, 0..2])
  for d in 1..9:
    Δ_d ← mean_c (R_opsd[c, d] − R_base[c, d])
    p_d ← Wilcoxon(R_opsd[:, d], R_base[:, d])
```

No gradient is ever taken; no checkpoint is ever updated.

## 1.5 Expected outcome and failure modes

- **Expected**: $\Delta$ positive on 5–8 of 9 dims, with aesthetic/imaging/subject-consist
  showing the strongest effects (these are the dims OPSD was explicitly tuned for).
- **Possible negative** on dynamic degree, because OPSD's T2I rewards favor well-composed static
  frames which the I2V model animates with less motion.
- **Null hypothesis realized** (no transfer) would itself be a useful negative result; in that
  case Tier 2 becomes the primary contribution.

---

# TIER 2 — Temporal OPSD (bonus, modest training)

Preserved from the original design. See the first README draft for the full derivation; summary
below.

## 2.1 OPSD recap (parent paper, T2I)

Behavior policy $v_{\mathrm{old}}$, trainable $v_\theta$ (LoRA). Four steps per update:

1. **On-policy query**: sample $(z_q, c, \sigma_q)$ from a rollout of $v_{\mathrm{old}}$ stopped
   at intermediate noise level $\sigma_q$.
2. **Clean anchor**: $y_0 = z_q - \sigma_q v_{\mathrm{old}}(z_q, c, \sigma_q)$, detached.
3. **Reward-guided bounded targets**: project $\nabla_{y_0} R(y_0, c)$ to a trust region of
   radius $c_{\mathrm{adv}} \sigma_q$, giving $y_0^\pm$ (both detached).
4. **Finite-step fit** ($K{=}1$): minimize $\|\hat y_0(\theta) - y_0^+\|^2 - \|\hat y_0(\theta) - y_0^-\|^2$.
5. **EMA refresh**: periodically $v_\theta \to v_{\mathrm{old}}$.

## 2.2 Three I2V adaptations

- **First-frame-conditioned anchor**: $x_0^{\mathrm{src}}$ enters the conditioning dict, is never
  sampled/updated. Subject-identity drift is bounded by base behavior.
- **Temporal-only LoRA**: injected on `attn_temporal.{to_q,k,v,out}` + motion-module linears
  only. ~2M params. Spatial priors frozen.
- **Clipped dynamic-degree reward**: counter-force against first-frame leakage; see
  `src/rewards/dynamic_degree.py`.

## 2.3 Core training algorithm (I2V)

```python
# Initialize
v_old = base_I2V.clone().freeze()
v_theta = inject_temporal_lora(base_I2V.clone(), rank=16, target="temporal_only")
ema = EMA(decay=0.995)

for step in range(num_steps):
    x0_src, c = sample_batch()
    sigma_q = sample_sigma()
    with torch.no_grad():
        z_q = rollout_until(v_old, x0_src, c, sigma_q)
        y0 = z_q - sigma_q * v_old(z_q, (c, x0_src), sigma_q)
    with torch.enable_grad():
        y0_leaf = y0.detach().requires_grad_(True)
        r = R(vae.decode(y0_leaf), c)
        g = torch.autograd.grad(r.sum(), y0_leaf)[0]
    g_norm = g / (g.flatten(1).norm(dim=1, keepdim=True) + 1e-8)
    y0_plus  = (y0 + c_adv * sigma_q * g_norm).detach()
    y0_minus = (y0 - c_adv * sigma_q * g_norm).detach()
    y0_hat = z_q - sigma_q * v_theta(z_q, (c, x0_src), sigma_q)
    loss = ((y0_hat - y0_plus)**2).mean() - ((y0_hat - y0_minus)**2).mean()
    loss.backward(); opt.step(); opt.zero_grad()
    if step % refresh_every == 0:
        ema.update(v_old, v_theta)
```

## 2.4 Long-run failure and stabilized fit

The historical contrastive quadratic fit is not lower-bounded as a function of the model prediction. The 2,000-step V2 attempt exhibited this failure: its rolling positive-target error grew by 10.95× and the guard stopped the run at step 306.

V3 keeps the same finite reward target but replaces indefinite negative-target repulsion with an explicit behavior anchor:

$$
\mathcal L_{\mathrm{V3}}
=\lVert\hat y_0-y_0^+\rVert_2^2
+\lambda_a\lVert\hat y_0-y_0\rVert_2^2.
$$

The negative target remains logged as a diagnostic. V3 also uses a smooth `tanh` map before pixel-naturalness computation, preventing the hard input clamp from zeroing derivatives outside the nominal display range.

These are hypotheses under test, not validated improvements. The fixed preflight, long-run gates, and current results are in [`RESULTS_LONG.md`](RESULTS_LONG.md).

---

# Contribution boundary

1. DiffusionOPSD, DynamiCrafter, CLIP, and RAFT are reused and credited.
2. This repository contributes an I2V conditioning adaptation, temporal-only LoRA path, differentiable video-reward integration, and auditable failure diagnosis.
3. The stabilized V3 objective is motivated by the observed long-run failure but is not considered successful until it completes the registered run and held-out evaluation.
4. No claim is made that the current method beats the frozen base, InstructVideo, DPO-Video, or a video foundation model.
