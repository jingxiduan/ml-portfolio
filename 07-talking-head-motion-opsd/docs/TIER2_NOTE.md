# 07 Tier-2 · iterating on the reward design

> **Context.** Project 07 is this portfolio's extension of OPSD (originally
> [`worldbench/DiffusionOPSD`](https://github.com/worldbench/DiffusionOPSD),
> an image-diffusion RL method) **into the motion-diffusion space of a
> talking-head stack** (JoyVASA + LivePortrait). Tier-1 (the README's main
> body) established that OPSD transfers: train a motion-space liveliness
> reward, run OPSD, the head/eye motion stops being frozen. Tier-2 (this note)
> is the natural follow-up — now that the mechanism works, **what is the
> reward design actually doing, and what are the levers for improving it?**
>
> There is no "bug" to fix here. The Tier-1 reward is the one the README
> openly reports, with its caveats; this note is about the next lap of the
> same research direction, not a cleanup pass.

## 1 · What Tier-1 left on the table

The README's "Honest caveats" section calls out two overshooting signals in
the Tier-1 result (both are real, both come from running MediaPipe on the
rendered videos, 8 held-out clips):

- **Blink rate 55.4 /min**, well above the human 10-20 /min band
- **Head jitter 248** (vs base 118), i.e. the policy adds high-frequency pose
  motion to earn amp and couple at the same time

These are not accidents. They follow directly from how the Tier-1 reward
`motion_reward.py` was *designed*:

- The `blink` term was written as "**at least one blink per ~2 s window**"
  (smooth-max). Soft-max is unbounded above — a policy that produces many
  blinks keeps scoring better.
- The `jitter` term is a **bounded** negative that keeps shrinking as motion
  grows, so a policy that trades a little jitter for a lot of amp wins.

The Tier-1 note was deliberately kept honest: "a target rate (penalize *too
many* blinks) would fix it, not retrained here". Tier-2 is that retraining,
plus two cheaper levers that reach the same place.

## 2 · Three levers for the next lap

OPSD's reward is a *design choice*, and in a method extension like this one
the research question is: **at which level of the stack is the right place to
push on a signal?** Tier-2 answers that empirically — the same end-state
(blink ≈ 24 /min, lower jitter) can be reached from three independent levels,
and they compound.

### Lever 1 · Inference-time (`src/tier2_bon.py`, Track A)

Keep the Tier-1 LoRA. For each audio clip, generate N=16 motion samples with
different seeds, score each with a sharper **anti-overshoot** version of the
reward (gaussian centered on 18 /min for blink, hard quadratic penalty for
jitter above ~1° high-freq std), pick the best. **Zero training compute.**

### Lever 2 · Post-process (`blink_thin` in the same file, Track B)

The 73-d motion coefficients are an *interpretable, structured representation*
(21 keypoints × 3 coords plus pose + scale + translation), unlike a video
latent. That means a signal like "where are the blink events in time" is
directly editable. Track B peak-detects blinks on the BoN winner, greedily
keeps the strongest subject to a 2.5 s refractory period, and cosine-attenuates
the suppressed events on **non-lip** keypoints only — so lip-sync is preserved.
This is a lever that **only exists because the motion space is structured**;
there is no video-latent analog.

### Lever 3 · Training-time (`motion_reward_v2.py` + `opsd_train_motion_v2.py`, Track C)

Rewrite the two offending terms and retrain. Two surgical changes:

| term | Tier-1 | Tier-2 (reward v2) |
|---|---|---|
| `blink` | smooth-max over 2 s windows (unbounded above) | **Tier-1 + overshoot penalty** `−4·ReLU(mean(blink_sig) − 0.15)` |
| `jitter` | `−std_high / 0.5` (bounded) | **Tier-1 + hard cap** `−5·ReLU(std_high − 1°)` (quadratic) |

Everything else (`amp`, `couple`, `gaze`, `ood`, lip mask, OPSD mechanism,
LoRA scope, 600 rounds, batch 8) is unchanged — this is a pure reward swap.
Retrains in ~11 minutes on Mac M3 Pro.

**Each lever is at a different point in the deployment stack**, which is
useful on its own: a product shipping this would ship whichever lever is
cheapest for the constraint it has (no training compute → Lever 1; need to
edit existing clips → Lever 2; have training budget → Lever 3).

## 3 · Motion-space evidence (n=20, held-out)

20 held-out TTS clips (eval_00..eval_19), scored on the generated 73-d motion
coefficients with the eye probe from Tier-1 (held-out face r≈0.70). Averaged
over the 20 clips:

| variant | blink_rate | jitter_hp | naturalness v2 ↑ |
|---|---|---|---|
| base (JoyVASA, naive) | 0.0 | 13.6 | 0.99 |
| Tier-1 OPSD (naive) | **48.9** | 26.7 | 1.00 |
| Tier-1 + BoN (Lever 1) | 32.4 | 25.6 | 1.78 |
| Tier-1 + BoN + thin (Lever 1+2) | **23.9** | 25.6 | 2.61 |
| Reward-v2 OPSD (naive, Lever 3) | 41.1 | **23.7** | 1.18 |
| **Reward-v2 + BoN (Lever 3+1)** | **23.9** | **23.8** | **2.70** |

![blink rate across 7 variants](../outputs/tier2/fig_tier2_blink.png)

![3-panel summary: blink / jitter / total](../outputs/tier2/fig_tier2_summary.png)

**Two things to notice.**

1. **Lever 1+2 and Lever 3+1 land at the same blink rate (23.9 /min).** The
   two paths share zero code — one is a post-hoc filter, the other is a
   retrained model with a different reward. They converge because the
   anti-overshoot scorer and reward v2 encode the same prior (gaussian around
   ~18 /min); the equilibrium is where that prior's cost of being wrong
   balances against the amp/couple terms, and both paths find it.

2. **Lever 3 alone (naive, no BoN) already improves naturalness** (1.18 vs
   1.00 for Tier-1 naive) **and** gives the lowest `jitter_hp` across all
   variants (23.7). That is the reward v2's hard jitter cap doing what it was
   written to do.

## 4 · Training curves — where the design change bites

![v1 vs v2 training curves](../outputs/tier2/fig_v1_vs_v2_curve.png)

Right panel: the `blink` term diverges around step ~250. In Tier-1 the term
keeps climbing (soft-max has no ceiling, the policy keeps winning). In
reward v2 the term saturates once `mean(blink_sig) > 0.15`, because the
overshoot penalty kicks in. **That's the whole mechanism of Lever 3 in one
curve.**

Left panel: total reward climbs for both — the OPSD mechanism itself is
unchanged, it's happily optimizing whatever reward you hand it. The values
are not directly comparable (v1 and v2 have different term weights by design).

## 5 · What this says about OPSD as a method

This Tier-2 pass is less about numbers and more about a structural point:

- **The reward is where almost all the design work in OPSD lives.** Once the
  on-policy sampler + reward-guided target + detached distillation + EMA
  refresh mechanism is in place (Tier-1 verified that), the "model" of each
  variant is 100 % determined by its reward specification.
- **In a structured action space like 73-d motion coefficients**, you get a
  cheap second lever (post-process) that you don't get in a dense video latent.
  This is directly relevant to project 08 (image-to-video, dense latent), which
  only has Lever 1 and Lever 3 as options and had to work harder.
- **The naive Tier-1 numbers are honest, not broken.** Tier-1 published the
  overshoot in its caveats section specifically because it is the signal
  that drives Tier-2's reward design. The research progression is
  "mechanism works → reward design is now the lever → iterate on reward".

## 6 · Reproduce

```bash
# produce reward v2 + retrained LoRA (11 min on Mac M3 Pro)
python src/opsd_train_motion_v2.py --rounds 600

# N=16 BoN across base, Tier-1 OPSD, Reward-v2 OPSD for 20 held-out clips
# (writes outputs/tier2/selected_coefs.pt, 8 MB)
python src/tier2_bon.py --motion-only --save-coefs \
    --n-candidates 16 --lora-v2 outputs/opsd_v2/lora.pt

# figures from the summary + metrics jsonl
python src/make_tier2_figures.py
```

## 7 · Known limits of this Tier-2 note

- **Motion-space only, so far.** Blink/jitter numbers come from the eye probe
  and from the pose coefficients, not from the final rendered pixels. The
  single rendered clip on Mac (commit `0cffe4e`) matches these predictions
  (video blink 37.5 → 18.75), but n=1 is not n=20. The 140-video H20 render
  pass will produce the n=20 video-space table.
- **No human eval.** `naturalness_v2` is a sharper version of the training
  reward, not a human judgment. A small human-rated set (n=10-20 clips,
  pairwise preference) would calibrate whether 23.9 /min is really perceived
  as more human than 48.9 /min. Listed in the backlog.
- **The engine routing module (Part 2) has not been re-run with the Lever 3
  variant yet.** Adding `opsd_v2` as a new engine config and re-running
  `src/route_engines.py` would show whether LinUCB prefers it for any of the
  simulated user segments — an independent downstream signal.
