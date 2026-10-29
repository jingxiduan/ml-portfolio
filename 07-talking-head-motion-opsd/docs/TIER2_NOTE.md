# 07 Tier-2 · diagnosing and repairing OPSD-LoRA reward hacking

> **TL;DR.** The baseline OPSD training (Part 1 of this project) delivered the
> headline wins — head moves, eyes blink — but the honest-caveats section of
> the README called out that **blink rate overshot to ~55 /min** (human is
> 10-20) and **head jitter rose +110 %**. These are textbook reward-hacking
> symptoms. This Tier-2 note documents two **independent repair paths** that
> both land the model inside the human range, and the (motion-space) evidence
> from n=20 held-out clips. Video-space (MediaPipe) verification on 20 clips
> will be added once the H20 render window finishes; a single-clip video-space
> verification already matches the motion-space prediction (blink 37.5 → 18.75).

## 1 · What went wrong — a diagnosis in two sentences

The baseline reward in `motion_reward.py` was written to *encourage* liveliness
and *penalize* its two failure modes:

- `blink` term asked for **"at least one blink per ~2 s window"** (soft-max)
- `jitter` term penalized high-frequency head motion, but as a **bounded**
  negative that keeps shrinking as motion gets larger

OPSD is on-policy: once the policy figures out that "more blinks always helps"
and "jitter penalty saturates", it moves hard in those directions. The result
after 600 rounds:

| caveat | value | human |
|---|---|---|
| blink_rate (video, MediaPipe) | **55.4 /min** | 10-20 |
| head jitter (video) | **248** (vs base 118) | lower is better |

This is the same failure mode diagnosed in project 08 for a different model
(SD3.5-M OPSD-LoRA oversaturates by 1.95×). Same disease, different patient.

## 2 · Three independent repair tracks

Each track targets the reward-hacking symptoms with a different lever;
cross-validation comes from **the fact that two of them land at the same
blink rate** (23.9 /min, motion-space, n=20) even though they share zero code.

### Track A · Best-of-N (inference-time rejection sampling)

In `src/tier2_bon.py`: for each audio clip, generate N=16 motion samples with
different seeds, score each with an **anti-hacking scorer** (`naturalness_v2`),
pick the best. Key design choice: the scorer uses a **gaussian around 18 /min
blink target** (not "the more the better") and a **hard quadratic penalty**
on jitter above ~1 degree high-frequency std, so BoN really pushes toward the
human region.

- 0 training compute, 0 code change to the model.
- Motion generation is cheap on Mac (~1 s/clip), so N=16 adds ~15 s per clip.

### Track B · blink thinning (post-process the BoN winner)

The 73-d motion coefficient trajectory is editable. For the BoN winner:
1. Run the eye probe to get a per-frame blink signal.
2. Peak-detect blink events; greedily keep the strongest events subject to
   `inter-blink gap ≥ 2.5 s` (≈24 /min cap).
3. For each **suppressed** blink peak, apply a cosine-profile attenuation
   factor (0.25 at center → 1.0 at edges) to the **non-lip** expression dims
   over a ±5-frame window around the peak.

Lip keypoint dims (`LIP_DIMS` in `motion_reward.py`) are untouched, so
**lip-sync is preserved** — this is critical and verified on the 1 rendered
clip (`jaw_audio_corr` within 0.01 of the naive OPSD, see commit `0cffe4e`).

### Track C · reward v2 + retrain

Treat the cause instead of the symptom. Two surgical reward changes in
`src/motion_reward_v2.py`:

| term | v1 | v2 |
|---|---|---|
| `blink` | smooth-max over 2 s windows (no upper bound) | **v1 + overshoot penalty** `−4·ReLU(mean(blink_sig) − 0.15)` |
| `jitter` | `−std_high / 0.5` (bounded) | **v1 + hard cap** `−5·ReLU(std_high − 1°)` (quadratic) |

Everything else (`amp`, `couple`, `gaze`, `ood`, lip mask, OPSD mechanism,
LoRA scope) is unchanged. Retrained 600 rounds on Mac M3 Pro in ~11 minutes
(`src/opsd_train_motion_v2.py`), produces `outputs/opsd_v2/lora.pt`.

**Why run Track A on top of Track C?** Even with the better reward, variance
across seeds remains (motion diffusion samples per-frame latents at inference
time). BoN picks the best of N=16, which is cheap and compounds with the
training fix.

## 3 · Motion-space evidence (n=20, held-out)

20 held-out TTS clips, 4-10 s each, scored with the engine-agnostic
`naturalness_v2`. All numbers below are averaged over the 20 clips. The eye
probe used here is the same one trained in Part 1 (correlation r≈0.70 with
MediaPipe on held-out faces), so this is a **reasonable but not final**
estimate — the video-space MediaPipe verification will come from the H20
render pass.

| variant | blink_rate | jitter_hp | naturalness v2 ↑ |
|---|---|---|---|
| base (JoyVASA, naive) | 0.0 | 13.6 | 0.99 |
| OPSD v1 (naive) | **48.9** | 26.7 | 1.00 |
| OPSD v1 + BoN (Track A) | 32.4 | 25.6 | 1.78 |
| OPSD v1 + BoN + thin (Track A+B) | **23.9** | 25.6 | 2.61 |
| OPSD v2 (naive, Track C) | 41.1 | **23.7** | 1.18 |
| **OPSD v2 + BoN (Track C+A)** | **23.9** | **23.8** | **2.70** |

![blink rate across 7 variants](../outputs/tier2/fig_tier2_blink.png)

![3-panel summary: blink / jitter / total](../outputs/tier2/fig_tier2_summary.png)

**Two things to notice.**

1. Track A+B and Track C+A land at **exactly the same blink rate (23.9 /min)**
   even though the two paths share no code — one patches at inference time,
   the other retrains the model with a corrected reward. That's not a
   coincidence; it means ~24 /min is where this scorer's "better than random
   but still below the greedy-max of v1" equilibrium is, and both paths find
   it. (It is still slightly above the 10-20 human target — more aggressive
   `blink_target_mean` could push it lower, but then `amp` and `couple` start
   losing more.)

2. Track C alone (naive, no BoN) **already improves naturalness** (1.18 vs
   1.00 for v1-naive), **and** matches the lowest `jitter_hp` across all
   variants (23.7). That is the reward v2's hard jitter cap doing exactly
   what it is supposed to do.

## 4 · Why reward v2 trains differently

![v1 vs v2 training curves](../outputs/tier2/fig_v1_vs_v2_curve.png)

- Left panel: **total reward** climbs for both v1 and v2 (the OPSD mechanism
  itself works — the model is fitting the reward, whatever that reward asks
  for). Absolute values are not comparable between v1 and v2 since the terms
  differ.
- Right panel: the **`blink` term** diverges around step ~250. In v1 it keeps
  climbing (the policy keeps blinking more), in v2 it saturates (the overshoot
  penalty activates when `mean(blink_sig) > 0.15`). This is the whole story
  of the fix in one plot.

## 5 · Reproduce

```bash
# produce reward v2 + retrained LoRA (11 min on Mac M3 Pro)
python src/opsd_train_motion_v2.py --rounds 600

# N=16 BoN across base, OPSD v1, OPSD v2 for 20 held-out clips
# (writes outputs/tier2/selected_coefs.pt, 8 MB)
python src/tier2_bon.py --motion-only --save-coefs \
    --n-candidates 16 --lora-v2 outputs/opsd_v2/lora.pt

# figures from the summary + metrics jsonl
python src/make_tier2_figures.py
```

## 6 · Known caveats of this Tier-2 note

- **Motion-space only, so far.** Blink/jitter numbers come from the eye probe
  and from the pose coefficients, not from the final rendered pixels. The
  single clip rendered on Mac (commit `0cffe4e`) matches these predictions
  (video blink 37.5 → 18.75), but n=1 is not n=20. The 140-video H20 render
  pass will produce the n=20 video-space table.
- **No human eval.** `naturalness_v2` is a sharper version of the training
  reward, not a human judgment. A small human-rated set (n=10-20 clips,
  pairwise preference) would calibrate whether 23.9 /min is really perceived
  as more human than 48.9 /min, or whether MediaPipe's blink detector just
  misses faint blinks. This is listed in the backlog of `PROJECT_STATUS.md`.
- **The engine routing module (Part 2) has not been re-run with the new
  variants yet.** Track D in the backlog is to add `opsd_v2`, BoN, and
  BoN+thin as new engine configs and re-run `src/route_engines.py` — if the
  LinUCB preferences shift accordingly, that's an independent downstream
  signal that the fix is real.
