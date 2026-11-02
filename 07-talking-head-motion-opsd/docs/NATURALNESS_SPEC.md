# 07 · Acceptance criteria for a natural, lively talking-head video, NATURALNESS SPEC

> Created 2026-11-01
> **Status of this document: above every training task list.** Training before the parameters
> are settled is just adding compute in the wrong direction. Every later training target must be
> taken from here, and the finished run is accepted against the table here.

---

## 0 · Why this document exists

The delivered three-industry round is numerically fine, all three personas land inside the
measured motion band at 0.84 to 0.90x of target, but side by side it looks **no different from
base** and even faker. Measured evidence:

| | finance persona vs base | short-video persona vs base |
|---|---|---|
| head speed | **1.13x** | 1.89x |
| head jitter | **1.29x, shakier** | 1.70x, shakier |
| expression, brow, cheek, jaw | **zero change across three frames** | **zero change across three frames** |

The problem is not the number of training rounds. The problem is that **good was never
defined**. Until now only head amplitude inside some band was defined, and head amplitude happens
to be the single quantity **weakly related to naturalness**.

---

## 1 · Three hard principles

1. **Every criterion must be measurable**: a tool, a unit, a real human number. Phrases like
   looks natural are not allowed.
2. **Real human targets must be measured, never hand tuned.** The only permitted source is
   public real video, and **the units must match the ones inside the reward**, the lesson of v4
   section 7.7 where mismatched units were translated anyway and trained the eyes half shut.
3. **Some acceptance items must sit outside the reward.** Right now blink and jitter are both
   training targets and evaluation metrics, which is circular. The column **in reward?** below
   marks this, and only the independent items are trustworthy evidence.

**Two measurement spaces**, which must be cross-calibrated:

| space | content | used for |
|---|---|---|
| coefficient space | LivePortrait 73 dims, expression 0 to 62, scale 63, translation 64 to 66, pose 67 to 69 | reward internals and training targets |
| video space | MediaPipe blendshapes, solvePnP pose, EAR | **what the human eye actually sees**, used for acceptance |

---

## 2 · Evaluation table

Marks: `measured` means the number comes from an artifact already on disk; `to measure` means
C1 must measure it, and **no number may be filled in and no training may start before it is
measured**.

### A · Expression dynamics, against the mask face, currently completely missing, highest priority

| # | metric | how it is measured | real human target | base | v4 persona | in reward? |
|---|---|---|---|---|---|---|
| A1 | **non-lip low band expression amplitude** | coefficient space, non-lip expression dims excluding `LIP_DIMS`, 0.3 to 3 Hz std | to measure | about constant | about constant | no, added in v5 |
| A2 | blink rate | video space EAR valley events | **15 per min, band 10 to 20**, measured | 0 to 5.5, measured | 35 to 77, measured, too high | **yes** |
| A3 | blink duration | same | 100 to 400 ms, to measure | n/a | n/a | no |
| A4 | blink interval burstiness, CV of IBI | same | to measure | n/a | n/a | no |
| A5 | brow movement event rate | MediaPipe `browDown/InnerUp` blendshape threshold crossings | to measure | n/a | n/a | no |
| A6 | gaze structure, fixation dwell plus saccade rate | video space gaze speed | to measure | n/a | n/a | **yes** |
| A7 | eye openness EAR | MediaPipe | **at or above 0.30**, measured | 0.343, measured | 0.348, measured | no |
| A8 | expression-speech coupling | Pearson of envelope and absolute d expr_low over dt | to measure | n/a | n/a | no |

### B · Temporal structure, against constant-speed shaking

| # | metric | how it is measured | real human target | base | v4 persona | in reward? |
|---|---|---|---|---|---|---|
| B1 | speech onset phase lock hit rate | share of onsets with a speed peak within 80 to 250 ms after onset | to measure | n/a | n/a | no |
| B2 | silence segment amplitude ratio | silence low_std divided by speech low_std | to measure | n/a | n/a | no |
| B3 | still_ratio, share of near static frames | `video_metrics.py` | to measure | 0.041, measured | 0.018, measured | no |
| B4 | breathing floor, power share at 0.2 to 0.3 Hz | coefficient space pose | to measure | n/a | n/a | no |

### C · Head motion quality, the only class already measured

| # | metric | how it is measured | real human target | base | v4 persona | in reward? |
|---|---|---|---|---|---|---|
| C1 | low band amplitude, 0.3 to 3 Hz std | degrees | professional broadcast **1.878 deg**, measured; finance 1.384, realestate 1.257, shortvideo 3.039, measured | 0.88 to 1.04 deg, 0.47 to 0.55x, measured | 0.84 to 0.90x of target, measured | **yes** |
| C2 | high band jitter, 6 to 12.5 Hz std | degrees | professional broadcast **0.107 deg**, measured; 0.094, 0.118, 0.349, measured | 0.119, measured | 0.112 to 0.156, measured | **yes** |
| C3 | **smoothness = jitter divided by amplitude** | dimensionless | to measure, derived from the two rows above | n/a | n/a | no |
| C4 | speech_head_corr | Pearson of loudness and head speed | human **0.2 to 0.4**, measured | 0.13 to 0.16, measured | finance about 0, shortvideo 0.51, measured | **yes** |

### D · Face-body consistency and image quality, render side, not trained

| # | metric | how it is measured | target |
|---|---|---|---|
| D1 | shoulders and torso following head motion | correlation of shoulder line displacement and head roll | to be decided; qualitative requirement: no visible head-body disconnect |
| D2 | resolution and no seams | visual check plus pasteback edge inspection | 1280 px, no visible seam |
| D3 | temporal stability, no flicker or warp | inter-frame optical flow residual | to be decided |

### E · Final acceptance: human A/B, the only criterion that cannot be bypassed

| # | metric | method | threshold |
|---|---|---|---|
| E1 | blind win rate | at least 30 clips, base vs trained randomly paired and randomly left or right, ask which one looks more like a real person; only ids are shown, the key is stored separately | trained win rate **significantly above 50 percent**, binomial test |
| E2 | clear improvement judgement | same people then asked how large the difference is, none, slight, obvious | share answering obvious at or above 50 percent |

> E1 and E2 are the real success criteria of this round. Everything in A to D is a proxy, and
> **hitting every proxy while the human eye sees no difference is a failure**.

---

## 3 · Veto conditions, any one of these means FAIL no matter how good the rest is

| condition | reason |
|---|---|
| non-lip expression amplitude **below 50 percent of the real human median** | **mask face**, the first problem this round has to solve |
| blink rate **below 8 or above 25 per minute** | no blinking is rigid, too much is nervous, current value is 35 to 77 |
| head amplitude **above 1.5x the real human** | uncanny valley, that is exactly how v2 and v3 reached 2.4 to 2.7x |
| jitter divided by amplitude **above 1.5x the real human** | reads as shaking |
| EAR **below 0.30** | half closed eyes, the bug of the first industry round version |
| base and trained **indistinguishable** on E1 | training was pointless |

---

## 4 · Execution order, the order must not change

```
1. measure real humans, C1   ->  fill every "to measure" cell in tables A, B, C
2. measure base              ->  use the same table to find exactly which items base fails
3. set training targets      ->  peaked targets only for the items base fails, peaked is what stops overshoot
4. train                     ->  add one group of items at a time, leave everything else untouched
5. render the comparison     ->  same face, same voice, same seed, base vs trained
6. accept against this table ->  veto conditions first, then A to D, then human A/B, E1 and E2
7. only if 6 fully passes may the number of demos grow
```

**No training starts before step 1 finishes.** There is not a single real human number for the
expression dimensions yet, so any expression target would be a guess.

---

## 5 · Confirmed base defects, partial result of step 2, for orientation

| item | base measured | real human | verdict |
|---|---|---|---|
| head amplitude | 0.88 to 1.04 deg | 1.878 deg | **0.47 to 0.55x, too rigid** |
| blink | 0 to 5.5 per minute | 15 | **too few, rigid** |
| non-lip expression dynamics | about constant | to measure | **mask face** |
| speech_head_corr | 0.13 to 0.16, finance about 0 | 0.2 to 0.4 | weak |

**Key judgement: base looks reasonably natural because it moves so little, which hides the mask
face.** So the fix **cannot be more head amplitude**, that only exposes the mask face more
clearly, which is the v2 and v3 uncanny valley mechanism that v4 sidestepped by compressing
amplitude, and sidestepping is not solving.
The real order of repair is: **A, expression, then B, temporal, then C, amplitude**.

---

## 6 · Honest limitations of this specification

1. Most metrics in groups A and B are still to measure: expression dynamics and temporal
   structure have **never been measured** in this project. Until they are, only group C plus
   A2 and A7 are hard numbers in this specification.
2. Industry labels are a proxy split, public data has no industry annotation, consistent with
   section 7.1 of `INDUSTRY_ROUND.md`.
3. Lip sync via SyncNet has **never been measured**, and `jaw_audio_corr` cannot substitute for it.
4. Human A/B, E1 and E2, has no scores yet. Before this round, every claim of more natural was
   only a proxy metric.
