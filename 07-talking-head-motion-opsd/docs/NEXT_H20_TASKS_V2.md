# 07 · Next H20 task list **v2**, make trained visibly better than base

> Version v2, 2026-11-01. **This file replaces the task scope of `docs/NEXT_H20_TASKS.md`**,
> whose T1 to T3 were finished on 10-30 with results in `docs/RESULTS_NEXT.md`. Its section 0b
> rule of changing only the jitter cap is cancelled this round, see section 0.3.
>
> Written for the CodeBuddy of the next H20 session. Read `AGENTS.md`, `PROJECT_STATUS.md` and
> `docs/INDUSTRY_ROUND.md` first, then this file.

---

## 0 · The single problem this round solves

Not a metric, an impression:

> **Put the base video and the trained persona video side by side, and it must be obvious at
> first glance that the trained one looks more like a real person, and that the specific defect
> of base was clearly fixed.**

Current state, the three-industry round delivered on 2026-11-01: finance is only **1.31x** base
and real-estate **1.27x**, which reads as the same face shaking slightly more, and the trained
version actually looks **faker**.

### 0.1 Diagnosis: why no visible difference, confirmed, not a hypothesis

Of the six terms in the v4 reward, four tune the same **3 numbers**, pitch, yaw and roll:

| term | weight | what it actually optimizes | dimensionality |
|---|---|---|---|
| `amp` | 1.0 | gaussian peak of head low band amplitude | **3, pose** |
| `jitter` | 0.5 | head high band jitter toward 0 | **3, pose** |
| `couple` | 0.5 | global Pearson of loudness envelope vs head speed over a 4 s window | **1 scalar** |
| `blink` | 0.5 | smoothmax peak of the blink signal every 2 s minus an overshoot penalty | via eye_probe |
| `gaze` | 0.3 | `tanh(gaze_std/0.1)`, more drift is better | via eye_probe |
| `ood` | 1.0 | only clips values above `abs(exp) > 3` | no direction |

Among the 63 expression dimensions only blink and gaze are touched indirectly, and **no term
rewards brow, cheek, mouth corner or jaw micro motion following speech**, so expression stays
frozen on the source portrait, which `INDUSTRY_ROUND.md` section 5 states itself: smile comes
from the source portrait and is not in the reward.

So the **tunable space is one knob, how much the head shakes.** That is the entire reason
finance lands at 1.31x with no visible difference. v4 did the right thing by moving amplitude
into the target band, 0.84 to 0.90x, but after that **the remaining tunable space was zero by
construction**.

### 0.2 Three measured sources of fakeness, not impressions

1. **Blink rate overshoots by 2 to 5x.** Measured human is 15 per minute, band 7.5 to 37.5; base
   manages only 5.5; the trained version counts **35 to 77 per minute** from EAR valleys on the
   rendered video, acknowledged but unfixed in section 7.6 of `INDUSTRY_ROUND.md`. Base never
   blinks, rigid; the trained version blinks nervously. **The middle ground of 15 to 18 per
   minute is where neither one stands.**
   Root cause: the blink term rewards the **peak** within the window via smoothmax and only
   constrains the mean, so it cannot control **frequency**.
2. **Silence segments are unconstrained.** `couple` is a global Pearson over 4 s, and during
   silence `env` is about zero so the gradient vanishes, so the model keeps shaking at the same
   rhythm while listening. Real humans are **almost still with occasional small nods**. This is
   the largest single source of fakeness.
3. **Expression is constant and only the head moves.** The head moves while the face is dead; the
   shoulders and upper body are a still image with no coupling to the head, giving a pasted-on
   head.

**Base looks more natural because it moves so little, which hides these three defects.** Simply
adding amplitude amplifies the fakeness, which is exactly the v2 and v3 uncanny valley
mechanism; v4 sidestepped it by compressing amplitude, but **sidestepping is not solving**.

### 0.3 The plan this round

Replace the optimization target, three amplitude scalars, with a whole set of visible
behavioural events:

| new or rewritten | what it fixes | stage |
|---|---|---|
| **`expr`, expression dynamics** | bring the face alive, brow, cheek and jaw micro motion following speech | **P1, only this one first** |
| `listen`, silence behaviour | the model listens instead of shaking at constant speed | P2 |
| `phase`, phase lock | nod on stress instead of shaking evenly | P2 |
| `blink` rewritten as events | blink rate back to 15 to 18 per minute | P3 |
| `gaze` rewritten as fixation and saccade | the eyes stop drifting constantly | P3 |
| render side: half body and image quality | remove the pasted-on head, raise sharpness | P4 |
| many demos | at least 10 sets per industry, slow motion, eye crops | P5 |

**Ordering principle: add one group of terms at a time.** This is not a hard rule, it exists so
attribution still works: add only `expr` and train one version, confirm the face came alive, then
stack listen and phase, then touch blink and gaze.

---

## 0b · Hard constraints, only five this round

1. **Finish the full backup and push at least 60 minutes before the window ends**, section 0c.
2. **Delete nothing**, no `git push --force`, no `git reset --hard`. Only add files and
   directories.
3. Every training run goes through `tools/train_guard.py`; run `--selftest` first, it must print
   `selftest OK`.
4. Numbers must be verifiable; write FAIL when a gate is missed. **Do not change the success
   criteria, do not rewrite a failure as mitigated.**
5. When each task ends, normally, stopped by the guard, or stopped manually, **back up
   immediately before starting the next one**.

> The old task list rule in section 0b, change only the jitter cap and keep everything identical
> to v2, is **cancelled** this round. Attributability is now guaranteed by the batching strategy
> of adding one group at a time, section 0.3, instead of freezing the reward formula.

## 0c · Backup and push

| content | command | destination |
|---|---|---|
| jsonl logs, metrics, diagnostic json, figures, GIFs, LoRA, manifests, md | `tools/backup_push.sh small "<msg>" <paths...>` | current branch |
| mp4, audio subsets, anything above 90 MB | `tools/backup_push.sh big r3-07 <paths...>` | branch `backup/r3-07` |
| see how much is still unpushed | `tools/backup_push.sh status` | |

- **Stop line: no new tasks in the last 60 minutes of the window**, switch to a full backup:
  pack all mp4 and render output with `big` first, then push `main` with `small` as a catch-all.
- Total size of `backup/*` branches must stay at or below 1.5 GB, shared with 08. If it goes
  over, compress to 480p crf 28; if still over, keep only the eval set and the final output and
  state in the results document what was dropped.
- Push with the SSH setup already configured on the machine, and **never** write credentials into
  a file or a command.

---

## 1 · Stage overview and gates

| stage | content | estimate | gate, if unpassed stop and write the conclusion, do not continue |
|---|---|---|---|
| T0 | environment and reproduce current state | 0.5 h | base vs persona from the three industry LoRAs matches the delivered demo |
| **C1** | **calibration**: expression dynamics, silence, phase, blink events, gaze structure | 1.5 h | all five quantities measured with median and IQR, units aligned with the reward internals |
| **P1** | **add only `expr`**, train once per industry | 1.0 h | **the face came alive**: non-lip expression low band std at least 1.5x base, and ood does not worsen |
| **P1e** | render P1 demos and confirm by eye | 1.0 h | the side by side shows a difference and nothing became faker |
| P2 | stack `listen` plus `phase` | 1.0 h | silence segment head amplitude at most 0.6x the speech segment; onset hit rate above base |
| P3 | rewrite `blink` and `gaze` | 1.0 h | rendered video blinks 12 to 22 per minute |
| P4 | render side: half body and image quality | 1.5 h | output includes shoulders and no head-body seam is visible |
| P5 | many demos, figures, tables | 1.5 h | at least 10 sets per industry |
| P6 | write results, update README and STATUS, push | 0.7 h | |

Total about **9.5 h**. **C1 runs on CPU in parallel with T0 and P1 and does not occupy the GPU.**

---

## 2 · T0 · Environment

```bash
git clone git@github.com:jingxiduan/ml-portfolio.git && cd ml-portfolio/07-talking-head-motion-opsd
bash scripts/h20_render_setup.sh
source .venv/bin/activate
nvidia-smi                              # record current usage of both cards
python ../tools/train_guard.py --selftest   # must print selftest OK
```

Baseline to confirm the environment is not broken:

```bash
python scripts/eval_industries.py --n-clips 12 --out outputs/industry_eval_r3_baseline.json
```

This should match the table in section 3 of `docs/INDUSTRY_ROUND.md`: base 0.890 deg, finance
1.162 deg, realestate 1.134 deg, shortvideo 2.699 deg. If it differs a lot, fix the environment
first, do not continue.

---

## 3 · C1 · Calibration, the source of every target band this round, all measured

Reuse what worked in the industry round: **target bands are measured from real public video and
are never hand tuned.** Add `scripts/calibrate_behaviour_bands.py`, writing
`outputs/behaviour_bands.json`.

Input: `data/videos_public/hdtf/*` for finance and `data/videos_public/celebv/*` for realestate
and shortvideo, reusing the grouping of `outputs/industry_bands.json`, with the video's own voice
as audio.

### 3.1 Unit alignment, critical, do not walk into the v4 section 7.7 trap

- **pose**: MediaPipe plus solvePnP, same definition as v4, already validated.
- **expression: it must be measured in LivePortrait coefficient space.** Use the motion extractor
  of `LivePortraitWrapper` in `src/engine.py` to extract 63 dim exp per frame, then apply the
  **exact same** Butterworth bandpass used inside the reward, `src/motion_reward.py:_bandpass`,
  so the units line up directly. MediaPipe cannot produce LivePortrait keypoints and **must not**
  be used to set expression targets.

### 3.2 Quantities to measure

| quantity | definition | used by |
|---|---|---|
| `expr_low_std` | median over non-lip exp dims, excluding `LIP_DIMS`, of the 0.3 to 3 Hz std | `expr` term target |
| `expr_env_corr` | Pearson of env and the absolute d expr_low over dt | `expr` coupling weight |
| `silence_ratio` | silence low_std of pose divided by speech low_std of pose | `listen` term target |
| `silence_nod_hz` | power share of 1 to 3 Hz during silence | `listen` term target |
| `onset_hit` | share of speech onsets with a speed peak within 80 to 240 ms after onset | `phase` term target |
| `onset_lag_ms` | median latency of that speed peak | `phase` term window |
| `blink_rate`, `blink_dur_s`, `ibi_cv` | EAR valley event detection, rate, duration, IBI coefficient of variation | `blink` rewrite targets |
| `gaze_saccade_sparsity`, `fixation_dwell_s` | L1 over L2 ratio of gaze speed, share of low speed frames | `gaze` rewrite targets |

VAD: an energy method is enough, `env` against the clip's own quantiles with hysteresis, no new
dependency. If `silero-vad` is downloadable prefer it, and note that in `data/DATA_SOURCES.md`.

Output: `outputs/behaviour_bands.json`, median, IQR and n per industry per quantity.
**If something cannot be measured, write not measured honestly, do not substitute a hand tuned
value.** That is exactly the v4 section 7.7 lesson of translating across mismatched units anyway.

---

## 4 · P1 · Add only `expr`, bring the face alive

### 4.1 New file `src/motion_reward_v5.py`

Inherit `LivelinessRewardV4`, **add one `expr` term only, and copy the weights and formulas of
`amp / jitter / couple / blink / gaze / ood` verbatim from v4**. Start with weight
`expr = 1.0`, one shared value across the three industries for now.

```python
NON_LIP = [i for i in range(63) if i not in LIP_DIMS]
exp     = coef[..., NON_LIP]
low     = _bandpass(exp, 0.3, 3.0)                 # same filter as pose
std_low = (low.pow(2).mean(1) + 1e-6).sqrt()       # (B, |NON_LIP|)
dev     = (std_low.mean(-1) - self.expr_target) / self.expr_tol
expr    = torch.exp(-0.5 * dev * dev)              # peaked gaussian, same shape as amp
```

`expr_target` and `expr_tol` are read from `outputs/behaviour_bands.json`, the measured median per
industry; start `tol` at the v4 validated 1.0 and use `max(1.0, 0.5*(p75-p25))` if the industry
IQR is wider.

**`expr_active_frac` must be recorded**, the share of samples whose std_low falls inside
target plus or minus tol, and written into `metrics.jsonl`. If it stays at 0 or 1 throughout, the
target is wrong, which is the same class of failure as the v2 jitter cap.

### 4.2 New file `src/opsd_train_motion_v5.py`

A thin wrapper over `src/opsd_train_motion_v4.py`, swapping only the reward class and adding
`--expr-target / --expr-tol / --expr-weight`. **Keep the `lip_mask` logic**, do not move the lips,
so lip sync is preserved.

### 4.3 Training, once per industry, about 14 min each on H20

Reuse the validated amplitude parameters from `outputs/industry_bands.json` and the blink
parameters from after the half-closed-eye fix in the industry round:

| industry | `--amp-target-deg` | `--jitter-cap-deg` |
|---|---|---|
| finance | 1.3838 | 0.1883 |
| realestate | 1.2570 | 0.2358 |
| shortvideo | 2.9299 | 0.5000 |

Also fixed: `--amp-tol-deg 1.0 --blink-target-mean 0.12 --blink-overshoot-penalty 8.0`.

```bash
OUT=outputs/opsd_v5_finance
mkdir -p $OUT
nohup env CUDA_VISIBLE_DEVICES=0 python src/opsd_train_motion_v5.py \
    --out $OUT --data-dir data/audio_public/train --rounds 3000 --seed 0 \
    --amp-target-deg 1.3838 --amp-tol-deg 1.0 --jitter-cap-deg 0.1883 \
    --blink-target-mean 0.12 --blink-overshoot-penalty 8.0 \
    --expr-target <C1 measured> --expr-tol <C1 measured> > $OUT/train.log 2>&1 &
TRAIN_PID=$!
nohup python ../tools/train_guard.py --kind motion --log $OUT/metrics.jsonl \
    --pid $TRAIN_PID > $OUT/guard.log 2>&1 &
```

### 4.4 Gate, P1

Measure with `src/tier2_bon.py --motion-only` on the same eval set:

- **non-lip expression low band std at least 1.5x base**, base counts as 1.00x, this is the
  quantitative test that the face came alive;
- `ood` no worse than v4, must not fall below -0.05;
- `amp` still inside 0.8 to 1.1x of target, the head must not be crushed or inflated to grow
  expression;
- `expr_active_frac` between 0.2 and 0.8, showing the target has gradient.

**If any fails, stop**: write the numbers into `docs/RESULTS_R3.md`, report, do not enter P2.

---

## 5 · P1e · Confirm by eye, the most important step of this round

```bash
python scripts/render_industry_demo.py --manifest data/industry_demo_manifest_v3.json \
    --lora-dir outputs --out outputs/v5_demo --gpu 0 --tag-prefix v5_
python scripts/make_demo_gifs.py --video-dir outputs/v5_demo/finance \
    --out outputs/v5_gifs/finance --variants base finance
```

Render 3 base vs v5 pairs per industry, 1280 px with pasteback. Look at three things and write all
three into `docs/RESULTS_R3.md`:

1. **Did the face come alive**: do brow, cheek and jaw show micro motion following speech, while
   base is constant;
2. **Did it become faker**: is blinking more exaggerated, is the face warped or stretched;
3. **Is the sense of industry still there**: are the three personas still separable.

**If it came alive but became faker, stop at P1**: tune the dimension that turned faker, most
likely blink frequency or amplitude, on its own before entering P2.

---

## 6 · P2 · Stack `listen` plus `phase`

Add `src/motion_reward_v6.py`, a new file, do not modify v5, adding two more terms on top of v5:

- **`listen`**: use VAD to split the window into speech and silence, with one peaked target each,
  speech equals `amp_target` and silence equals `amp_target * silence_ratio`; plus one peaked term
  on the 1 to 3 Hz power share during silence, target `silence_nod_hz`.
  **Keep the original global `amp` untouched** for comparison.
- **`phase`**: take onsets from `env`, rising edge plus hysteresis, and reward the share of onsets
  with a local peak of absolute speed inside the window `onset_lag plus or minus 80 ms`, peaked at
  the measured `onset_hit`. **Keep the original `couple` untouched** for comparison.

Gate: silence segment head amplitude at most 0.6x the speech segment; onset hit rate significantly
above base, report the value with the base comparison.

---

## 7 · P3 · Rewrite `blink` and `gaze`

- **`blink`**: drop the smoothmax peak formulation and switch to **event detection**, upward
  crossing of 0.35 on `b` plus a 0.2 s refractory, giving rate, duration and IBI, each with its own
  peaked gaussian, targets from C1; **keep** the hard constraint `mean(b) at most 0.12`, the lesson
  of section 4 of `INDUSTRY_ROUND.md` where a high target closes the eyes halfway.
- **`gaze`**: replace `tanh(g_std/0.1)` with two peaked terms, L1 over L2 sparsity of gaze speed
  and fixation dwell share; keep the original mean offset penalty.

Gate: **12 to 22 EAR events per minute** on the rendered video, human is 10 to 20 with a measured
median of 15. Reuse `scripts/measure_eye_openness.py` for detection, but **validate the counting
method on a small sample first**, since section 7.6 of `INDUSTRY_ROUND.md` already showed the old
counting method is sensitive to render noise and untrustworthy.
**If it is untrustworthy, change the method, or write honestly that the gate was missed. Do not
build a conclusion on an untrustworthy number.**

---

## 8 · P4 · Render side, no training, immediate perceptual gain

Add `scripts/render_torso.py`, or a flag on `render_industry_demo.py`:

1. **Half body**: during pasteback apply a light affine warp to the body region driven by head
   pose, shoulders at 0.3 to 0.5x of roll and pitch amplitude plus a 0.2 Hz breathing rise and
   fall, and prefer **half body portraits that include the shoulders**. Goal: head and body no
   longer disconnect.
2. **Image quality**: restore the animated face with CodeFormer or GFPGAN **before** pasteback;
   apply Real-ESRGAN 2x **after** pasteback; encode at crf 18. Super-resolve the eye region once
   more on its own.
3. Keep a baseline video with each step switched off, so it is clear which step actually helps.

Gate: the output includes shoulders and no head-body seam is visible side by side.

---

## 9 · P5 · Many demos

- At least **10 sets per industry**, different faces and different voice timbres:
  `python scripts/pick_portraits.py --scan 400 --per-industry 10 --min-ear 0.30 --min-px 800`
- Keep the same-face cross-industry four-up comparison, the most convincing one
- Add two comparison views: **0.25x slow motion**, amplitude and timing differences only show at
  slow speed, and **eye region crops**
- An overview contact sheet plus one side by side figure per industry
- mp4 goes to `backup/r3-07`; `main` keeps only GIF, json, md and figures

---

## 10 · P6 · Results

Write everything into `docs/RESULTS_R3.md`: the key numbers of each gate, PASS or FAIL, and the
base versus trained comparison table. Update the 07 section of `README.md`, `STATUS.md` and
`/PROJECT_STATUS.md` together. Wording must match the results, if there is no improvement say so.
Run the forbidden-word grep from `AGENTS.md` before pushing.

---

## 11 · Time budget and stop line

| stage | time |
|---|---|
| T0 | 0.5 h |
| C1, CPU, parallel | 1.5 h |
| P1 | 1.0 h |
| P1e | 1.0 h |
| P2 | 1.0 h |
| P3 | 1.0 h |
| P4 | 1.5 h |
| P5 | 1.5 h |
| P6 | 0.7 h |
| **total** | **about 9.5 h** |

Cut order when time runs short, from the back: set count in P5, super resolution in P4, P3, P2.
**Do not cut C1 or P1 and P1e**, C1 is the source of every later target band and P1 plus P1e are
the core judgement of this round.

**Stop all new tasks 60 minutes before the window ends and switch to a full backup, section 0c.**
