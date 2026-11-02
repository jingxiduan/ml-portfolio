# 07 · Next H20 task list: single-point jitter tuning, more data, held-out evaluation

> Written for the CodeBuddy of the next H20 session. Read `/AGENTS.md` and `/PROJECT_STATUS.md`
> first, then this file.
> This file only defines what to do, how success and failure are judged, and when to stop. Do
> not move on while a gate is unpassed.

---

## 0 · Background, only the facts relevant to the task

n=20 video space results, `outputs/tier2/bon_metrics.md`:

| metric | base | OPSD v1 | v1+BoN+thin | v2+BoN | human range |
|---|---|---|---|---|---|
| blink per min | 0 | 50.5 | 24.8 | 25.6 | 10 to 20 |
| head_jitter | 120 | 262 | 265 | 257 | about 120 |
| jaw_audio_corr | -0.14 | -0.09 | -0.09 | -0.10 | 0.3 to 0.5 |

- Blink: BoN plus thinning pushes it to about 25, still outside the human range but the
  direction is right.
- **head_jitter: it doubles after OPSD and nothing moved it. This is the single point this
  round has to solve.**
- jaw_audio_corr: base itself is negative and the 6 variants are indistinguishable, so it can
  **no longer be used as lip-sync evidence**.
- Training data is only 25 macOS TTS audio clips, `data/audio/train`, with 20 for eval.

### Hypothesis for why jitter did not drop, to be verified by T1, not a conclusion

The `jitter` term in `outputs/opsd/metrics.jsonl` and `outputs/opsd_v2/metrics.jsonl`
**gets worse** during training:

| run | step 1 | step 300 | step 600 |
|---|---|---|---|
| v1 | -0.312 | -0.389 | -0.458 |
| v2, with hard cap | -0.312 | -0.394 | -0.445 |

Inverting `jitter_bounded = -(weighted mean of std_high)/0.5` from `motion_reward_v2.py` gives
std_high rising from about 0.16 deg to about 0.22 deg. The hard cap threshold is
`jitter_cap_deg = 1.0 deg`, and **the actual value is far below the threshold, so the hard cap
was never triggered**, the relu is always zero and there is no gradient. The v2 jitter change
therefore did nothing, which matches the v1 and v2 curves lying almost on top of each other.

This is the same class of problem as the V1 `ReLU+clamp` gradient dead zone in 08: the penalty
is written but lands in a region where it does nothing.

Second possibility: jitter in the reward is the std of the 6 to 12.5 Hz band of the three pose
dimensions, in degrees, while video space `head_jitter`, MediaPipe, measures something else,
and the two need not move together.

---

## 0b · Hard constraints, read this first, it outranks every later section

### A. The failure mode specific to 07

After tightening the jitter penalty, the policy can score perfectly by simply not moving the
head. This is a more realistic risk than NaN: the amplitude term fights back, but if the jitter
threshold is set too low the head gets crushed. In existing logs the lowest 50 step window mean
of `amp` is 0.307, rising to 0.85 afterwards, and the guard script uses 0.15 as the freeze line.

### B. Prohibitions

1. **Do not modify existing files**: `src/motion_reward.py`, `src/motion_reward_v2.py`,
   `src/opsd_train_motion*.py`, `outputs/opsd*`, `outputs/tier2/*`, `data/audio/eval/*`. Only add
   new files, `motion_reward_v3.py` and similar.
2. **Only change the jitter cap threshold.** Weights and formulas for
   `amp / couple / blink / gaze / ood` must stay identical to v2, otherwise nothing is
   attributable.
3. **Eval and training speakers must not overlap**; split by speaker before sampling when
   scaling up data.
4. Do not change the success criteria, do not change the guard script thresholds.
5. Do not move to the next stage while a gate is unpassed.
6. No `git push --force`, no `git reset --hard`, do not delete existing `outputs/`.
7. Push and backup rules are in section 0c. **Back up immediately when each training, rendering
   or data prep task ends**, do not batch it to the end.

### C. Parameter caps

| item | cap or fixed value |
|---|---|
| `jitter_cap_deg` | the base p90 measured by T1, **restricted to the range 0.1 to 0.5 deg**; below 0.1 crushes the head |
| `jitter_cap_penalty` | at most 5.0, same as v2 |
| `lr` | 3e-4, unchanged |
| `rounds` | 600 for the 25 clip set; at most 3000 for the enlarged set |
| LoRA | r=8, alpha=8, batch 8, seed 0 |
| retries | at most 1, loosening `jitter_cap_deg` by 1.5x; stop if it trips again |

### D. Training must be launched through the guard script

Self test first: `python ../tools/train_guard.py --selftest`, it must print `selftest OK`.

```bash
OUT=outputs/opsd_v3
mkdir -p $OUT
nohup python src/opsd_train_motion_v3.py --out $OUT > $OUT/train.log 2>&1 &
TRAIN_PID=$!
nohup python ../tools/train_guard.py --kind motion --log $OUT/metrics.jsonl --pid $TRAIN_PID > $OUT/guard.log 2>&1 &
```

These thresholds were replayed over the v1 and v2 600 step logs with **no false positives**, and
they trip correctly on synthetic NaN, loss explosion and head freeze scenarios.

| rule | trigger, 50 step window mean | basis |
|---|---|---|
| `nonfinite` | any metric becomes NaN or inf | |
| `loss_blowup` | `loss` above 3x the mean of the previous 50 steps | measured range 0.0089 to 0.0118, very stable |
| `ood_blowup` | `ood` below -0.05 | worst measured -0.0046 |
| `reward_collapse` | `reward` more than 0.3 below the best window so far | v1 and v2 both rise monotonically |
| `jitter_blowup` | `jitter` term more than 0.5 below the previous 50 steps | v1 and v2 only fell about 0.15 |
| `head_frozen` | `amp` below 0.15 | lowest measured 0.307 |
| `stalled` | no new log for 1200 seconds | |

### E. What to do when it trips

1. Do not continue manually, do not restart as is.
2. Keep `outputs/<run>/` and `GUARD_STOP.json`, **immediately** back up per section 0c, then
   handle the rest.
3. One retry is allowed, loosening the cap per section C; stop if it trips again.
4. Record in `docs/RESULTS_NEXT.md`: which rule tripped, the step number, the log values.

### F. No effect is also a failure

The point of v3 is to make jitter drop. At step 300, if the `jitter` term shows **no
improvement** relative to the first 50 steps, meaning the window mean has not risen, the change
did not take effect. **Stop, do not run all 600 rounds**, write the diagnosis and report.

### G. Gate log

Append one line per gate to `docs/GATE_LOG.md`: time, gate, key values, PASS or FAIL.

---

## 0c · Pushing, backup and data sources

### Pushing and backup: the machine is reclaimed at a deadline, anything unpushed is lost

**Rule: back up immediately when each task ends, before starting the next one.** Ending
includes finishing normally, being stopped by the guard, and being stopped manually. Use
`tools/backup_push.sh`, already tested on a scratch repo: small files go to main, large files
and media are packed onto a separate branch and can be restored.

| content | command | destination |
|---|---|---|
| jsonl logs, metrics, diagnostic json, curve figures, GIFs, small LoRA, manifests | `tools/backup_push.sh small "<msg>" outputs/<run> data/audio_manifest_large.json ...` | `main` |
| rendered mp4, extracted audio subsets, anything above 90 MB | `tools/backup_push.sh big r2-07 outputs/tier2_r2 data/audio_public` | separate branch `backup/r2-07` |
| see how much is still unpushed | `tools/backup_push.sh status` | |

- Deadline = window start plus window length, use the user's value if given, otherwise 14 h. Do
  not start new tasks in the last 90 minutes, do one full backup instead.
- At T0, run `big` once on a very small directory to measure upload speed, then estimate how
  long a full backup takes.
- Total size of `backup/*` branches must stay under 1.5 GB, shared with 08. If it goes over,
  compress video first, 480p crf 28; if still over, keep only the eval set and the final output
  and state in the results document what was dropped.
- Push using the method already configured on this machine, SSH. Never write credentials into a
  file or a command. Never use `--force`.

### Data sources: must be public datasets

The current 25 training and 20 eval clips are all synthesized by macOS `say`, not real human
voices, and the speaking style is uniform. This round switches to a public Mandarin speech set:

- **Preferred AISHELL-1**, OpenSLR #33, Apache 2.0, about 178 hours and 400 speakers, with its
  own train, dev and test speaker split. **Do not download everything**, only what is needed:
  about 60 speakers from train and about 15 from test, enough to assemble the required clip
  count. Run `df -h` before downloading.
- **Fallback AISHELL-3**, OpenSLR #93, multi-speaker, Apache 2.0, or another public Mandarin
  corpus with a clear licence.
- Must produce `data/DATA_SOURCES.md`: name, URL, licence, download date, which speakers were
  taken, concatenation method, clip count, duration. The speaker id list and the concatenation
  script go to `main`; the audio subset goes through `big` backup, **audio does not go to `main`**.
- At T0 verify network access: `curl -sI https://www.openslr.org | head -1`. **If it is blocked,
  report to the user, do not fall back to TTS and call it public data.**
- Data prep runs on CPU, in parallel with T1 and T2, and does not occupy the GPU.

---

## 1 · Task overview and gates

| stage | content | estimated time | pass condition, otherwise stop and write the conclusion |
|---|---|---|---|
| T0 | environment | 20 min | `h20_render_setup.sh` smoke passes |
| T1 | diagnose jitter | 40 min | a clear conclusion: is the cap ineffective, are reward and video metrics correlated |
| T2 | reward v3, jitter only, retrain on the 25 clips | 20 min | motion space jitter no higher than 1.3x base |
| T3 | scale data to about 2000 clips, train v2 and v3 once each | about 2 h | at equal data size v3 jitter beats v2 |
| T4 | held-out evaluation: render plus SyncNet plus blind sheet | about 2 h | see section 6 |
| T5 | write results, push | 30 min | |

**If T2 does not pass its gate, do not do T3.** Write the diagnosis and the reason for failure
into the results document, that is itself a valid deliverable.

---

## 2 · T0 · Environment

```bash
git clone git@github.com:jingxiduan/ml-portfolio.git && cd ml-portfolio/07-talking-head-motion-opsd
bash scripts/h20_render_setup.sh
source .venv/bin/activate
nvidia-smi   # record current usage of both cards
```

A long training for 08 also runs on this machine, see
`08-video-opsd-i2v/docs/NEXT_H20_TASKS.md`. The 07 motion model is small and can share a card
with 08; select with `CUDA_VISIBLE_DEVICES` after checking free memory in `nvidia-smi`.
Rendering with LivePortrait waits until T4.

---

## 3 · T1 · Diagnosis, new file `src/diag_jitter.py`

Input: `outputs/opsd/lora.pt`, `outputs/opsd_v2/lora.pt` and base, generating motion
coefficients on the 20 clips of `data/audio/eval` each, reusing the existing 20 clip by variant
set in `outputs/tier2/selected_coefs.pt` instead of resampling.

Answer three questions, write results to `outputs/diag/jitter_diag.json` and
`docs/RESULTS_NEXT.md`:

1. **Is the cap ineffective**: compute `std_high` from the reward, pose three axes at 6 to
   12.5 Hz, for base, v1 and v2, reporting mean, p90 and max. If p90 is far below 1.0 deg the
   hypothesis holds.
2. **Gradient share of each reward term**: backprop through a batch of coefficients and record
   the L2 norm of the gradient on the coefficients from
   `amp / jitter / couple / blink / gaze / ood`. Judge whether jitter is drowned out by amp and
   blink.
3. **Do reward and video metrics move together**: over the 120 points of 20 clips by 6 variants,
   compute the Spearman correlation between reward side `std_high` and video side `head_jitter`
   from `outputs/tier2/bon_metrics.json`. Also confirm the definition and unit of `head_jitter`
   in `src/video_metrics.py` and write it into the results document.

Decision:
- Question 1 holds, then T2 uses a threshold calibrated on the base distribution.
- Question 3 correlation is low, absolute rho below 0.5, then the T2 jitter term must be changed
  to a quantity with the same definition as the video side, for example the second difference of
  pose, and the document must explain why.
- Neither holds, then stop, write the observation clearly, do not force the pipeline forward.

---

## 4 · T2 · reward v3, jitter is the only change

Add `src/motion_reward_v3.py` and `src/opsd_train_motion_v3.py`, written like v2, a thin wrapper
reusing `add_lora / build_windows / guided_x0 / rollout_to / set_mode` from
`opsd_train_motion.py`.

**Change only the jitter term, keep `amp / couple / blink / gaze / ood` exactly as in v2**, so
the difference is attributable.

- Change `jitter_cap_deg` from 1.0 deg to near the base distribution p90 found by T1, expected
  around 0.2 deg, use the T1 measurement.
- Make sure the cap has gradient early in training, record the cap activation ratio in metrics.
- Other hyperparameters identical to v2: 600 rounds, lora r 8, lr 3e-4, seed 0.

Output: `outputs/opsd_v3/lora.pt`, `outputs/opsd_v3/metrics.jsonl`, additionally recording
`cap_active_frac`.

Gate: evaluate with `src/tier2_bon.py --motion-only` on the 20 eval clips; motion space
`jitter_hp` must not exceed 1.3x base, base about 13.6 and v1 and v2 about 24 to 27, and blink
must not be worse than v2. If it fails, go back to T1, do not enter T3.

---

## 5 · T3 · Scale up data, only after T2 passes its gate, data prep itself can run earlier on CPU

Reason: training currently uses only 25 macOS TTS clips, so conclusions generalize poorly. OPSD
training needs audio only and motion space training is cheap, so scaling up is low cost. Data
source rules are in section 0c, **it must be a public dataset, TTS must not be passed off as
one**.

- Target: about 2000 training segments and 100 eval segments, 8 to 15 seconds each, 16 kHz mono.
- AISHELL-1 averages about 5 seconds per sentence while the training window is 4 seconds with a
  1 second stride. Method: concatenate adjacent sentences of the same speaker in index order
  past 8 seconds, leave 0.3 seconds of silence at the join, never concatenate across speakers.
- Speaker split uses the dataset's own train and test speaker lists, **no eval speaker appears
  in training at all**.
- Write the manifest to `data/audio_manifest_large.json`, speaker, original filename,
  concatenation order, duration, and to `data/DATA_SOURCES.md`.
- Keep the old 20 clip TTS set as a `legacy eval`, and evaluate on both old and new.

Controlled experiment, to prove the gain comes from the change and not from data volume:

| group | reward | training data | rounds |
|---|---|---|---|
| A | v2 | public data, about 2000 segments | 3000 |
| B | v3 | public data, about 2000 segments | 3000 |

The existing v2 and v3 results on 25 clips serve as the small scale reference. All three groups
fix seed 0; add seed 1 and 2 for B if there is budget. The 3000 round cap stays, **do not raise
it just because there is more data**.

---

## 6 · T4 · Held-out evaluation, stop using the reward's own terms as the conclusion

Right now blink and jitter are both training rewards and evaluation metrics, which is circular.
This round must add independent quantities.

1. **SyncNet LSE-C and LSE-D**: compute lip and audio sync on the rendered video. Needs SyncNet
   pretrained weights, confirm they are downloadable first; if not, state in the document that it
   was not done, and do not substitute `jaw_audio_corr`.
2. **Blind rating sheet**: take 30 clips from the eval set, randomly pair
   `base / v2 / v3(+BoN)` and randomize left and right, generating
   `outputs/human_eval/sheet.csv`, ids and rating columns only, and `key.csv`, the mapping, saved
   separately. Nobody has to rate it yet, it only needs to be generated.
3. **Video side MediaPipe metrics**, reusing `video_metrics.py`: blink_rate, head_jitter,
   gaze_std and others.
4. Render scale: eval set, 50 of the 100 public segments, times 4 variants,
   base, v2, v3, v3+BoN, 200 videos, about 1.2 h. mp4 does not go to `main`; push it to a
   `backup/*` branch immediately after rendering, section 0c, and keep only GIF, json and md in
   `main`. Portraits are still the existing 3 in `data/portraits`, state this limitation in the
   document.

### Success criteria, written down first, not changed after results arrive

- Video side head_jitter: v3 at most 1.4x base, base about 120, so at most about 170; v2 is
  currently about 260.
- Blink: v3+BoN not worse than v2+BoN, 25.6.
- SyncNet: v3 not worse than v2 beyond the noise level, report value and standard deviation.
- If not met: write honestly that single-point jitter tuning did not work, with the reason, and
  attach the T1 diagnosis.

---

## 7 · Time budget and stop line

| stage | time |
|---|---|
| T0 | 0.3 h |
| T1 | 0.7 h |
| T2 | 0.4 h |
| T3 | 2.0 h |
| T4 | 2.0 h |
| T5 | 0.5 h |
| total | about 6 h |

When sharing the GPU with 08, 07 runs in the gaps of the 08 long training. Past 80 percent of
the remaining window, cut the second and third seed of T3, do not cut T1 or T4. **Do not start
new tasks in the last 90 minutes, switch to a full backup.**

---

## 8 · Commit rules

- **Push as soon as training finishes, do not wait for the end of a stage.** Method and branches
  are in section 0c; the first thing after each task ends is `tools/backup_push.sh`, and record
  the commit hash in `docs/GATE_LOG.md`.
- Push using the method already configured on this machine, SSH. Never write credentials into
  any file or command.
- Before pushing to `main`, run the forbidden-word check per the house rules in `/AGENTS.md`.
- Write all results to `docs/RESULTS_NEXT.md`, with the numbers as they are, including failures.
  Wording in the README and `TIER2_NOTE.md` must match the results, for example if jitter did not
  drop it must not say it was mitigated.
- When done, update the 07 section of `STATUS.md` and `/PROJECT_STATUS.md`.
