# AGENTS.md — Project overview for AI coding assistants

> **If you are an AI assistant (CodeBuddy / Cursor / Claude Code / etc.) opening
> this repo for the first time: read this file first, then `08-video-opsd-i2v/docs/H20_SESSION_HANDOFF.md`
> if you are on the H20 training host. That gets you fully up to speed.**

---

## What this repo is

`jingxiduan/ml-portfolio` — personal ML research portfolio.

Flagship project: **`08-video-opsd-i2v/`** — "Diagnosing and Repairing a
Reward-Hacked Image-to-Video LoRA". See its README for the full story.

## House rules (apply to the whole repo)

1. **No conference / paper / submission language anywhere in commits or docs.**
   The user flagged this as sensitive. Legitimate paper *citations* (e.g.
   "InstructVideo, CVPR 2024") are fine; our own "submitting to", "under
   review", "paper deadline" wording is not. Grep for `cvpr|under review|
   submission` before every push.

2. **Every number in README / docs must be verifiable from on-disk artefacts.**
   If a figure disagrees with the file it claims to come from, treat it as a
   bug — do not "round" or "estimate" numbers in prose. The user explicitly
   asked for honest numbers; previous iterations contained fabricated rows
   that had to be removed.

3. **`did not deliver` columns are not failures — they are features.**
   The honest-limitations section in `08-video-opsd-i2v/README.md` §4 is a
   design choice. When you solve one of those items, move it out of "did not
   deliver" and into the delivery table; do not silently delete it.

4. **The `08-video-opsd-i2v/` directory in this repo is the *shipping* copy.**
   The user's local workspace at `/Users/jingxiduan/CodeBuddy/<id>/08-video-opsd-i2v/`
   is the *development* copy. Changes flow dev → shipping via explicit `cp`
   or `rsync`, not the other way around. Shipping copy is what GitHub sees.

5. **Branch is `main`. Commits should be prefixed `08:`** (or the project
   number). One logical change per commit; include test results in commit
   message when applicable.

## Current state (2026-10-29 evening)

- Portfolio last commits: `d4fd926` (H20 playbook + SSH setup);
  pending commit: 07 Tier-2 BoN+thin (motion coefs + 1 verified clip)
- Project 08's 3 tracks of Tier-2 output already shipped:
  1. **Best-of-N** (inference-time rejection sampling) — `docs/tier2/tier2_bon_r{1,2}_metrics.md`
  2. **lightness post-process** — `docs/tier2/verdict_table.md`
     (paired Wilcoxon p=0.016 vs base, p=0.031 vs scale=0.2, n=6)
  3. **BoN × lightness composition** — `docs/tier2/bon_plus_lightness_table.md`
     (mean Δnat +0.174, 6/6 positive)
- Project 07's Tier-2 (new, pending H20 render):
  - `src/tier2_bon.py` — anti-hacking naturalness_v2 (gaussian blink target
    around 18/min + hard jitter penalty) + blink-thinning post-process
    (greedy keep by peak height, ≥2.5s inter-blink, cosine attenuate non-lip
    expression dims)
  - `outputs/tier2/selected_coefs.pt` — 2.1 MB bundle, 8 clips × 5 variants of
    pre-selected motion coefficients (Mac BoN output; H20 renders them)
  - 1 clip verified end-to-end on Mac: video-space blink_rate
    `0 → 37.5 → 31.3 → 18.75`, lip-sync preserved
- **Not yet shipped** (currently queued for the 10/30 H20 window):
  - 07 Tier-2: remaining 7 clips rendered + 8-clip averaged metrics
  - 08 Tier-2: real trained LoRA checkpoint + 3-way comparison GIFs

## Where to go next

| If your task is… | Start with |
|---|---|
| H20 training/render on 10/30 | `08-video-opsd-i2v/docs/H20_SESSION_HANDOFF.md` ← **entry point** |
| 07 Tier-2 render on H20 | `07-talking-head-motion-opsd/docs/H20_RENDER_PLAYBOOK.md` |
| 08 Tier-2 training on H20 | `08-video-opsd-i2v/docs/H20_PLAYBOOK.md` |
| Understanding project 08 | `08-video-opsd-i2v/README.md` |
| Understanding project 07 | `07-talking-head-motion-opsd/README.md` |
| Adding a Tier-2 track to 08 | `08-video-opsd-i2v/docs/tier2/METHOD.md` |
| Debugging the trainer | `08-video-opsd-i2v/docs/tier2/trainer_smoke_log.md` + `tests/test_cond_stage.py` |
