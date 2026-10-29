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

## Current state (2026-10-29)

- Portfolio last commit: `fbbcc41` — H20 playbook + scripts shipped, waiting
  for the 10/30 00:00–10:00 training window
- Project 08's 3 tracks of Tier-2 output already shipped:
  1. **Best-of-N** (inference-time rejection sampling) — `docs/tier2/tier2_bon_r{1,2}_metrics.md`
  2. **lightness post-process** — `docs/tier2/verdict_table.md`
     (paired Wilcoxon p=0.016 vs base, p=0.031 vs scale=0.2, n=6)
  3. **BoN × lightness composition** — `docs/tier2/bon_plus_lightness_table.md`
     (mean Δnat +0.174, 6/6 positive)
- **Not yet shipped** (and currently in flight on H20):
  - A real trained Tier-2 LoRA checkpoint
  - 3-way comparison GIFs (base | scale=0.2 | trained_tier2_lora)
  - `docs/tier2/tier2_lora_metrics.md`

## Where to go next

| If your task is… | Start with |
|---|---|
| H20 training on 10/30 | `08-video-opsd-i2v/docs/H20_PLAYBOOK.md` + `H20_SESSION_HANDOFF.md` |
| Understanding project 08 | `08-video-opsd-i2v/README.md` |
| Adding a Tier-2 track | `08-video-opsd-i2v/docs/tier2/METHOD.md` |
| Debugging the trainer | `08-video-opsd-i2v/docs/tier2/trainer_smoke_log.md` + `tests/test_cond_stage.py` |
