# 20-Day Timeline — CVPR 2027 (Tier 1 centric)

Current date: **2026-10-27 (Tuesday)**.
CVPR 2027: **Paper registration Nov 10 AoE / Submission Nov 16 AoE / Supp Nov 23 AoE**.

Compared to the first draft, Tier 1 (zero video training) is now the critical path. Tier 2 is a
bonus arc. The headline paper can ship with Tier 1 alone.

| Day | Date | Deliverable | Gate / Risk |
|---|---|---|---|
| **D1** | 10/27 Tue | Scaffold v2 pushed. Tier 1 pipeline file in. | — |
| D2 | 10/28 Wed | Env + weights + VBench installed. First base-SD3.5 image generated on MPS. | **Gate 1: inference stack works** |
| D3 | 10/29 Thu | OPSD LoRA loaded on SD3.5-M; sanity-check 10 pairs base vs OPSD frames; eyeball looks reasonable. | **Gate 2: LoRA correctly applied** |
| D4 | 10/30 Fri | DynamiCrafter inference wired to accept generated frames; 10 videos per side rendered. | **Gate 3: end-to-end pipeline runs** |
| D5 | 10/31 Sat | Full 200-prompt generation (3 seeds × 2 methods × 200 prompts = 1200 videos). ~ 6 h on MPS, parallel if cloud GPU available. | — |
| D6 | 11/01 Sun | VBench-I2V on all 1200 videos. Compute Table 1. | **Gate 4: Δ > 0 on ≥ 6 of 9 dims (p<0.05)** |
| D7 | 11/02 Mon | Qwen2-VL judge on 200 pairs. Matched-aesthetic ablation (H3). | — |
| D8 | 11/03 Tue | **(if Gate 4 passed)** Start Tier 2 training run 1. **(if Gate 4 failed)** Analyze why; consider pivoting to Tier 2 as primary. | **Gate 5: Tier 1 is paper-ready** |
| D9 | 11/04 Wed | Tier 2 run 1 (seed 0, 500 steps). | — |
| D10 | 11/05 Thu | Tier 2 runs 2–3 (seeds 1, 2). | — |
| D11 | 11/06 Fri | Tier 2 VBench eval. Baselines (InstructVideo, DPO-Video). | **Gate 6: Tier 2 shows gap > ReFL** |
| D12 | 11/07 Sat | Per-category breakdown, failure-mode analysis, figures. | — |
| D13 | 11/08 Sun | All experiments done. Freeze results. | **GATE 7: results freeze** |
| **D14** | **11/09 Mon** | Paper registration content (title + abstract + authors) ready. Submit by Nov 10 AoE. | **GATE 8: REGISTRATION Nov 10 AoE** |
| D15 | 11/10 Tue | Paper §1 Intro + §2 Related + §3 Method. | — |
| D16 | 11/11 Wed | §4 Experiments + §5 Analysis. | — |
| D17 | 11/12 Thu | §6 Limitations + §7 Conclusion. Complete draft. | — |
| D18 | 11/13 Fri | **(your personal ddl)** polish + figure pass. | — |
| D19 | 11/14 Sat | Internal review pass; final figure cleanups. | — |
| **D20** | **11/15 Sun** | Submit paper (official ddl Nov 16 AoE). | **GATE 9: SUBMISSION Nov 16 AoE** |
| D21–D27 | 11/16–22 | Supp: 2-min demo video, extended tables, code polish, human study if any. | — |
| **D28** | **11/23 Mon** | Submit supplementary. | **GATE 10: SUPP Nov 23 AoE** |

## Critical path differences from v1

- **D2–D7 is Tier 1 only** — zero video training. Fail-safe: even if Tier 2 doesn't materialize,
  Tier 1 alone is a submittable paper (though a weaker one).
- **D8 is the decision point** for Tier 2 inclusion. If Tier 1 passed Gate 4, we have time; if
  not, Tier 2 is still the fallback, but we would narrow the paper claim.
- **D13 "results freeze"** is non-negotiable. From D14 onward it's writing-only.

## Abort / pivot plan

| Failure | Pivot |
|---|---|
| Gate 1 (D2) | DynamiCrafter MPS fails → swap to CogVideoX-I2V-2B on cloud GPU for Tier 1 (same pipeline). |
| Gate 2 (D3) | OPSD LoRA fails to apply → use `z-image-turbo-hpsv3` variant instead. |
| Gate 3 (D4) | Pipeline integration bug > 1 day → ship with 100 prompts × 2 seeds instead of 200 × 3. |
| Gate 4 (D6) | Δ not significant on ≥ 6 dims → investigate: wrong LoRA scale? wrong prompt subset? Re-tune in 1 day; if still fails, flip paper to **"when does T2I alignment NOT transfer to video?"** — still publishable at a workshop. |
| Gate 5 (D8) | Tier 1 narrative too thin → promote Tier 2 to primary (revert to original scaffold plan). |
| Gate 6 (D11) | Tier 2 < ReFL → drop Tier 2 from main paper, keep in supp as "a modest further extension". |
| Gate 8 (D14) | Missed registration → submit to CVPR Workshop (deadline ~1 month after main) or ICLR 2027 (deadline ~Sep 2027 next cycle). |

## Daily discipline

- Every day end with `git commit -m "d{N}: ..."` and (if changes significant) a push.
- Every gate pass or fail is recorded in `docs/POSTMORTEM.md`.
- No new features after **D13**. Only bugfixes and writing from D14 onward.
