# 20-Day Timeline — CVPR 2027

Current date: **2026-10-27 (Tuesday)**

| Day | Date | Deliverable | Gate / Risk |
|---|---|---|---|
| **D1** | 10/27 Tue | Scaffold + README + docs committed; `01_setup_env.sh` runs clean on M3 Pro | — |
| D2 | 10/28 Wed | DynamiCrafter loaded on MPS; 10 base videos rendered on user-provided first frames | **Gate 1**: base inference works |
| D3 | 10/29 Thu | VBench-I2V repo cloned, 5 reward dims run on base videos | — |
| D4 | 10/30 Fri | OPSD-I2V trainer first forward+backward pass green (1 step, loss finite) | **Gate 2**: trainer compiles |
| D5 | 10/31 Sat | OPSD-I2V 50-step smoke train; EMA refresh verified | — |
| D6 | 11/01 Sun | OPSD-I2V 500-step full training, run 1 (seed 0) | — |
| D7 | 11/02 Mon | OPSD-I2V 500-step full training, runs 2–3 (seeds 1, 2) | **Gate 3**: 3 seeds done, variance ≤ expected |
| D8 | 11/03 Tue | InstructVideo baseline 500 × 3 seeds | — |
| D9 | 11/04 Wed | DPO-Video baseline 500 × 3 seeds | — |
| D10 | 11/05 Thu | All 4 model VBench-I2V eval on held-out split | **Gate 4**: OPSD beats both baselines on ≥7/9 dims |
| D11 | 11/06 Fri | Qwen2-VL LLM-judge pairwise eval | — |
| D12 | 11/07 Sat | Ablation: LoRA rank {8,16,32}, $c_{adv}$ {3,5,7}, refresh {every 25,50,100} | — |
| D13 | 11/08 Sun | Figures: method diagram, teaser, main table, ablation plots, qualitative grid | — |
| **D14** | **11/09 Mon** | **Paper registration draft** (title/abstract/authors) ready to submit by Nov 10 AoE | **GATE 5: REGISTRATION (Nov 10 AoE)** |
| D15 | 11/10 Tue | Paper writing: §1 intro, §2 related, §3 method | — |
| D16 | 11/11 Wed | Paper writing: §4 experiments, §5 analysis, §6 discussion | — |
| D17 | 11/12 Thu | First complete draft; internal pass | — |
| D18 | 11/13 Fri | **(your personal ddl)** second-pass draft; polish figures | — |
| D19 | 11/14 Sat | Final checks; supp outline | — |
| **D20** | **11/15 Sun** | **Paper submission** (buffer: Nov 16 AoE is official) | **GATE 6: SUBMISSION (Nov 16 AoE)** |
| D21–D27 | 11/16–11/22 | Supp: 2-min demo video + extended results + code release prep | — |
| **D28** | **11/23 Mon** | **Supplementary submission** (Nov 23 AoE) | **GATE 7: SUPP (Nov 23 AoE)** |

## Abort conditions

If any gate is missed by > 24h, trigger the **downgrade plan**:

- **Gate 2 miss (D4)** → drop DynamiCrafter, swap to **AnimateDiff + SD1.5** (simpler, better
  MPS compat).
- **Gate 3 miss (D7)** → cut 3 seeds to 2, cut 500 steps to 300.
- **Gate 4 miss (D10)** → sharpen reward weights (reduce dynamic-degree weight if OPSD is
  reward-hacking, or increase it if too passive). One re-train allowed.
- **Gate 5 miss (D14)** → **submit to CVPR Workshop** (which has later ddl) or ICCV 2027 (fallback,
  ddl ~ March 2027).

## Rules

- No new features after **D11**. Only bug fixes and writing.
- Every day end with a `git commit -m "d{N}: ..."` so progress is auditable.
- Each gate failure must be documented in `docs/POSTMORTEM.md` before proceeding.
