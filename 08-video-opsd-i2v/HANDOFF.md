# Handoff — Tier 1 edition

**Current date:** 2026-10-27 (D1 of 20). **CVPR 2027 registration: Nov 10 AoE (14 days).**

The scaffold on `main` is the **Tier 1 + Tier 2** two-tier plan. Tier 1 is the critical path for
the CVPR submission; Tier 2 is a bonus. See `README.md` for the full story and `docs/METHOD.md`
for the formal setup.

---

## 0 · One-time machine setup (D1, ~ 15 min)

```bash
cd 08-video-opsd-i2v
bash scripts/01_setup_env.sh                           # Python venv + torch MPS
source .venv/bin/activate && source .envrc
python -c "import src; print('ok')"
cp .env.example .env                                    # fill in HF_TOKEN at minimum
```

**Open `https://huggingface.co/stabilityai/stable-diffusion-3.5-medium` and click "Agree and
access repository" NOW.** SD3.5-M is gated; the agreement is instant but required.

---

## 1 · Download weights (D1-D2, ~45 min)

```bash
bash scripts/02_download_weights.sh
```

Fetches: DynamiCrafter code + ckpt, VBench, aesthetic predictor, **SD3.5-M (~12 GB)**, and
**OPSD LoRA (~50 MB)**. Needs ~ 25 GB free on disk and a stable connection.

**Gate 1 (D2 end):** verify the install works
```bash
python -c "from diffusers import StableDiffusion3Pipeline; print('sd3 ok')"
python -c "import vbench; print('vbench ok')"
```

---

## 2 · Prepare prompt lists (D2, ~ 2 min)

```bash
bash scripts/03_prepare_data.sh
```

Writes `data/vbench_i2v_prompts/{train,eval}_split.json` and `data/prompts/{train,eval}.json`.
For Tier 1 we only need the eval split (200 prompts).

**No images needed from you** — Tier 1 generates first frames from SD3.5-M. If you *also* want
to run a scene-diversity ablation with CC0 photos, do:

```bash
# optional — only if your ablation plan needs it
cp .env.example .env     # add PEXELS_API_KEY (free, from pexels.com/api/)
bash scripts/04_crawl_pexels.sh
```

---

## 3 · Smoke test (D2 end, 20 min)

```bash
# first, run Tier 1 on only 10 prompts × 1 seed to validate the pipeline
python -m src.tier1_keyframe_pipeline \
  --config configs/tier1_carryover.yaml \
  --skip_judge                  # judge needs Qwen2-VL, skip for smoke
# add temporarily to tier1_carryover.yaml: `limit: 10` and `seeds: [0]`
```

**Gate 2 (D2 end):** 10 videos generated for both base and OPSD, VBench scores printed.
Expected delta on aesthetic + imaging quality already positive.

---

## 4 · Full Tier 1 run (D3-D6)

Edit `configs/tier1_carryover.yaml` back to `seeds: [0, 1, 2]`, `limit: null`.

```bash
bash scripts/30_run_tier1_pipeline.sh
```

Expected wall time on M3 Pro 36 GB:
- T2I 200 prompts × 3 seeds × 2 modes = 1200 frames @ ~ 15 s each ≈ **5 h**
- I2V 1200 videos @ ~ 90 s each ≈ **30 h** (run overnight × 2 nights, or batch to cloud GPU)
- VBench ~ 2 h
- Qwen2-VL judge 200 pairs × 4 rubrics ≈ **5 h**

If you can get a 1× A100 for 24 h on vast.ai or RunPod, this drops to ~ 4 h total. For CVPR this
is worth the ~ $20.

**Gate 3 (D6 end):** `outputs/tier1/tier1_main_table.md` shows Δ > 0 on ≥ 6 of 9 VBench-I2V dims
with p<0.05 (Holm-Bonferroni). If so → paper's main result is already in hand.

---

## 5 · Tier 2 bonus (D7-D13) — optional

Only if Tier 1 is already green (Gate 3 passed):

```bash
bash scripts/10_train_opsd_i2v.sh 0        # seed 0
bash scripts/11_train_baseline_instructvideo.sh
bash scripts/12_train_baseline_dpovideo.sh
bash scripts/20_eval_vbench_i2v.sh 500
bash scripts/22_make_figures.sh
```

See the first version of `README.md` and `docs/METHOD.md §2` for the Tier 2 derivation. Code is
already in `src/opsd_trainer_i2v.py` and `src/baselines/`.

---

## 6 · Paper (D14-D20)

- D14 (Nov 09): submit title + abstract + authors to CVPR OpenReview. Use `docs/CVPR_OUTLINE.md`
  abstract draft.
- D15-D17: write §1-§7 following `docs/CVPR_OUTLINE.md`.
- D20 (Nov 15): submit full paper (deadline Nov 16 AoE).
- D28 (Nov 23 AoE): submit supplementary.

---

## 7 · Push to GitHub

Already configured (SSH key, D1 2026-10-27):

```bash
# edit anything, then:
cd /Users/jingxiduan/CodeBuddy/20261027111906/_portfolio_repo/ml-portfolio
git pull --rebase origin main
# rsync changes in from the staging area:
rsync -a --delete \
  --exclude ".venv/" --exclude ".envrc" --exclude ".env" \
  --exclude "weights/" --exclude "third_party/" --exclude "outputs/" \
  --exclude "wandb/" --exclude "data/first_frames/*" \
  --exclude "data/pexels_cc0/*" --exclude "data/prompts/*.json" \
  --exclude "data/vbench_i2v_prompts/*.json" \
  --exclude "__pycache__/" --exclude ".DS_Store" \
  /Users/jingxiduan/CodeBuddy/20261027111906/08-video-opsd-i2v/ \
  08-video-opsd-i2v/
git add 08-video-opsd-i2v
git commit -m "d{N}: ..."
git push origin main
```

Or run `scripts/99_push_to_github.sh` after setting `PORTFOLIO_PATH`.

---

## 8 · Known stubs to patch

Same as the first HANDOFF; still relevant after Tier 2 is wired. These are typically **15 min
each** of grep-replace once the real vendored code is on disk:

1. `src/dynamicrafter_wrapper.py :: rollout_until` — DynamiCrafter DiT forward signature.
2. `src/render_i2v.py :: _load_scheduler` — swap in DynamiCrafter's real DDIM sampler.
3. `src/eval/vbench_i2v_eval.py :: run_vbench` — verify VBench CLI version.
4. `src/utils/lora_utils.py :: TEMPORAL_TARGETS` — regex vs real param names.

**Tier 1 does not need #1 or #4** (no training, no LoRA on video); only #2 and #3 matter for the
main experiment. That's an easier D2 than before.
