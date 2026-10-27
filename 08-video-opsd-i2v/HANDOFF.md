# Handoff — what you need to do next

Scaffold is written. Follow this in order; each step has an obvious "done" criterion.

## 0 · Smoke test the scaffold (5 min, no training yet)

```bash
cd 08-video-opsd-i2v
bash scripts/01_setup_env.sh
source .venv/bin/activate && source .envrc
python -c "import src; from src.rewards import REWARD_REGISTRY; print(REWARD_REGISTRY)"
```
Expected: dict with 4 reward names printed. If this fails → fix the error before anything else.

## 1 · Fetch weights (30–60 min, big download)

```bash
bash scripts/02_download_weights.sh
```
Downloads DynamiCrafter (~ 10 GB), VBench, aesthetic predictor. Needs stable WiFi.

## 2 · Build prompt lists + collect first-frame images

```bash
bash scripts/03_prepare_data.sh                 # writes data/prompts/*.json
```
Then **you** drop first-frame images into `data/first_frames/000.png, 001.png, ...` matching
`data/prompts/train.json` by index. Minimum 50 images to get a signal; 200 for the paper.

Any of these work:
- Your own photos
- SDXL-generated (if you re-use `07/make_portraits.py` logic)
- Pexels CC0 downloads

## 3 · First training run (Day 2–3)

```bash
bash scripts/10_train_opsd_i2v.sh 0            # seed 0
```
Watch `outputs/opsd_i2v_default/`. Early signs it's working:
- `loss` drops from +0.X toward 0 and oscillates
- `reward` trends up over 100+ steps
- No NaNs

If NaNs → lower lr to 1e-5 or set `dtype_override: "fp32"` in the config.

## 4 · Baselines (Day 4–5)

```bash
bash scripts/11_train_baseline_instructvideo.sh
bash scripts/12_train_baseline_dpovideo.sh
```

## 5 · Evaluation (Day 6)

```bash
bash scripts/20_eval_vbench_i2v.sh 500
bash scripts/21_eval_llm_judge.sh 200
bash scripts/22_make_figures.sh
```

Open `outputs/tables/main_table.md` and `outputs/figures/*.pdf`. **This is your gate:**

- If OPSD-I2V wins **8 of 9** VBench dims → write the paper as planned.
- If OPSD-I2V wins **5–7 of 9** → adjust reward weights in `configs/opsd_i2v_default.yaml`
  (reduce `motion_smoothness.weight` if dynamic degree is low; reduce `dynamic_degree.weight`
  if subject consistency is low) and retrain **once**.
- If OPSD-I2V wins **<5 of 9** → consult `docs/TIMELINE.md` abort conditions.

## 6 · Paper registration (Nov 10 AoE — Day 14)

Title + abstract + authors on OpenReview. Use `docs/CVPR_OUTLINE.md` abstract draft.

## 7 · Push to GitHub

I cannot do this for you (needs your git credentials). When you're ready:

```bash
# first time:
git clone git@github.com:jingxiduan/ml-portfolio.git
export PORTFOLIO_PATH=$PWD/ml-portfolio

# every time thereafter:
bash scripts/99_push_to_github.sh
```

The `99_push_to_github.sh` script copies this folder into the ml-portfolio repo (excluding
weights, outputs, data blobs, venv) and pushes to `main`.

## Known gaps / where you'll likely need to patch

These are **stubs I could not test without the actual weights**. Expect to spend ~1 day fixing:

1. **`src/dynamicrafter_wrapper.py :: rollout_until`** — the forward signature of the
   DynamiCrafter DiT (`v(z, t, cond=cond)`) is my best guess from reading the repo. Once you
   clone `third_party/DynamiCrafter`, grep for `class LatentVisualDiffusion` and fix the call.
2. **`src/train_entry.py :: _make_scheduler`** — I used `DDIMScheduler` as a stand-in. The real
   DynamiCrafter sampler is in `third_party/DynamiCrafter/lvdm/models/samplers/ddim.py`. Swap
   it in when you vendor the repo.
3. **`src/eval/vbench_i2v_eval.py :: run_vbench`** — VBench's CLI changed between v1 and v2.
   Verify the invocation with `python -m vbench.evaluate --help` after installation.
4. **`src/utils/lora_utils.py :: TEMPORAL_TARGETS`** — regex patterns are my best guess. After
   loading the base model, do:
   ```python
   for n, _ in model.named_parameters(): print(n)
   ```
   and update the patterns if they don't match.

None of these are hard; each is 15 min of grep+replace once you have the vendored code.

## Timeline reminder

Today = 2026-10-27. CVPR 2027 **registration** Nov 10, **submission** Nov 16. **20 days.**
See `docs/TIMELINE.md` for the day-by-day plan.
