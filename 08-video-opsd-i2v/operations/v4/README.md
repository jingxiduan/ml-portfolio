# V4 operation record

## Goal

Test whether effective parameter-gradient clipping stabilizes the anchored I2V OPSD objective beyond the V3 failure at step 522. This run changes only `grad_clip` from `1.0` to `1e-3`. The first V4 process was externally interrupted at step 1038 by an OOM while a separate development renderer was mistakenly run concurrently; the clean rerun preserves algorithm, seed, data order and hyperparameters in a new output directory.

## Immutable inputs

- Config: `configs/long_v4_gradclip.yaml`
- Base: DynamiCrafter 512 I2V
- Train frames: `data/first_frames_200`
- Train prompts: `data/prompts/tier2_train_200.json`
- Development frames: `data/first_frames_dev_coco6/seed_00`
- Development prompts: `data/prompts/coco_dev_6.json`
- Confirmatory frames: `data/first_frames_eval_coco40/seed_00`
- Confirmatory prompts: `data/prompts/coco_eval_40.json`
- Provenance: `data/first_frames_dev_coco6/SOURCES.json` and `data/first_frames_eval_coco40/SOURCES.json`
- Train seed: 0
- Evaluation seed: 0

The COCO validation pool is split in dataset order after a modification-compatible license filter. The first six pairs are development-only; the following 40 are confirmatory. The confirmatory set is not used in training, calibration, checkpoint/scale selection, or reward tuning.

## Live run

```bash
python -m src.train_entry \
  --method opsd_i2v \
  --config configs/long_v4_gradclip_rerun.yaml

python ../tools/train_guard.py \
  --kind i2v \
  --log outputs/opsd_long_v4_gradclip_rerun/train_metrics.jsonl \
  --pid "$(cat outputs/opsd_long_v4_gradclip_rerun/train.pid)" \
  --poll 10 \
  --stall-sec 1200
```

Primary output directory: `outputs/opsd_long_v4_gradclip_rerun`. No rendering or other GPU process is permitted concurrently with the clean rerun.

## Fixed completion route

1. Preserve `train.log`, `guard.log`, `train_metrics.jsonl`, guard decision, and checkpoints 500/1000/1500/2000.
2. Generate `fig_v4_summary.png` with `scripts/plot_v4_summary.py`.
3. If and only if step 2,000 completes without a guard stop, evaluate checkpoints `{500,1000,1500,2000}` at LoRA scales `{0.01,0.02,0.05}` on exactly six development pairs.
4. A development candidate is eligible when Δnaturalness ≥ -0.02 and Δappeal ≥ -0.02; among eligible candidates select the maximum `Δappeal + 0.5·Δnaturalness + 0.5·Δdynamic`, breaking ties toward lower scale and earlier checkpoint.
5. Render exactly one selected candidate and Base on the untouched 40-pair confirmatory set.
6. Compute paired appeal, naturalness, and dynamic metrics with 10,000 bootstrap draws.
7. Generate fixed-index Base/candidate GIF demos for confirmatory indices 0–5; do not select examples after viewing quality.
8. Publish the result on the repository home page and in `docs/RESULTS_LONG.md`.
9. Commit small evidence to `main`, back up videos/checkpoint on the backup branch, and push immediately.

## Development smoke observed before selection

A full-strength step-1000 adapter produced severe green/overexposed output on development pair 0. The issue was found before viewing the confirmatory 40-pair split. The scale grid above is therefore fixed to low values and the full-strength adapter is not eligible for final evaluation. This smoke is a development diagnosis, not a held-out result.

## Claim rule

A quality metric is called improved only when the paired-bootstrap 95% interval for candidate-minus-base lies entirely above zero. Training reward, attractive hand-picked examples, or successful completion alone cannot establish improvement. A null or failed result remains published.

## Visible deliverables

- Training/stability curve
- n≥40 paired metric table and uncertainty intervals
- Six fixed-index GIF comparisons
- Base and V4 MP4 links
- Exact input provenance and license manifest
- Full logs/checkpoint/media backup
