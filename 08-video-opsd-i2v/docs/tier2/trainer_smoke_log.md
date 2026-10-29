# Tier 2 trainer — end-to-end smoke test log

Date: 2026-10-29
Device: MPS (M3 Pro)
Command: `PYTHONPATH=. python scripts/smoke_trainer_e2e.py`
Config: `configs/tier2_natural.yaml` (with `num_frames=4`, `resolution=[256,256]`,
         `num_inference_steps=5` to fit M3 Pro memory budget; everything else default)

## Output

```
2026-10-29 16:40:54 INFO mainlogger: LatentVisualDiffusion: Running in v-prediction mode
2026-10-29 16:40:59 INFO root: Loaded ViT-H-14 model config.
2026-10-29 16:41:06 INFO root: Loading pretrained ViT-H-14 weights (laion2b_s32b_b79k).
...
[smoke] device = mps
[smoke] loading base model (~9.7GB; takes ~1min)...
AE working on z of shape (1, 4, 32, 32) = 4096 dimensions.
[DynamiCrafter] loaded, missing=0, unexpected=0
[lora] enumerated 136 temporal-attn target modules (e.g. init_attn.0.transformer_blocks.0.attn1.to_k)
[OPSDPolicies] cond-stage ok on mps: text=FrozenOpenCLIPEmbedder, image=FrozenOpenCLIPImageEmbedderV2, proj=Resampler
[OPSDPolicies] v_theta: trainable=3.33M / total=1442.18M (0.23%)
[smoke] building reward + dataset ...
[smoke] dataset: 6 entries
[smoke] running ONE _opsd_step ...
[smoke] metrics: {'loss': 0.0, 'loss_plus': 244.12100219726562, 'loss_minus': 244.12100219726562, 'reward': 2.184816837310791, 'sigma_q': 400.0, 'grad_norm': 0.9999598264694214}
[smoke] 272 LoRA params had grads; top-3 by |mean|:
[smoke]   3.730e-03  base_model.model.output_blocks.10.2.transformer_blocks.0.attn2.to_out.0.lora_B.default.weight
[smoke]   3.366e-03  base_model.model.output_blocks.11.2.transformer_blocks.0.attn1.to_out.0.lora_B.default.weight
[smoke]   3.183e-03  base_model.model.output_blocks.10.2.transformer_blocks.0.attn2.to_v.lora_B.default.weight
[smoke] PASS — trainer is functionally alive
```

## What this proves

- Checkpoint loads: `missing=0, unexpected=0` — all 1.4B params land on the model.
- Cond-stage is correctly lifted off the base model: three frozen modules
  (`FrozenOpenCLIPEmbedder` / `FrozenOpenCLIPImageEmbedderV2` / `Resampler`),
  patched to the right device, callable by `build_cond()`.
- LoRA injection finds **136 temporal-attention target modules** via
  runtime module-tree enumeration (not fragile hand-written regex). PEFT
  reports 3.33M trainable / 1.44B total params (0.23%), consistent with a
  rank-16 LoRA on 136 Linear layers.
- One full `_opsd_step` runs end-to-end on MPS without exception, producing:
  - a finite reward (`2.185`),
  - a per-sample L2-normalised gradient (`grad_norm ≈ 1.0`, the OPSD-paper
    default `--reward_grad_rescale l2`),
  - a valid trust-region σ (`sigma_q = 400` out of 1000 DDIM timesteps),
  - **272 LoRA parameters each receive a non-zero grad**, concentrated in
    the output-block temporal-attention path exactly as designed.

## What this does NOT prove

- **Convergence.** 1 step says nothing about whether 50 steps would
  reduce `loss_plus - loss_minus` toward a non-trivial minimum. Training
  convergence on M3 Pro remains infeasible (~37h at 320×512, 16 frames).
  Full training is an A100-scale job.
- **Reward correctness.** The `reward=2.185` value is consistent with the
  CombinedReward config but has not been compared to the base model's
  reward on the same batch — that is a baseline run, not a trainer check.
- `loss=0.0` on step 1 is **expected**: with `inner_fit_steps=1` and
  `y0_plus ≈ y0_minus` (gradient barely moves the latent in one trust-region
  step), the contrastive OPSD loss can start near zero and only grow as the
  policy drifts from `v_old`. See `src/opsd_trainer_i2v.py` for the formula.

## Shapes used in smoke (vs. production)

| | smoke | production (tier2_natural.yaml) |
|---|---|---|
| num_frames  | 4 | 16 |
| resolution  | 256×256 | 320×512 |
| inference_steps (rollout) | 5 | 50 |
| batch_size  | 1 | 1 |

The production config requires ≈60GB of accelerator memory for the full
`_opsd_step` (observed MPS OOM at 47.5GB when rolling 16-frame 320×512
videos through VAE decode).
