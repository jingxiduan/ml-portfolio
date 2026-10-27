# VBench-I2V prompt lists

`scripts/03_prepare_data.sh` populates this folder with the official VBench-I2V prompt list
(cloned from `github.com/Vchitect/VBench`). Expected files after setup:

- `vbench_i2v_full.json` — 1118 prompts.
- `train_split.json` — our 80% training split (indices).
- `eval_split.json` — our 20% held-out eval split (indices).

Prompt file format:
```json
[
  {"id": "vb_000", "prompt": "a cat walking across a lawn", "category": "subject_motion"},
  ...
]
```
