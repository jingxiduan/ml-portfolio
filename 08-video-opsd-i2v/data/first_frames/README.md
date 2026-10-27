# First-frame images for I2V training

Place your `.png` / `.jpg` first-frame images here. Guidance:

- **Resolution**: 320×512 or 512×320 (DynamiCrafter native). Non-matching will be center-cropped.
- **Count**: ~200 for training, ~50 for evaluation (hold-out).
- **Content**: any licensed images. If using SDXL-generated frames, keep the generation prompts in
  a sibling `prompts.json` so (frame, prompt) can be paired.

Expected filename pattern: `000.png`, `001.png`, ..., paired with `../prompts/train.json` entries
by index.

**Nothing in this folder is tracked by git** (see `../../.gitignore`). This README is tracked.
