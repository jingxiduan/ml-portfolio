# 07 · Tier-2 BoN render playbook (H20 side)

**What this is:** render the remaining Tier-2 BoN motion coefficients that
were already selected on Mac. **No motion generation**, no training, pure
LivePortrait rendering on CUDA.

**Why on H20:** LivePortrait on Mac MPS has a missing op (`grid_sampler_3d`)
and falls back to CPU, so each 10s video takes 2-3 minutes. On H20 with
`onnxruntime-gpu` this should be sub-20 seconds per video, i.e. 7 clips × 4
variants × 20s ≈ 10 minutes end-to-end (versus 85 min on Mac).

---

## Inputs (already in the repo, cloned from GitHub)

| path | what |
|---|---|
| `outputs/tier2/selected_coefs.pt` | 2.1 MB bundle: 8 clips × 5 variants of (73-d motion coef, seed, score) |
| `outputs/opsd/lora.pt` | OPSD-trained LoRA (needed only if you re-generate motions, not for this task) |
| `outputs/eye_probe/eye_probe.pt` | eye probe (loaded on module import, used for in-pipeline scoring) |
| `data/audio/eval/*.wav` | 8 held-out audio clips |
| `data/portraits/host_{f,f2,m}.png` | 3 fictional faces (SDXL) |

## Expected outputs

| path | what |
|---|---|
| `outputs/tier2/bon_cXX_{base_naive,opsd_naive,opsd_bon,opsd_bon_thin}.mp4` | 8 × 4 = 32 videos |
| `outputs/tier2/bon_metrics.json/md` | video-space MediaPipe metrics, 4 variants averaged over 8 clips |
| `outputs/tier2/tier2_compare_{00..07}.gif` | 3-way side-by-side comparison GIFs (base \| opsd \| opsd+BoN+thin) |

---

## The commands

```bash
# 0. prerequisites: inside H20 container, after cloning the repo
cd ~/ml-portfolio/07-talking-head-motion-opsd

# 1. environment --- same recipe as the README's Repro section, but with
#    onnxruntime-gpu instead of onnxruntime (if available)
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision torchaudio "numpy<2" opencv-python librosa pyyaml tyro rich \
  imageio imageio-ffmpeg scikit-image onnxruntime-gpu onnx "transformers<4.46" pykalman \
  matplotlib scipy tqdm huggingface_hub mediapipe soundfile diffusers

# 2. third-party (JoyVASA + LivePortrait weights + MediaPipe landmarker)
#    - clone JoyVASA to third_party/JoyVASA
#    - download: KwaiVGI/LivePortrait (human), jdh-algo/JoyVASA, TencentGameMate/chinese-hubert-base
#    - MediaPipe face_landmarker.task -> third_party/mediapipe/
#    - then apply MPS patches (also harmless on CUDA):
python src/patch_joyvasa.py

# 3. render all 32 videos (motion coefs are cached, no gen)
python src/tier2_bon.py --load-coefs outputs/tier2/selected_coefs.pt --n-eval-clips 8

# 4. build 8 comparison GIFs
python src/make_tier2_gifs.py --max-side 240 --fps 12

# 5. commit and push
git add outputs/tier2/bon_metrics.{json,md} outputs/tier2/tier2_compare_*.gif
git commit -m "07: Tier-2 BoN+thin · 8 clips rendered on H20"
git push
```

## What success looks like

- `bon_metrics.md` averaged over 8 clips shows:
  - `blink_rate`: base ~0, opsd_naive ~50+, **opsd_bon_thin ~15-25** (human range)
  - `head_pitch_std / yaw_std`: opsd_bon_thin close to opsd_naive (motion preserved)
  - `jaw_audio_corr / jaw_open_mean`: unchanged across variants (lip-sync preserved)
- 1 clip on Mac already verified: base_naive=0, opsd_naive=37.5, opsd_bon=31.3,
  **opsd_bon_thin=18.75** → see `outputs/tier2/tier2_compare_00.gif`

## If LivePortrait is slow / fails on CUDA

- Check `onnxruntime` version: it must be the `onnxruntime-gpu` package;
  `pip uninstall onnxruntime; pip install onnxruntime-gpu`
- CPU fallback: `python src/tier2_bon.py --load-coefs ... --n-eval-clips 8` still
  works purely on CPU; just slower. All generation already done on Mac so this
  is cheap even on CPU (10 min/clip × 32 = 5 hours, still fits).
- If you really can't render on H20, Mac can finish the remaining 7 clips in
  ~1 hour wall-clock using `PYTORCH_ENABLE_MPS_FALLBACK=1`.
