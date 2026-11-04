# Reproducible talking-head demo

This page is the shortest path from a clean checkout to a visible before/after comparison. It deliberately separates a dependency-free CPU smoke test from the real GPU render.

## 1. CPU-only mock demo

The mock does not load model weights and does not claim model quality. It validates the repository entry point, output layout, comparison labels, and links to the pre-generated model render.

```bash
cd 07-talking-head-motion-opsd
python3 scripts/demo.py --mode mock
python3 -m http.server 8000 --directory outputs/demo/cpu_mock
# open http://localhost:8000
```

The command writes:

- `outputs/demo/cpu_mock/index.html`: animated Base vs Calibrated Policy mock;
- `outputs/demo/cpu_mock/comparison.svg`: dependency-free visual artifact;
- `outputs/demo/cpu_mock/manifest.json`: mode, timing, and claim boundary.

## 2. GPU complete render

The real pipeline uses JoyVASA audio-to-motion plus the frozen LivePortrait renderer. The setup is idempotent and installs CUDA-matched PyTorch, project dependencies, third-party code, and model weights:

```bash
cd 07-talking-head-motion-opsd
bash scripts/h20_render_setup.sh
python3 scripts/demo.py --mode gpu \
  --portrait data/portraits/host_f.png \
  --audio data/audio/demo/demo_00.wav \
  --seconds 4 \
  --seed 0 \
  --out outputs/demo/gpu_r3
```

The GPU command renders the matched `base`, `seed0`, `seed1`, and `seed2` variants with the same portrait, audio, duration, and generation seed. It writes `manifest.json` from the renderer and `benchmark.json` containing:

- total wall-clock inference time;
- GPU memory before, sampled peak, and after rendering;
- GPU utilization samples;
- the exact child command used.

Peak memory is sampled through `nvidia-smi` once per second, so it is intentionally reported as an operational estimate rather than a CUDA allocator trace. For a shorter smoke render, use `--seconds 1`; the default 4-second run matches the evaluation protocol.

### Reference hardware numbers

The documented H20 render playbook reports **under 20 seconds per generated video** on an H20 with `onnxruntime-gpu`; the full benchmark command above records the exact number for the current machine. The setup probe reports the available VRAM (the reference H20 environment exposes about 97,871 MiB per GPU). Do not transfer either number to another GPU without rerunning `benchmark.json`.

## 3. Pre-generated demo

No weights or GPU are needed to inspect the checked-in result. The high-resolution overview panel can be regenerated from the licensed 1024×1024 source portraits:

```bash
python3 scripts/build_holdout_portrait_panel.py
```

The command writes `outputs/new_face_demo_r3/holdout_portraits_male_female.png`; it does not rerun inference or change any evaluation result.

No weights or GPU are needed to inspect the checked-in videos:

- [Female held-out Base vs calibrated GIF](../outputs/new_face_demo_r3/female/female_new_face_base_vs_r3.gif)
- [Male held-out Base vs calibrated GIF](../outputs/new_face_demo_r3/male/male_new_face_base_vs_r3.gif)
- [Female Base MP4](../outputs/new_face_demo_r3/female/expression_r3__base.mp4) · [Female calibrated MP4](../outputs/new_face_demo_r3/female/expression_r3__seed0.mp4)
- [Male Base MP4](../outputs/new_face_demo_r3/male/expression_r3__base.mp4) · [Male calibrated MP4](../outputs/new_face_demo_r3/male/expression_r3__seed0.mp4)

The two sides use the same held-out portrait, audio, seed, and frozen renderer. The comparison isolates motion; it does not claim improved skin texture, lighting, or color.

## 4. Model weights and data

`h20_render_setup.sh` downloads the following into `third_party/JoyVASA/pretrained_weights/`:

| Component | Source | Approximate role |
|---|---|---|
| JoyVASA motion generator and template | `jdh-algo/JoyVASA` | audio to 73-D motion |
| Chinese HuBERT | `TencentGameMate/chinese-hubert-base` | audio features |
| LivePortrait human weights | `KwaiVGI/LivePortrait` | motion to pixels |
| InsightFace `buffalo_l` | official release | face detection/cropping |
| MediaPipe face landmarker | Google-hosted model | video metrics |
| SyncNet model | `nawta/syncnet` on Hugging Face | independent lip-sync evaluation |

The setup downloads about **2.5 GB** of runtime weights, depending on upstream revisions. If the network is unavailable but the repository backup chunks are present, restore the large JoyVASA weights first:

```bash
bash scripts/restore_chunks.sh
```

The restore script verifies `backups/weights_chunks/MANIFEST.sha256`; it does not replace the external download step for missing third-party components.

## 5. Failure diagnosis

| Symptom | Check | Fix |
|---|---|---|
| `CUDA is not available` | `python3 -c "import torch; print(torch.cuda.is_available())"` | Run setup on a CUDA host, or use the CPU mock. |
| `nvidia-smi` not found | `command -v nvidia-smi` | Install the NVIDIA driver/toolkit; the GPU benchmark cannot measure VRAM without it. |
| `missing portrait or audio` | `test -f data/portraits/host_f.png && test -f data/audio/demo/demo_00.wav` | Run from `07-talking-head-motion-opsd` or pass absolute paths. |
| missing `lora.pt` | `find outputs -path '*/lora.pt'` | Restore/check out the tracked adapter artifacts, or run the corresponding training recipe first. |
| renderer weights missing | `find third_party/JoyVASA/pretrained_weights -type f | head` | Re-run `bash scripts/h20_render_setup.sh`; it is safe to re-run. |
| `onnxruntime` provider error | `python3 -c "import onnxruntime as o; print(o.get_available_providers())"` | On GPU use `onnxruntime-gpu` and remove the CPU package; on CPU keep `onnxruntime`. |
| output duration mismatch | inspect `outputs/demo/gpu_r3/manifest.json` | Keep `--seconds` identical for audio trimming and render; delete only the target demo directory and rerun. |
| OOM | `nvidia-smi` and `benchmark.json` | Use `--seconds 1`, a smaller source image, or a GPU with more VRAM; do not silently change the evaluation claim. |

## 6. Before/after claim boundary

The strongest current claim is controlled motion adaptation: on held-out speakers, calibrated expression seeds reduce mean target error from `0.1202` to `0.0229–0.0238` in the registered coefficient metric. The project does **not** claim general perceptual naturalness improvement: the absolute jitter gate failed, absolute lip-sync quality remains unresolved, and a blind human preference result has not been collected.

The real pre-generated GIFs are therefore evidence of a matched visual comparison, not a substitute for the full evaluation protocol in [`RESULTS_R3.md`](RESULTS_R3.md).
