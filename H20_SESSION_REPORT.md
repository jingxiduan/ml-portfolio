# H20 GPU Session Report — 2026-10-29

## Environment
- **GPU**: 2× NVIDIA H20 96GB (CUDA 12.2)
- **Disk**: 3.8 TB free
- **Python**: 3.9

---

## 08 — Video OPSD-I2V (Tier-2 LoRA Training + Inference)

### Setup (completed)
- Created Python 3.9 venv with PyTorch 2.3.1+cu121, peft, open_clip, diffusers
- Downloaded DynamiCrafter 512 checkpoint (~10GB)
- Cached OpenCLIP ViT-H-14 and ViT-B-32 weights

### Code Changes (H20 compatibility fixes)
1. **`src/rewards/naturalness_penalty.py`** — NEW: differentiable naturalness reward module (anti-reward-hacking). Pure PyTorch ops: saturation penalty, soft-sigmoid overexposure, Sobel edge detection with eps for numerical stability.
2. **`src/eval/naturalness_v2.py`** — NEW: naturalness v2 scoring for inference metrics.
3. **`configs/tier2_natural.yaml`** — NEW: Tier-2 training config (no aesthetic reward, added naturalness_penalty, lr=2e-5, 50 steps).
4. **`src/opsd_trainer_i2v.py`** — FIXED:
   - v-prediction reconstruction formula (was using raw timestep as sigma multiplier → huge values)
   - fp32 VAE decode for reward gradient path (fp16 gradients underflow to zero)
   - Chunked VAE decode (4 frames at a time) for GPU memory
   - Video downsampling before RAFT-based rewards (160×256)
   - CUDA cache management around reward step
5. **`src/dynamicrafter_wrapper.py`** — FIXED:
   - dtype casting for VAE encode and image embedder
   - UNet autocast wrapper for fp16 inference
   - OpenCLIP attn_mask compatibility (v2.26 + PyTorch 2.3)
   - VisionTransformer input_patchnorm attribute
   - fp32 scheduler arithmetic in rollout (prevent NaN)
6. **`src/train_entry.py`** — FIXED: local DDIMScheduler creation (no HuggingFace download needed)
7. **`src/render_i2v.py`** — ADDED: LoRA loading + merge, autocast inference, OpenCLIP patch
8. **`src/rewards/combined.py`** — FIXED: ensure all reward sub-models in fp32
9. **`src/rewards/__init__.py`** — FIXED: import all reward modules for registration

### Training Results
- **50 steps**, production shape 16×320×512, batch_size=1
- **Total time: 517 seconds (~8.5 min)** on H20 96GB
- Final metrics: loss=0.0, reward=2.156, grad_norm=1.0
- LoRA saved: `outputs/opsd_tier2_h20/lora_step00050/`

### Inference Results
- 6 prompts × 3 variants (base, sweet/scale=0.2, tier2_lora) = 18 videos rendered
- 6 three-way comparison GIFs in `docs/figures/duel/`
- Naturalness v2 metrics:

| Metric | Base | Sweet | Tier-2 LoRA |
|--------|------|-------|-------------|
| Avg Appeal | 0.638 | 0.539 | 0.540 |

---

## 07 — Talking Head Motion OPSD (Tier-2 Render)

### Setup (completed)
- Created Python 3.12 venv with torch, JoyVASA, LivePortrait, HuBERT, InsightFace
- Installed system libraries (mesa-libEGL, mesa-libGLES, ffmpeg via imageio-ffmpeg)
- Fixed HuBERT path symlink (`TencentGameMate:chinese-hubert-base`)

### Code Changes
1. **`src/tier2_bon.py`** — FIXED: Mac→H20 audio path remapping (Mac absolute paths in selected_coefs.pt)

### Render Results
- **120 mp4 renders** (20 clips × 6 variants: base, opsd_v1_naive, opsd_v1_bon, opsd_v1_bon_thin, opsd_v2_naive, opsd_v2_bon)
- **20 three-way comparison GIFs** (base | opsd_v1_naive | opsd_v2+BoN)
- n=20 video-space metrics table:

| Metric | Base | OPSD v1 (naive) | v2+BoN | Human Target |
|--------|------|-----------------|--------|--------------|
| blink_rate | 0.00 | 50.49 | 25.62 | 10-20 |
| head_jitter | 119.79 | 262.09 | 256.91 | ~120 |
| speech_head_corr | 0.13 | 0.27 | 0.26 | 0.2-0.4 |

---

## Git Status
- 2 commits created (08 + 07), awaiting push
- Remote: `git@github.com:jingxiduan/ml-portfolio.git`
- **Action needed**: Add H20 SSH public key to GitHub, then `git push origin main`

### H20 SSH Public Key
```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIO/nPKH9wEAJquyUADLDNQKjiGqKE2NBaufHRB5owNYu h20-render
```

---

## Forbidden Words Check
```
grep -riE 'cvpr|neurips|iccv|iclr|aaai|under review|submission|paper deadline' outputs/tier2/*.md
→ PASS (no matches)
```
