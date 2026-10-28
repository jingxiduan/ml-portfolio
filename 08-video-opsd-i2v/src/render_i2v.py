"""Render videos with frozen DynamiCrafter from a directory of first frames.

Uses the **real** DynamiCrafter API: ``image_guided_synthesis`` from the vendored
``scripts.evaluation.inference`` module (functionally identical to upstream, we just avoid
having to re-implement the conditioning pipeline).

Expected input layout (produced by src/t2i/generate_first_frames.py):
    frames_dir/
      seed_00/000.png, 001.png, ...
      manifest.json
Output:
    out_dir/
      seed_00/000.mp4, 001.mp4, ...
      manifest.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import List

import torch
import torchvision
import torchvision.transforms as T
from einops import repeat
from omegaconf import OmegaConf
from PIL import Image

from .utils.mps_patch import apply_mps_patches, get_device, recommended_dtype


# ---------------------------------------------------------------------- vendored imports


def _ensure_vendored() -> None:
    """Add vendored DynamiCrafter to sys.path so we can import its modules."""
    third = Path(__file__).resolve().parents[1] / "third_party" / "DynamiCrafter"
    if not third.is_dir():
        raise RuntimeError(
            f"DynamiCrafter not vendored at {third}. "
            "Run `bash scripts/02_download_weights.sh` first."
        )
    s = str(third)
    if s not in sys.path:
        sys.path.insert(0, s)


# ---------------------------------------------------------------------- model loader


def _patch_cond_stage_device(model, device: torch.device) -> None:
    """DynamiCrafter's text/image encoders hard-code ``self.device = 'cuda'``. Overwrite them
    with our real device so ``tokens.to(self.device)`` doesn't go to CUDA."""
    for attr in ("cond_stage_model", "image_proj_model", "embedder"):
        sub = getattr(model, attr, None)
        if sub is None:
            continue
        if hasattr(sub, "device"):
            try:
                sub.device = device
            except AttributeError:
                pass
        # Recurse one level in case of wrapping
        for _, child in sub.named_modules():
            if hasattr(child, "device") and not isinstance(child, torch.nn.Module.__class__):
                try:
                    child.device = device
                except AttributeError:
                    pass


def _load_dc_model(ckpt_path: str, config_path: str, device: torch.device, dtype: torch.dtype):
    """Instantiate DynamiCrafter from its OmegaConf config and load weights."""
    _ensure_vendored()
    from utils.utils import instantiate_from_config  # type: ignore

    config = OmegaConf.load(config_path)
    model_cfg = config.pop("model", OmegaConf.create())
    # Disable gradient checkpointing (we're not training).
    model_cfg["params"]["unet_config"]["params"]["use_checkpoint"] = False
    model = instantiate_from_config(model_cfg)

    # Load ckpt (same logic as upstream `load_model_checkpoint`).
    state = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" in state:
        state = state["state_dict"]
    try:
        model.load_state_dict(state, strict=True)
    except Exception:
        # 256 variant rename trick: framestride_embed -> fps_embedding.
        renamed = OrderedDict()
        for k, v in state.items():
            if "framestride_embed" in k:
                renamed[k.replace("framestride_embed", "fps_embedding")] = v
            else:
                renamed[k] = v
        model.load_state_dict(renamed, strict=True)
    print(f"[DC] weights loaded from {ckpt_path}")

    model.perframe_ae = True            # required for 512 variant
    model.to(device=device, dtype=dtype)
    _patch_cond_stage_device(model, device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# ---------------------------------------------------------------------- synthesis


def _patch_ddim_sampler_device(device: torch.device) -> None:
    """DDIMSampler.register_buffer hard-codes ``torch.device('cuda')``. Override it."""
    _ensure_vendored()
    from lvdm.models.samplers.ddim import DDIMSampler  # type: ignore

    def register_buffer(self, name, attr):
        if isinstance(attr, torch.Tensor):
            # MPS doesn't support float64 — upcast to float32.
            if device.type == "mps" and attr.dtype == torch.float64:
                attr = attr.to(torch.float32)
            if attr.device != device:
                attr = attr.to(device)
        setattr(self, name, attr)
    DDIMSampler.register_buffer = register_buffer


@torch.no_grad()
def _synthesize(
    model,
    prompt: str,
    first_frame_px: torch.Tensor,       # [3, H, W] in [-1, 1]
    num_frames: int = 16,
    ddim_steps: int = 50,
    guidance_scale: float = 7.5,
    fs: int = 8,
    seed: int = 0,
) -> torch.Tensor:
    """Return a video tensor shaped [T, 3, H, W] in [-1, 1] (float32)."""
    _ensure_vendored()
    from scripts.evaluation.inference import image_guided_synthesis  # type: ignore

    torch.manual_seed(seed)

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # Build "videos" tensor: first frame repeated T times, shape [1, 3, T, H, W]
    img = first_frame_px.unsqueeze(1)                          # [3, 1, H, W]
    frame_tensor = repeat(img, "c t h w -> c (r t) h w", r=num_frames)  # [3, T, H, W]
    videos = frame_tensor.unsqueeze(0).to(device=device, dtype=dtype)   # [1, 3, T, H, W]

    # Noise shape: [B, C, T, H/8, W/8]
    H, W = first_frame_px.shape[-2], first_frame_px.shape[-1]
    channels = model.model.diffusion_model.out_channels                 # 4
    noise_shape = [1, channels, num_frames, H // 8, W // 8]

    # Fire. image_guided_synthesis returns [B=1, n_samples=1, C=3, T, H, W] in [-1, 1].
    out = image_guided_synthesis(
        model=model,
        prompts=[prompt],
        videos=videos,
        noise_shape=noise_shape,
        n_samples=1,
        ddim_steps=ddim_steps,
        ddim_eta=1.0,
        unconditional_guidance_scale=guidance_scale,
        cfg_img=None,
        fs=fs,
        text_input=True,
        multiple_cond_cfg=False,
        loop=False,
        interp=False,
        timestep_spacing="uniform",
        guidance_rescale=0.0,
    )
    # out shape: [1, 1, 3, T, H, W]
    vid = out[0, 0]                                           # [3, T, H, W]
    vid = vid.permute(1, 0, 2, 3).contiguous()                # [T, 3, H, W]
    return vid.float().clamp_(-1, 1)


# ---------------------------------------------------------------------- IO


def _save_mp4(video: torch.Tensor, path: Path, fps: int = 8) -> None:
    """video: [T, 3, H, W] in [-1, 1]  →  mp4.  Uses imageio-ffmpeg for portability."""
    import imageio
    v = (video + 1.0) * 127.5
    v = v.to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()   # [T, H, W, 3] uint8
    writer = imageio.get_writer(str(path), fps=fps, codec="libx264", quality=8)
    for f in v:
        writer.append_data(f)
    writer.close()


# ---------------------------------------------------------------------- CLI


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_dir", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--ddim_steps", type=int, default=50)
    ap.add_argument("--guidance_scale", type=float, default=7.5)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ckpt",
                    default="weights/dynamicrafter_512/model.ckpt")
    ap.add_argument("--config",
                    default="third_party/DynamiCrafter/configs/inference_512_v1.0.yaml")
    ap.add_argument("--height", type=int, default=320)
    ap.add_argument("--width", type=int, default=512)
    args = ap.parse_args()

    apply_mps_patches()
    device = get_device()
    dtype = recommended_dtype(device)
    print(f"[render_i2v] device={device}  dtype={dtype}")

    model = _load_dc_model(args.ckpt, args.config, device=device, dtype=dtype)
    _patch_ddim_sampler_device(device)

    # Image preprocess: identical to DynamiCrafter's inference pipeline.
    tf = T.Compose([
        T.Resize(min(args.height, args.width), antialias=True),
        T.CenterCrop((args.height, args.width)),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    with open(args.prompts) as f:
        prompts: List[dict] = json.load(f)
    if args.limit:
        prompts = prompts[: args.limit]

    frames_root = Path(args.frames_dir)
    out_root = Path(args.out_dir)
    manifest = []
    t0 = time.time()

    for seed in args.seeds:
        in_dir = frames_root / f"seed_{seed:02d}"
        out_dir = out_root / f"seed_{seed:02d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, item in enumerate(prompts):
            frame_path = in_dir / f"{i:03d}.png"
            if not frame_path.is_file():
                print(f"[render_i2v] WARN missing {frame_path}; skip")
                continue
            vid_path = out_dir / f"{i:03d}.mp4"
            if vid_path.is_file():
                manifest.append({"index": i, "prompt": item["prompt"], "seed": seed,
                                 "file": str(vid_path.relative_to(out_root))})
                continue
            img = Image.open(frame_path).convert("RGB")
            x0 = tf(img)                                       # [3, H, W] in [-1, 1]
            vid = _synthesize(
                model=model,
                prompt=item["prompt"],
                first_frame_px=x0,
                num_frames=args.num_frames,
                ddim_steps=args.ddim_steps,
                guidance_scale=args.guidance_scale,
                fs=args.fps,
                seed=seed,
            )
            _save_mp4(vid, vid_path, fps=args.fps)
            manifest.append({"index": i, "prompt": item["prompt"], "seed": seed,
                             "file": str(vid_path.relative_to(out_root))})
            elapsed = time.time() - t0
            print(f"[render_i2v] seed={seed} {i+1}/{len(prompts)} "
                  f"({elapsed:.1f}s, avg {elapsed/max(len(manifest),1):.1f}s/vid)")

    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[render_i2v] done. {len(manifest)} videos in {out_root}. "
          f"Total {time.time()-t0:.1f}s.")


if __name__ == "__main__":
    main()
