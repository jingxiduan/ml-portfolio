"""Re-process ALL GIFs to use a single global palette.
Eliminates inter-frame color flashing caused by per-frame quantization.
Also applies mild temporal smoothing to reduce brightness spikes."""
import os, glob
from pathlib import Path
from PIL import Image
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def load_gif_frames(gif_path):
    """Load all frames from a GIF as RGB PIL images."""
    g = Image.open(gif_path)
    frames = []
    try:
        while True:
            frames.append(g.convert("RGB").copy())
            g.seek(g.tell() + 1)
    except EOFError:
        pass
    # Get duration
    g.seek(0)
    duration = g.info.get("duration", 125)
    return frames, duration


def temporal_smooth_brightness(frames, strength=0.3):
    """Mild temporal smoothing to reduce sudden brightness spikes between frames.
    Blends each frame slightly toward the running average brightness."""
    if len(frames) < 2:
        return frames

    smoothed = []
    arrays = [np.array(f, dtype=np.float32) for f in frames]

    # Compute per-frame mean brightness
    means = [a.mean() for a in arrays]
    # Running average
    avg = means[0]

    for i, (arr, mean_b) in enumerate(zip(arrays, means)):
        if i == 0:
            smoothed.append(frames[i])
            continue
        avg = avg * 0.7 + mean_b * 0.3
        # If this frame is much brighter than running avg, pull it back
        if mean_b > avg * 1.15:  # >15% brighter = spike
            ratio = avg / mean_b
            # Blend toward average brightness
            adj = arr * (1.0 - strength + strength * ratio)
            adj = np.clip(adj, 0, 255).astype(np.uint8)
            smoothed.append(Image.fromarray(adj))
        elif mean_b < avg * 0.85:  # >15% darker = dip
            ratio = avg / mean_b
            adj = arr * (1.0 - strength + strength * ratio)
            adj = np.clip(adj, 0, 255).astype(np.uint8)
            smoothed.append(Image.fromarray(adj))
        else:
            smoothed.append(frames[i])

    return smoothed


def save_gif_global_palette(frames_rgb, out_path, duration=125):
    """Save frames with a single global palette -- no flicker."""
    if not frames_rgb:
        return 0

    # Build global palette from a representative sample of frames
    # Combine pixels from first, middle, and last frame for better coverage
    sample_indices = [0, len(frames_rgb) // 2, len(frames_rgb) - 1]
    sample_frames = [frames_rgb[i] for i in sample_indices if i < len(frames_rgb)]

    # Concatenate samples vertically for palette generation
    w = sample_frames[0].width
    total_h = sum(f.height for f in sample_frames)
    combined = Image.new("RGB", (w, total_h))
    y = 0
    for f in sample_frames:
        combined.paste(f, (0, y))
        y += f.height

    palette_img = combined.quantize(colors=256, method=Image.Quantize.MEDIANCUT)

    quantized = []
    for fr in frames_rgb:
        q = fr.quantize(palette=palette_img, dither=Image.Dither.FLOYDSTEINBERG)
        quantized.append(q)

    quantized[0].save(
        str(out_path),
        save_all=True,
        append_images=quantized[1:],
        duration=duration,
        loop=0,
        optimize=False,
    )
    return os.path.getsize(out_path)


def process_gif(gif_path):
    """Load, smooth, re-palette, save."""
    frames, duration = load_gif_frames(gif_path)
    if not frames:
        print(f"  SKIP (no frames): {gif_path}")
        return

    old_size = os.path.getsize(gif_path)

    # Temporal brightness smoothing
    frames = temporal_smooth_brightness(frames, strength=0.35)

    # Re-save with global palette
    new_size = save_gif_global_palette(frames, gif_path, duration=duration)

    pct = (new_size - old_size) / old_size * 100 if old_size > 0 else 0
    print(f"  {os.path.basename(gif_path)}: {old_size//1024}KB -> {new_size//1024}KB ({pct:+.0f}%)")


def main():
    gif_dirs = [
        ROOT / "docs/figures/duel",
        ROOT / "docs/figures/tier2_new",
        ROOT / "docs/figures/eval_compare",
        ROOT / "docs/figures",  # loose GIFs in root
    ]

    total = 0
    for d in gif_dirs:
        gifs = sorted(d.glob("*.gif")) if d.exists() else []
        if not gifs:
            continue
        print(f"\n=== {d.relative_to(ROOT)} ({len(gifs)} GIFs) ===")
        for g in gifs:
            process_gif(g)
            total += 1

    print(f"\nDone! Fixed {total} GIFs total.")


if __name__ == "__main__":
    main()
