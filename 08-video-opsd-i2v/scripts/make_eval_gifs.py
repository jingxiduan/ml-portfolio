"""Convert eval mp4 pairs (base vs v2) into side-by-side GIFs for README.
Uses a global palette to prevent inter-frame flickering."""
import json, os
from pathlib import Path
import imageio.v3 as iio
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent


def mp4_to_frames(path, max_frames=16, width=256):
    frames_np = list(iio.imiter(str(path)))
    frames = []
    step = max(1, len(frames_np) // max_frames)
    for i in range(0, len(frames_np), step):
        if len(frames) >= max_frames:
            break
        img = Image.fromarray(frames_np[i])
        w, h = img.size
        new_h = int(h * width / w)
        img = img.resize((width, new_h), Image.LANCZOS)
        frames.append(img)
    return frames


def save_gif_global_palette(frames_rgb, out_path, fps=8):
    """Save list of RGB PIL images as GIF with a single global palette (no flicker)."""
    # Quantize first frame to get the global palette
    first_q = frames_rgb[0].quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    palette = first_q.getpalette()

    quantized = []
    for fr in frames_rgb:
        q = fr.quantize(palette=first_q, dither=Image.Dither.FLOYDSTEINBERG)
        quantized.append(q)

    duration_ms = 1000 // fps
    quantized[0].save(
        str(out_path),
        save_all=True,
        append_images=quantized[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    return os.path.getsize(out_path)


def make_sidebyside_gif(left_mp4, right_mp4, out_gif, labels=("Base", "V2 (sigmoid)"), fps=8):
    left_frames = mp4_to_frames(left_mp4)
    right_frames = mp4_to_frames(right_mp4)
    n = min(len(left_frames), len(right_frames))

    gap = 4
    label_h = 20
    combined = []
    for lf, rf in zip(left_frames[:n], right_frames[:n]):
        w = lf.width + gap + rf.width
        h = max(lf.height, rf.height) + label_h
        canvas = Image.new("RGB", (w, h), (30, 30, 30))
        canvas.paste(lf, (0, label_h))
        canvas.paste(rf, (lf.width + gap, label_h))
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        except Exception:
            font = ImageFont.load_default()
        draw.text((4, 2), labels[0], fill=(255, 100, 100), font=font)
        draw.text((lf.width + gap + 4, 2), labels[1], fill=(100, 220, 100), font=font)
        combined.append(canvas)

    sz = save_gif_global_palette(combined, out_gif, fps=fps)
    print(f"  -> {out_gif.name} ({sz // 1024}KB)")


def main():
    prompts = json.load(open(ROOT / "data/prompts/tier2_eval_40.json"))
    base_dir = ROOT / "outputs/eval/videos_base/seed_00"
    v2_dir = ROOT / "outputs/eval/videos_v2_nat_sigmoid/seed_00"
    out_dir = ROOT / "docs/figures/eval_compare"
    out_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for idx in range(40):
        base_mp4 = base_dir / f"{idx:03d}.mp4"
        v2_mp4 = v2_dir / f"{idx:03d}.mp4"
        if not base_mp4.exists() or not v2_mp4.exists():
            continue

        if isinstance(prompts, list):
            p = prompts[idx]
            prompt = p if isinstance(p, str) else p.get("prompt", str(p))
        else:
            prompt = str(list(prompts.values())[idx]) if idx < len(prompts) else ""

        short = prompt[:40].replace(" ", "_").replace(",", "").replace("'", "")
        out_gif = out_dir / f"eval_{idx:03d}_{short}.gif"
        print(f"[{idx:03d}] {prompt[:50]}...")
        try:
            make_sidebyside_gif(base_mp4, v2_mp4, out_gif)
            count += 1
        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\nDone! {count} GIFs in {out_dir}")


if __name__ == "__main__":
    main()
