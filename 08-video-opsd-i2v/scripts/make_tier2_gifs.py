"""Generate 3-way GIFs: Base | LoRA alpha=6 (scale~0.2) | LoRA alpha=32 (full)
for Tier-2 I2V LoRA comparison on new eval scenes."""
import json, os
from pathlib import Path
import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent

def mp4_to_frames(path, max_frames=16, width=192):
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

def make_threeway_gif(mp4_a, mp4_b, mp4_c, out_gif, labels, fps=8):
    fa = mp4_to_frames(mp4_a)
    fb = mp4_to_frames(mp4_b)
    fc = mp4_to_frames(mp4_c)
    n = min(len(fa), len(fb), len(fc))
    gap = 3
    label_h = 18
    combined = []
    for i in range(n):
        w = fa[i].width + gap + fb[i].width + gap + fc[i].width
        h = max(fa[i].height, fb[i].height, fc[i].height) + label_h
        canvas = Image.new("RGB", (w, h), (255, 255, 255))
        x = 0
        for frame in [fa[i], fb[i], fc[i]]:
            canvas.paste(frame, (x, label_h))
            x += frame.width + gap
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
        except:
            font = ImageFont.load_default()
        colors = [(200, 60, 60), (60, 160, 60), (60, 60, 200)]
        x = 4
        for j, (label, color) in enumerate(zip(labels, colors)):
            draw.text((x, 2), label, fill=color, font=font)
            x += fa[i].width + gap
        combined.append(np.array(canvas))
    iio.imwrite(str(out_gif), combined, duration=1000//fps, loop=0)
    return os.path.getsize(out_gif)

def make_twoway_gif(mp4_a, mp4_b, out_gif, labels, fps=8):
    fa = mp4_to_frames(mp4_a, width=256)
    fb = mp4_to_frames(mp4_b, width=256)
    n = min(len(fa), len(fb))
    gap = 4
    label_h = 18
    combined = []
    for i in range(n):
        w = fa[i].width + gap + fb[i].width
        h = max(fa[i].height, fb[i].height) + label_h
        canvas = Image.new("RGB", (w, h), (255, 255, 255))
        canvas.paste(fa[i], (0, label_h))
        canvas.paste(fb[i], (fa[i].width + gap, label_h))
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
        except:
            font = ImageFont.load_default()
        draw.text((4, 1), labels[0], fill=(200, 60, 60), font=font)
        draw.text((fa[i].width + gap + 4, 1), labels[1], fill=(60, 160, 60), font=font)
        combined.append(np.array(canvas))
    iio.imwrite(str(out_gif), combined, duration=1000//fps, loop=0)
    return os.path.getsize(out_gif)

def main():
    prompts = json.load(open(ROOT / "data/prompts/tier2_eval_40.json"))
    base_dir = ROOT / "outputs/eval/videos_base/seed_00"
    a6_dir = ROOT / "outputs/eval/videos_v2_alpha6/seed_00"
    a32_dir = ROOT / "outputs/eval/videos_v2_alpha32/seed_00"

    out_3way = ROOT / "docs/figures/tier2_new"
    out_3way.mkdir(parents=True, exist_ok=True)

    count = 0
    for idx in range(10):
        base = base_dir / f"{idx:03d}.mp4"
        a6 = a6_dir / f"{idx:03d}.mp4"
        a32 = a32_dir / f"{idx:03d}.mp4"
        prompt_text = prompts[idx] if isinstance(prompts[idx], str) else str(prompts[idx])
        short = prompt_text[:35].replace(" ", "_").replace(",", "").replace("'", "")

        if base.exists() and a6.exists() and a32.exists():
            out_gif = out_3way / f"tier2_{idx:03d}_{short}.gif"
            print(f"[{idx:03d}] {prompt_text[:50]}...")
            try:
                sz = make_threeway_gif(base, a6, a32, out_gif,
                    labels=["Base", "LoRA a=6", "LoRA a=32"])
                print(f"  -> {out_gif.name} ({sz//1024}KB)")
                count += 1
            except Exception as e:
                print(f"  ERROR: {e}")
        elif base.exists() and a32.exists():
            out_gif = out_3way / f"tier2_{idx:03d}_{short}.gif"
            print(f"[{idx:03d}] 2-way: {prompt_text[:50]}...")
            try:
                sz = make_twoway_gif(base, a32, out_gif, labels=["Base", "V2 LoRA"])
                print(f"  -> {out_gif.name} ({sz//1024}KB)")
                count += 1
            except Exception as e:
                print(f"  ERROR: {e}")

    print(f"\nDone! {count} GIFs in {out_3way}")

if __name__ == "__main__":
    main()
