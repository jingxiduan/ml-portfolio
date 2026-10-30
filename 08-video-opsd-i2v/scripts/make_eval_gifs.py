"""Convert eval mp4 pairs (base vs v2) into side-by-side GIFs for README."""
import json, os, sys
from pathlib import Path
import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent

def mp4_to_frames(path, max_frames=16):
    """Read mp4 and return list of PIL Images, resized to 256px wide."""
    frames_np = list(iio.imiter(str(path)))
    frames = []
    step = max(1, len(frames_np) // max_frames)
    for i in range(0, len(frames_np), step):
        if len(frames) >= max_frames:
            break
        img = Image.fromarray(frames_np[i])
        # Resize to 256px wide, keep aspect
        w, h = img.size
        new_w = 256
        new_h = int(h * new_w / w)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        frames.append(img)
    return frames

def make_sidebyside_gif(left_mp4, right_mp4, out_gif, labels=("Base", "V2 (sigmoid)"), fps=8):
    """Create side-by-side GIF: left=base, right=v2."""
    left_frames = mp4_to_frames(left_mp4)
    right_frames = mp4_to_frames(right_mp4)
    
    n = min(len(left_frames), len(right_frames))
    left_frames = left_frames[:n]
    right_frames = right_frames[:n]
    
    gap = 4  # pixels between left and right
    label_h = 20  # height for label text
    
    combined = []
    for i, (lf, rf) in enumerate(zip(left_frames, right_frames)):
        w = lf.width + gap + rf.width
        h = max(lf.height, rf.height) + label_h
        canvas = Image.new("RGB", (w, h), (255, 255, 255))
        canvas.paste(lf, (0, label_h))
        canvas.paste(rf, (lf.width + gap, label_h))
        
        # Add labels on first frame
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        except:
            font = ImageFont.load_default()
        draw.text((4, 2), labels[0], fill=(255, 80, 80), font=font)
        draw.text((lf.width + gap + 4, 2), labels[1], fill=(80, 180, 80), font=font)
        
        combined.append(np.array(canvas))
    
    duration = 1000 // fps  # ms per frame
    iio.imwrite(str(out_gif), combined, duration=duration, loop=0)
    print(f"  → {out_gif} ({os.path.getsize(out_gif)//1024}KB)")

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
        
        # Get short name from prompt
        if isinstance(prompts, list):
            prompt = prompts[idx] if isinstance(prompts[idx], str) else prompts[idx].get("prompt", "")
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
    
    print(f"\nDone! Created {count} GIFs in {out_dir}")

if __name__ == "__main__":
    main()
