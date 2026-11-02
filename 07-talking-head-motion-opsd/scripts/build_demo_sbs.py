"""Build the README front-page demos: one side-by-side mp4 + one GIF per industry.

Left  = `base`  (frozen pretrained model, audio-only)
Right = `real`  (same face driven by a real human's motion coefficients)

Both sides get an English caption burned in as a PIL banner (this container's
ffmpeg has no drawtext), are stacked horizontally, and written at 1080 lines.

Usage:
    python scripts/build_demo_sbs.py --tag f3 --industry finance \
        --base outputs/real_driven/f3__base_hq.mp4 \
        --real outputs/real_driven/f3__real_hq.mp4
"""
import argparse
import os
import subprocess
import tempfile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FFMPEG = "ffmpeg"
FONT = "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"
SIDE = 1080          # each half is 1080 wide
BANNER = 72          # caption strip on top
GIF_SIDE = 324
GIF_BANNER = 40
GIF_FRAMES = 120       # source frames kept (~4.8 s at 25 fps -> 60 gif frames)
GIF_FPS = 12.5


def banner_img(text, w, h, sub=None):
    img = Image.new("RGB", (w, h), (16, 16, 18))
    d = ImageDraw.Draw(img)
    fs = int(h * 0.42)
    fnt = ImageFont.truetype(FONT, fs)
    tb = d.textbbox((0, 0), text, font=fnt)
    d.text(((w - (tb[2] - tb[0])) / 2 - tb[0], (h - (tb[3] - tb[1])) / 2 - tb[1]),
           text, font=fnt, fill=(240, 240, 240))
    if sub:
        fs2 = int(h * 0.26)
        f2 = ImageFont.truetype(FONT, fs2)
        tb2 = d.textbbox((0, 0), sub, font=f2)
        d.text(((w - (tb2[2] - tb2[0])) / 2 - tb2[0], h - (tb2[3] - tb2[1]) - 4 - tb2[1]),
               sub, font=f2, fill=(150, 150, 158))
    return np.array(img)[:, :, ::-1]  # RGB -> BGR


def label_video(src, dst, text, sub, side=SIDE, bh=BANNER, max_frames=None):
    cap = cv2.VideoCapture(src)
    ow, oh = side, side + bh
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    bar = banner_img(text, ow, bh, sub)
    tmp = dst + ".silent.mp4"
    wri = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (ow, oh))
    n = 0
    while True:
        ok, fr = cap.read()
        if not ok or (max_frames and n >= max_frames):
            break
        fr = cv2.resize(fr, (side, side), interpolation=cv2.INTER_LANCZOS4)
        out = np.empty((oh, ow, 3), np.uint8)
        out[:bh] = bar
        out[bh:] = fr
        wri.write(out)
        n += 1
    cap.release()
    wri.release()
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", tmp, "-i", src,
                    "-map", "0:v", "-map", "1:a?", "-c:v", "libx264", "-crf", "14",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", dst], check=True)
    os.remove(tmp)
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--industry", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--real", required=True)
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "outputs", "demos"))
    ap.add_argument("--label-base", default="BASE")
    ap.add_argument("--label-real", default="REAL-VIDEO DRIVEN")
    ap.add_argument("--sub-base", default="audio-only · v8")
    ap.add_argument("--sub-real", default="human coefficients · v8")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        lb = label_video(a.base, os.path.join(td, "base.mp4"),
                         a.label_base, a.sub_base)
        lr = label_video(a.real, os.path.join(td, "real.mp4"),
                         a.label_real, a.sub_real)
        mp4 = os.path.join(a.out_dir, f"{a.industry}_base_vs_real_1080.mp4")
        subprocess.run([FFMPEG, "-y", "-loglevel", "error",
                        "-i", lb, "-i", lr, "-i", a.base,
                        "-filter_complex", "[0:v][1:v]hstack=inputs=2[v]",
                        "-map", "[v]", "-map", "2:a?",
                        "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", "-shortest", mp4], check=True)
        print(f"  [mp4] {mp4}")

        gb = label_video(a.base, os.path.join(td, "gbase.mp4"), "BASE", None,
                         side=GIF_SIDE, bh=GIF_BANNER, max_frames=GIF_FRAMES)
        gr = label_video(a.real, os.path.join(td, "greal.mp4"), "REAL DRIVEN", None,
                         side=GIF_SIDE, bh=GIF_BANNER, max_frames=GIF_FRAMES)
        gif = os.path.join(a.out_dir, f"{a.industry}_base_vs_real.gif")
        subprocess.run([FFMPEG, "-y", "-loglevel", "error",
                        "-i", gb, "-i", gr,
                        "-filter_complex",
                        f"[0:v][1:v]hstack=inputs=2,fps={GIF_FPS},"
                        "split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse[v]",
                        "-map", "[v]", gif], check=True)
        print(f"  [gif] {gif}")


if __name__ == "__main__":
    main()
