"""Re-grade an already-rendered gallery gif/mp4 with the natural_light() color pipeline.

Why this exists
----------------
`scripts/build_persona_grid.py` already proved that raw LivePortrait renders need
`natural_light()` (gamma solved per-clip, black lift, local contrast, mild
saturation+vibrance, 3% warm shift) to stop looking like a grey/oversaturated
screencap -- see outputs/persona_grids/*. But that grading was only ever wired
into the persona-grid builder. Every other gallery (industry_gifs_v3,
demo_gifs_frontal, demo_gifs_v4, demos/*, new_face_demo_r3,
expression_render_set_r3) still ships the ungraded raw render, which is the
"看着很假，色彩丑陋" the user is pointing at.

This script applies the exact same grading (same defaults as
build_persona_grid.py's grade_resize) directly to an existing gif or mp4,
re-encoding to the same format/resolution/fps so it's a drop-in replacement.
Grading is applied to the WHOLE frame (including any side-by-side banner), so
both halves of a base-vs-trained comparison get identical treatment -- it
changes color/light only, not the motion being compared.

Usage:
    python scripts/regrade_demo.py IN.gif OUT.gif
    python scripts/regrade_demo.py IN.mp4 OUT.mp4 --warm 0.03 --sat 1.05
"""
import argparse
import os
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from postprocess_video import natural_light  # noqa: E402


def read_frames(path):
    cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    if not frames:
        raise RuntimeError(f"decoded 0 frames from {path}")
    return frames, fps


def grade(frames, **kw):
    gamma = None
    out = []
    for fr in frames:
        fr, gamma = natural_light(fr, gamma=gamma, **kw)
        out.append(fr)
    return out, gamma


def write_video(frames, fps, dst, src_for_audio=None):
    h, w = frames[0].shape[:2]
    tmp = dst + ".tmp.mp4"
    wri = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for fr in frames:
        wri.write(fr)
    wri.release()

    is_gif = dst.lower().endswith(".gif")
    if is_gif:
        n_colors = 128
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-filter_complex",
               f"fps={fps},split[s0][s1];[s0]palettegen=max_colors={n_colors}[p];"
               "[s1][p]paletteuse=dither=none[v]",
               "-map", "[v]", dst]
    else:
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", tmp]
        if src_for_audio:
            cmd += ["-i", src_for_audio, "-map", "0:v", "-map", "1:a?"]
        cmd += ["-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p"]
        if src_for_audio:
            cmd += ["-c:a", "aac", "-shortest"]
        cmd += [dst]
    subprocess.run(cmd, check=True)
    os.remove(tmp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--target-mean", type=float, default=118.0)
    ap.add_argument("--black-lift", type=float, default=16.0)
    ap.add_argument("--local", type=float, default=1.6)
    ap.add_argument("--clarity", type=float, default=0.30)
    ap.add_argument("--sat", type=float, default=1.05)
    ap.add_argument("--vibrance", type=float, default=0.25)
    ap.add_argument("--skin", type=float, default=0.35)
    ap.add_argument("--warm", type=float, default=0.03)
    ap.add_argument("--keep-audio", action="store_true", default=True)
    a = ap.parse_args()

    frames, fps = read_frames(a.inp)
    graded, gamma = grade(
        frames, target_mean=a.target_mean, black_lift=a.black_lift, local=a.local,
        clarity=a.clarity, sat=a.sat, vibrance=a.vibrance, skin=a.skin, warm=a.warm)

    before = np.mean([f.mean() for f in frames])
    after = np.mean([f.mean() for f in graded])
    print(f"[regrade] {os.path.basename(a.inp)}: {len(frames)}f @ {fps:.1f}fps  "
          f"luma {before:5.1f} -> {after:5.1f}  gamma={gamma:.3f}")

    src_audio = a.inp if (a.keep_audio and not a.out.lower().endswith(".gif")) else None
    write_video(graded, fps, a.out, src_for_audio=src_audio)
    print(f"[ok] {a.out}")


if __name__ == "__main__":
    main()
