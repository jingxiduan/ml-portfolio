"""Build side-by-side comparison GIFs for the qualitative gallery.

Pairs base vs v3 (and optionally v2) rendered from the same clip + portrait,
stacks them into one GIF so a reader sees the difference without opening videos.

Also emits a compact contact sheet of portrait thumbnails as a PNG.

Outputs:
  outputs/demo_gifs/<clip>__<portrait>__base_vs_v3.gif
  outputs/demo_gifs/portrait_sheet.png
"""
import argparse
import glob
import os
import subprocess
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FFMPEG = "/usr/local/bin/ffmpeg"


def find_videos(vdir, variants):
    """Return {(clip, portrait): {variant: path}}."""
    out = defaultdict(dict)
    for p in sorted(glob.glob(os.path.join(vdir, "*.mp4"))):
        base = os.path.splitext(os.path.basename(p))[0]
        # tag format: c<NN>_<portrait>__<variant>
        if "__" not in base:
            continue
        head, variant = base.rsplit("__", 1)
        if variants and variant not in variants:
            continue
        parts = head.split("_", 1)
        if len(parts) != 2:
            continue
        clip, portrait = parts
        out[(clip, portrait)][variant] = p
    return out


def make_gif(paths, labels, dst, fps=12, width=360):
    """Stack videos horizontally and write a GIF.

    No drawtext: this container has no fontconfig fonts, so labels are encoded
    in the output filename instead (see main()).
    """
    n = len(paths)
    filt, inputs = [], []
    for i, p in enumerate(paths):
        inputs += ["-i", p]
        filt.append(f"[{i}:v]fps={fps},scale={width}:-1,setsar=1[v{i}]")
    filt.append("".join(f"[v{i}]" for i in range(n)) + f"hstack=inputs={n}[v]")
    r = subprocess.run([FFMPEG, "-y", *inputs, "-filter_complex", ";".join(filt),
                        "-map", "[v]", dst], capture_output=True, text=True)
    if not os.path.exists(dst):
        print(f"    ffmpeg: {r.stderr[-200:]}")
    return os.path.exists(dst)


def portrait_sheet(portrait_dir, dst, cell=200, cols=6):
    imgs = sorted(glob.glob(os.path.join(portrait_dir, "*.png")))[:24]
    if not imgs:
        return
    inputs, filt = [], []
    for i, p in enumerate(imgs):
        inputs += ["-i", p]
        filt.append(f"[{i}:v]scale={cell}:{cell}[t{i}]")
    rows = (len(imgs) + cols - 1) // cols
    # pad last row
    while len(filt) < rows * cols:
        filt.append(f"color=black:s={cell}x{cell}[t{len(filt)}]")
    filt.append("".join(f"[t{i}]" for i in range(rows * cols)) +
                f"xstack=inputs={rows*cols}:layout=" +
                "|".join("0_" + f"{r*cell}" if c == 0 else f"{c*cell}_" + f"{r*cell}"
                         for r in range(rows) for c in range(cols)) + "[out]")
    subprocess.run([FFMPEG, "-y", *inputs, "-filter_complex", ";".join(filt),
                    "-map", "[out]", "-frames:v", "1", dst],
                   capture_output=True, text=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-dir", default=os.path.join(ROOT, "outputs", "hd_compare_public"))
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "demo_gifs"))
    ap.add_argument("--variants", nargs="*", default=["base", "v3"])
    ap.add_argument("--portrait-dir", default=os.path.join(ROOT, "data", "portraits_public"))
    ap.add_argument("--max", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    groups = find_videos(a.video_dir, a.variants)
    n = 0
    for (clip, portrait), variants in sorted(groups.items()):
        if not all(v in variants for v in a.variants):
            continue
        paths = [variants[v] for v in a.variants]
        dst = os.path.join(a.out, f"{clip}__{portrait}__" + "_vs_".join(a.variants) + ".gif")
        if os.path.exists(dst):
            continue
        if make_gif(paths, a.variants, dst):
            print(f"  [gif] {os.path.basename(dst)}")
            n += 1
        if n >= a.max:
            break
    sheet = os.path.join(a.out, "portrait_sheet.png")
    if not os.path.exists(sheet):
        portrait_sheet(a.portrait_dir, sheet)
        if os.path.exists(sheet):
            print(f"  [sheet] {sheet}")
    print(f"[ok] {n} gifs -> {a.out}")


if __name__ == "__main__":
    main()
