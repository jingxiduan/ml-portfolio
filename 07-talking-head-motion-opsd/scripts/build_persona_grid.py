"""One industry = TWO different people, base vs that industry's persona, in one frame.

  row 1: person A  base | persona
  row 2: person B  base | persona

People are chosen by scripts/check_persona_clips.py + add_gaze_checks.py
(English speech, not singing, voice==face gender by construction -- portrait and
audio come from the same source clip, verified in outputs/industry_demo_v3/manifest.json,
eyes on the camera). Output is deliberately small so GitHub actually renders it.
"""
import argparse, os, sys
import cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from postprocess_video import levels_and_clahe

FONT = "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"
PANEL = 480
BANNER = 40
PAIRS = {
    "finance":     [("finance_02_p06_17Km-P0IJZY_10", "PERSON 1"),
                    ("finance_01_p03_0tx4o3yXM64_12", "PERSON 2")],
    "realestate":  [("realestate_00_p01_0wLgQQGob1U_13", "PERSON 1"),
                    ("realestate_01_p04_0TvKsVxgbF4_2", "PERSON 2")],
    "shortvideo":  [("shortvideo_02_p08_-2KG4lLGEl0_3", "PERSON 1"),
                    ("shortvideo_01_p05_0wLgQQGob1U_0", "PERSON 2")],
}


def banner(text, w, h):
    img = Image.new("RGB", (w, h), (14, 14, 16))
    d = ImageDraw.Draw(img)
    f = ImageFont.truetype(FONT, int(h * 0.5))
    tb = d.textbbox((0, 0), text, font=f)
    d.text(((w - (tb[2] - tb[0])) / 2 - tb[0], (h - (tb[3] - tb[1])) / 2 - tb[1]),
           text, font=f, fill=(238, 238, 240))
    return np.array(img)[:, :, ::-1]


def grade_resize(path, side, max_frames, gamma=None):
    cap = cv2.VideoCapture(path)
    frames = []
    while len(frames) < max_frames:
        ok, fr = cap.read()
        if not ok:
            break
        fr, gamma = levels_and_clahe(fr, clahe=1.0, contrast=1.18, sat=1.30,
                                     target_mean=128.0, gamma=gamma)
        frames.append(fr)
    cap.release()
    n = min(len(frames), max_frames)
    out = []
    for fr in frames[:n]:
        fr = cv2.resize(fr, (side, side), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((side + BANNER, side, 3), np.uint8)
        canvas[BANNER:] = fr
        out.append(canvas)
    return out, gamma


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vdir", default=os.path.join(ROOT, "outputs", "industry_demo_v3"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "outputs", "persona_grids"))
    ap.add_argument("--frames", type=int, default=100)   # 4 s @25 fps
    ap.add_argument("--gif-frames", type=int, default=36)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    for ind, people in PAIRS.items():
        tiles = []
        for tag, who in people:
            base = os.path.join(a.vdir, ind, tag + "__base.mp4")
            pers = os.path.join(a.vdir, ind, tag + "__%s.mp4" % ind)
            b, _ = grade_resize(base, PANEL, a.frames)
            p, _ = grade_resize(pers, PANEL, a.frames)
            n = min(len(b), len(p))
            if len(tiles) and len(tiles[0][0]) < n:
                n = len(tiles[0][0])
            tiles.append((b[:n], p[:n], who, ind))
            print(f"[{ind}] {tag}: {n} frames (base {len(b)}, persona {len(p)})", flush=True)
        n = min(len(t[0]) for t in tiles)
        lab = [[f"{t[2]} · BASE (untrained)", f"{t[2]} · {t[3].upper()} PERSONA (trained)"] for t in tiles]

        mp4 = os.path.join(a.out_dir, f"{ind}_two_people_base_vs_persona.mp4")
        tmp = mp4 + ".silent.mp4"
        wri = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), 25.0,
                              (PANEL * 2, (PANEL + BANNER) * 2))
        for i in range(n):
            row0 = np.hstack([np.vstack([banner(lab[0][0], PANEL, BANNER), tiles[0][0][i][BANNER:]]),
                              np.vstack([banner(lab[0][1], PANEL, BANNER), tiles[0][1][i][BANNER:]])])
            row1 = np.hstack([np.vstack([banner(lab[1][0], PANEL, BANNER), tiles[1][0][i][BANNER:]]),
                              np.vstack([banner(lab[1][1], PANEL, BANNER), tiles[1][1][i][BANNER:]])])
            wri.write(np.vstack([row0, row1]))
        wri.release()
        import subprocess
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp,
                        "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", mp4], check=True)
        os.remove(tmp)
        print(f"  [mp4] {mp4}", flush=True)

        # small silent GIF for the README
        import subprocess
        gif = os.path.join(a.out_dir, f"{ind}_two_people_base_vs_persona.gif")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", mp4, "-filter_complex",
                        "fps=10,scale=640:-2:flags=lanczos,setsar=1,"
                        "split[s0][s1];[s0]palettegen=max_colors=64[p];[s1][p]paletteuse=dither=none[v]",
                        "-map", "[v]", "-frames:v", str(a.gif_frames), gif], check=True)
        print(f"  [gif] {gif}", flush=True)


if __name__ == "__main__":
    main()
