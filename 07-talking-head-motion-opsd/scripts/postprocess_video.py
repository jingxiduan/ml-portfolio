"""Make rendered demos look like video, not like a grey screencap.

Three defects the user called out, each fixed by a measured step:

  1. too dark / washed out  -> stretch the luminance histogram to a target
     (p2 -> 16, p98 -> 235) instead of blindly adding brightness, and lift
     shadows with CLAHE so dark-side detail comes back ("光感").
  2. not sharp              -> 2x EDSR super-resolution (cv2 dnn_superres) when
     the source is below 1024 px, then an unsharp mask.
  3. flat grey              -> mild saturation lift in LAB.

Every step is optional and the before/after luma statistics are printed, so the
effect is verifiable rather than "looks better, trust me".
"""
import argparse
import os

import cv2
import numpy as np


def levels_and_clahe(bgr, target_p2=16.0, target_p98=235.0, clip=2.0, clahe=1.6):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32)
    p2, p98 = np.percentile(L, clip), np.percentile(L, 100 - clip)
    if p98 - p2 < 1e-3:
        p2, p98 = float(L.min()), float(L.max()) + 1e-3
    L = (L - p2) * ((target_p98 - target_p2) / (p98 - p2)) + target_p2
    lab[:, :, 0] = np.clip(L, 0, 255).astype(np.uint8)
    if clahe > 0:
        cl = cv2.createCLAHE(clipLimit=clahe, tileGridSize=(8, 8))
        lab[:, :, 0] = cl.apply(lab[:, :, 0])
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    # mild saturation lift
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.12, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def unsharp(bgr, amount=0.55, radius=1.4):
    blur = cv2.GaussianBlur(bgr, (0, 0), radius)
    return cv2.addWeighted(bgr, 1 + amount, blur, -amount, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--sr", default="auto", choices=["auto", "none", "edsr", "espcn"])
    ap.add_argument("--sr-below", type=int, default=1024, help="super-resolve if width below this")
    ap.add_argument("--no-levels", action="store_true")
    ap.add_argument("--crf", type=int, default=16)
    a = ap.parse_args()

    cap = cv2.VideoCapture(a.inp)
    W, H = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[pp] {os.path.basename(a.inp)} {W}x{H} {fps:.0f}fps {n}f")

    use_sr = a.sr
    if use_sr == "auto":
        use_sr = "edsr" if W < a.sr_below else "none"
    sr = None
    if use_sr != "none":
        m = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "third_party", "sr",
                         "EDSR_x2.pb" if use_sr == "edsr" else "ESPCN_x2.pb")
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        sr.readModel(m)
        sr.setModel("edsr" if use_sr == "edsr" else "espcn", 2)
        print(f"[pp] super-resolution: {use_sr} x2 ({W}->{W*2})")

    upscale = 2.0 if (sr is None and W < a.sr_below) else 1.0
    ow, oh = (W * 2, H * 2) if sr else (int(W * upscale), int(H * upscale))
    tmp = a.out + ".silent.mp4"
    wri = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (ow, oh))
    before, after = [], []
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        if i % 10 == 0:
            before.append((g.mean(), np.percentile(g, 5), np.percentile(g, 95)))
        if not a.no_levels:
            fr = levels_and_clahe(fr)
        if sr:
            fr = sr.upsample(fr)
        elif upscale != 1.0:
            fr = cv2.resize(fr, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_LANCZOS4)
        fr = unsharp(fr)
        if i % 10 == 0:
            g2 = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            after.append((g2.mean(), np.percentile(g2, 5), np.percentile(g2, 95)))
        wri.write(fr)
        i += 1
    cap.release()
    wri.release()

    b = np.array(before).mean(0)
    af = np.array(after).mean(0)
    print(f"[pp] luma  mean {b[0]:5.1f} -> {af[0]:5.1f}   "
          f"p5 {b[1]:5.1f} -> {af[1]:5.1f}   p95 {b[2]:5.1f} -> {af[2]:5.1f}   "
          f"range {b[2]-b[1]:5.1f} -> {af[2]-af[1]:5.1f}")
    print(f"[pp] resolution {W}x{H} -> {ow}x{oh}")

    import subprocess
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-i", a.inp,
                    "-map", "0:v", "-map", "1:a?", "-c:v", "libx264", "-crf", str(a.crf),
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", a.out], check=True)
    os.remove(tmp)
    print(f"[ok] {a.out}")


if __name__ == "__main__":
    main()
