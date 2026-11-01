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


def skin_smooth(bgr, amount=0.45, d=5, sc=30, ss=6):
    """Very light "beauty" pass: smooth only skin-coloured pixels.

    Eyes, teeth, hair and background stay sharp, so the frame keeps looking HD
    while skin reads smoother. amount=0 -> off.
    """
    if amount <= 0:
        return bgr
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    cr, cb = ycrcb[:, :, 1], ycrcb[:, :, 2]
    mask = ((cr >= 133) & (cr <= 178) & (cb >= 77) & (cb <= 132)).astype(np.float32)
    mask = cv2.GaussianBlur(mask, (0, 0), 3.0)[:, :, None]
    sm = cv2.bilateralFilter(bgr, d, sc, ss).astype(np.float32)
    w = amount * mask
    out = bgr.astype(np.float32) * (1.0 - w) + sm * w
    return np.clip(out, 0, 255).astype(np.uint8)


def levels_and_clahe(bgr, target_p2=12.0, target_p98=236.0, clip=2.0,
                     clahe=0.8, contrast=1.08, sat=1.10,
                     target_mean=114.0, gamma=None, highlight=0.80):
    """Stretch, lift the mid-tone to a measured target, then an S-curve for punch.

    Earlier version used CLAHE clipLimit 1.6 and no S-curve; the result measured
    flatter than the input and read as grey. CLAHE is now mild (it only has to
    bring back shadow detail, not carry the contrast) and the contrast comes
    from an S-curve around mid grey, which keeps highlights and shadows planted.

    The gamma step is what actually fixes "too dark": a clip whose stretched mean
    is 96/255 stays dark no matter how much contrast you add, and an S-curve
    around 0.5 makes it *darker* still. gamma is solved once (first frame) and
    reused for every frame so the grading cannot flicker.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32)
    p2, p98 = np.percentile(L, clip), np.percentile(L, 100 - clip)
    if p98 - p2 < 1e-3:
        p2, p98 = float(L.min()), float(L.max()) + 1e-3
    L = (L - p2) * ((target_p98 - target_p2) / (p98 - p2)) + target_p2
    L = np.clip(L, 0, 255)
    if clahe > 0:
        cl = cv2.createCLAHE(clipLimit=clahe, tileGridSize=(8, 8))
        L = cl.apply(L.astype(np.uint8)).astype(np.float32)
    # mid-tone lift (gamma solved on the first frame, then frozen)
    if target_mean is not None:
        if gamma is None:
            m = max(float(L.mean()), 8.0)
            gamma = float(np.clip(np.log(target_mean / 255.0) / np.log(m / 255.0),
                                  0.55, 1.8))
        L = 255.0 * np.clip(L / 255.0, 0, 1) ** gamma
    # soft highlight rolloff: the old hard stretch to 243 blew the highlights out
    if highlight < 1.0:
        knee = 200.0
        hi = np.clip(L - knee, 0, None)
        L = np.where(L > knee, knee + hi * highlight, L)
    # S-curve around 128
    if abs(contrast - 1.0) > 1e-3:
        x = L / 255.0
        x = np.clip(x, 0, 1)
        x = x ** contrast / (x ** contrast + (1 - x) ** contrast + 1e-9)
        L = np.clip(x * 255.0, 0, 255)
    lab[:, :, 0] = L.astype(np.uint8)
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR), gamma


def natural_light(bgr, gamma=None, target_mean=118.0, black_lift=16.0,
                  white_target=240.0, knee=200.0, rolloff=0.80,
                  local=1.6, clarity=0.45, contrast=1.10,
                  sat=1.05, vibrance=0.25, skin=0.35, warm=0.03):
    """Brightness is global; LIGHT FEEL ("光感") is local.

    A picture can sit at the right mean luma and still look dead: flat local
    contrast, crushed blacks, clipped highlights. This pass targets light feel:

      1. black lift  -- shadows stop at `black_lift` instead of 0: air in the shadows.
      2. highlight rolloff -- everything above `knee` is compressed, so highlights
         keep gradient instead of turning into white paper.
      3. local contrast (CLAHE, tile 8x8) -- the "sun through air" effect; this is
         the single biggest contributor to transparency.
      4. clarity -- halo-free micro-contrast: L += c * (L - blur(L, sigma 12)).
         Large radius, so it adds depth without the crunchy unsharp look.
      5. gentle S-curve (1.10) + gamma solved to a measured mean.
      6. saturation kept LOW (1.05) with vibrance: low-chroma pixels are lifted
         more than already-saturated ones, so it reads colourful, not oversaturated.
      7. skin-only smoothing, so clarity does not exaggerate pores.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32)
    p2, p98 = np.percentile(L, 2), np.percentile(L, 98)
    if p98 - p2 < 1e-3:
        p2, p98 = float(L.min()), float(L.max()) + 1e-3
    L = (L - p2) * ((white_target - black_lift) / (p98 - p2)) + black_lift
    L = np.clip(L, 0, 255)
    if target_mean is not None and gamma is None:
        m = max(float(L.mean()), 8.0)
        gamma = float(np.clip(np.log(target_mean / 255.0) / np.log(m / 255.0), 0.55, 1.8))
    if gamma is not None:
        L = 255.0 * np.clip(L / 255.0, 0, 1) ** gamma
    if local > 0:
        cl = cv2.createCLAHE(clipLimit=local, tileGridSize=(8, 8))
        L = cl.apply(np.clip(L, 0, 255).astype(np.uint8)).astype(np.float32)
    if clarity > 0:
        blur = cv2.GaussianBlur(L, (0, 0), 12.0)
        L = L + clarity * (L - blur)
    L = np.clip(L, 0, 255)
    if abs(contrast - 1.0) > 1e-3:
        x = np.clip(L / 255.0, 0, 1)
        L = 255.0 * (x ** contrast / (x ** contrast + (1 - x) ** contrast + 1e-9))
    if rolloff < 1.0:
        hi = np.clip(L - knee, 0, None)
        L = np.where(L > knee, knee + hi * rolloff, L)
    lab[:, :, 0] = np.clip(L, 0, 255).astype(np.uint8)
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
    s_ch = hsv[:, :, 1]
    if vibrance > 0:
        s_ch = np.clip(s_ch * sat * (1.0 + vibrance * (1.0 - s_ch / 255.0)), 0, 255)
    else:
        s_ch = np.clip(s_ch * sat, 0, 255)
    hsv[:, :, 1] = s_ch
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    if warm > 0:
        f = out.astype(np.float32)
        f[:, :, 2] = np.clip(f[:, :, 2] * (1.0 + warm), 0, 255)   # R up
        f[:, :, 0] = np.clip(f[:, :, 0] * (1.0 - warm), 0, 255)   # B down
        out = f.astype(np.uint8)
    return skin_smooth(out, skin), gamma


def unsharp(bgr, amount=0.25, radius=1.8):
    blur = cv2.GaussianBlur(bgr, (0, 0), radius)
    return cv2.addWeighted(bgr, 1 + amount, blur, -amount, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--sr", default="auto", choices=["auto", "none", "edsr", "espcn"])
    ap.add_argument("--sr-below", type=int, default=1024, help="super-resolve if width below this")
    ap.add_argument("--no-levels", action="store_true")
    ap.add_argument("--clahe", type=float, default=0.8)
    ap.add_argument("--contrast", type=float, default=1.08)
    ap.add_argument("--sat", type=float, default=1.10)
    ap.add_argument("--target-mean", type=float, default=114.0,
                    help="solve a gamma lift so the graded clip lands on this mean luma")
    ap.add_argument("--highlight", type=float, default=0.80,
                    help="<1 compresses highlights above 200 so nothing blows out")
    ap.add_argument("--light", action="store_true", default=True,
                    help="use natural_light() (brightness + light-feel pass)")
    ap.add_argument("--local", type=float, default=1.6, help="CLAHE clipLimit: local contrast = light feel")
    ap.add_argument("--clarity", type=float, default=0.30, help="halo-free micro-contrast")
    ap.add_argument("--warm", type=float, default=0.03, help="warm the tone: R up / B down by this fraction")
    ap.add_argument("--sharp", type=float, default=0.25, help="unsharp amount (lower = softer)")
    ap.add_argument("--skin", type=float, default=0.35)
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
    gamma = None
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        if i % 10 == 0:
            before.append((g.mean(), np.percentile(g, 5), np.percentile(g, 95)))
        if not a.no_levels:
            if a.light:
                fr, gamma = natural_light(fr, gamma=gamma, target_mean=a.target_mean,
                                          local=a.local, clarity=a.clarity,
                                          sat=a.sat, skin=a.skin)
            else:
                fr, gamma = levels_and_clahe(fr, clahe=a.clahe, contrast=a.contrast,
                                             sat=a.sat, target_mean=a.target_mean,
                                             gamma=gamma, highlight=a.highlight)
                fr = skin_smooth(fr, a.skin)
        if sr:
            fr = sr.upsample(fr)
        elif upscale != 1.0:
            fr = cv2.resize(fr, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_LANCZOS4)
        fr = unsharp(fr, a.sharp, 1.8)
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
    print(f"[pp] resolution {W}x{H} -> {ow}x{oh}   gamma={gamma:.3f}")

    import subprocess
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-i", a.inp,
                    "-map", "0:v", "-map", "1:a?", "-c:v", "libx264", "-crf", str(a.crf),
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", a.out], check=True)
    os.remove(tmp)
    print(f"[ok] {a.out}")


if __name__ == "__main__":
    main()
