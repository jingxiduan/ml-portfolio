"""Derive the reward target band from REAL professional speech video.

The reward's parameters must not be guessed. This script measures, on real
news-anchor / lecturing footage (HDTF, 512x512), exactly the quantities the
reward scores:

  motion : pose_std (deg), low-band "amp" power, high-band "jitter" power
  blink  : blinks per minute (EAR-based)
  look   : mean luminance, contrast (std), saturation  -- the "光感/亮度"

The output is the empirical band (median / p10 / p90) that the reward should
aim at, written to outputs/reward_targets.json.

Only "professional narrating" footage is used: news anchors and lecturers
speaking to camera. Theatrical / performance footage is excluded on purpose,
because it inflates the amplitude band and makes the trained model look
unnatural (measured earlier: VFHQ's mixed content gave pose_std median 39.8 deg,
far above the professional band).
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(ROOT, "src")))

FPS = 25.0
LP, HP = 0.3, 3.0
JH_LO, JH_HI = 6.0, 12.5

# MediaPipe eye landmarks for EAR (eye aspect ratio)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

MODEL_POINTS = np.array([
    [0.0, 0.0, 0.0], [0.0, -330.0, -65.0],
    [-225.0, 170.0, -135.0], [225.0, 170.0, -135.0],
    [-150.0, -150.0, -125.0], [150.0, -150.0, -125.0]], dtype=np.float64)
POSE_IDS = [1, 152, 33, 263, 61, 291]


def _landmarker():
    import mediapipe as mp
    opts = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=os.path.join(
            ROOT, "third_party", "mediapipe", "face_landmarker.task")),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1)
    return mp.tasks.vision.FaceLandmarker.create_from_options(opts)


def _ear(fl, ids):
    p = np.array([[fl[i].x, fl[i].y] for i in ids])
    v = np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])
    h = np.linalg.norm(p[0] - p[3])
    return float(v / (2.0 * h + 1e-8))


def band_power(x, lo, hi, fs):
    n = len(x)
    if n < 8:
        return 0.0
    fr = np.fft.rfftfreq(n, d=1.0 / fs)
    sp = np.abs(np.fft.rfft(x - np.mean(x)))
    m = (fr >= lo) & (fr <= hi)
    return float(sp[m].mean()) if m.any() else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-dir", default=os.path.join(ROOT, "data", "hdtf", "videos"))
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--max-frames", type=int, default=250)
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "reward_targets.json"))
    a = ap.parse_args()

    vids = sorted(glob.glob(os.path.join(a.video_dir, "*.mp4")))[: a.n]
    print(f"[target] {len(vids)} professional-speech clips")
    lm = _landmarker()

    rows, ts = [], 0
    for v in vids:
        cap = cv2.VideoCapture(v)
        poses, ears, lum, sat = [], [], [], []
        idx = 0
        while idx < a.max_frames:
            ok, fr = cap.read()
            if not ok:
                break
            import mediapipe as mp
            h, w = fr.shape[:2]
            res = lm.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)), ts)
            ts += int(1000.0 / FPS)
            if res.face_landmarks:
                fl = res.face_landmarks[0]
                ears.append((_ear(fl, LEFT_EYE) + _ear(fl, RIGHT_EYE)) / 2.0)
                pts = np.array([[fl[i].x * w, fl[i].y * h] for i in POSE_IDS], dtype=np.float64)
                cam = np.array([[w, 0, w / 2.0], [0, w, h / 2.0], [0, 0, 1]], dtype=np.float64)
                okp, rvec, _ = cv2.solvePnP(MODEL_POINTS, pts, cam, np.zeros((4, 1)),
                                            flags=cv2.SOLVEPNP_ITERATIVE)
                if okp:
                    rmat, _ = cv2.Rodrigues(rvec)
                    sy = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
                    yaw = np.degrees(np.arctan2(rmat[1, 0], rmat[0, 0]))
                    pitch = np.degrees(np.arctan2(-rmat[2, 0], sy))
                    roll = np.degrees(np.arctan2(rmat[2, 1], rmat[2, 2]))
                    if yaw > 90:
                        yaw -= 180
                    elif yaw < -90:
                        yaw += 180
                    poses.append([pitch, yaw, roll])
            hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
            lum.append(float(hsv[..., 2].mean()))
            sat.append(float(hsv[..., 1].mean()))
            idx += 1
        cap.release()

        if len(poses) < 40:
            print(f"  [skip] {os.path.basename(v)}: {len(poses)} tracked frames")
            continue
        P = np.asarray(poses)
        T = len(P)
        amp = np.mean([band_power(P[:, i], LP, HP, FPS) for i in range(3)])
        jit = np.mean([band_power(P[:, i], JH_LO, JH_HI, FPS) for i in range(3)])
        E = np.asarray(ears)
        # blink = EAR dips below (median - 3*mad) for >=2 consecutive frames
        med = np.median(E)
        thr = med - 0.04
        below = E < thr
        blinks = int(np.sum(np.diff(below.astype(int)) == 1))
        rows.append({
            "video": os.path.basename(v), "frames": T,
            "pose_std_deg": float(np.mean(np.std(P, axis=0))),
            "amp_power": amp, "jitter_power": jit,
            "blink_per_min": blinks / (T / FPS) * 60.0,
            "luminance": float(np.mean(lum)), "contrast": float(np.std(lum)),
            "saturation": float(np.mean(sat)),
        })
        r = rows[-1]
        print(f"  {os.path.basename(v):<32} pose_std={r['pose_std_deg']:5.2f} "
              f"blink/min={r['blink_per_min']:5.1f} lum={r['luminance']:5.1f}")

    if not rows:
        print("[warn] nothing measured")
        return

    def stats(key):
        v = [r[key] for r in rows]
        return {"median": float(np.median(v)), "p10": float(np.percentile(v, 10)),
                "p90": float(np.percentile(v, 90)), "mean": float(np.mean(v))}

    out = {
        "source": "HDTF (news-anchor / professional speech), MediaPipe + solvePnP",
        "n_clips": len(rows),
        "pose_std_deg": stats("pose_std_deg"),
        "amp_power": stats("amp_power"),
        "jitter_power": stats("jitter_power"),
        "blink_per_min": stats("blink_per_min"),
        "luminance": stats("luminance"),
        "contrast": stats("contrast"),
        "saturation": stats("saturation"),
        "per_clip": rows,
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"\n[ok] PROFESSIONAL-SPEECH TARGET BAND (n={len(rows)}):")
    for k in ("pose_std_deg", "amp_power", "jitter_power", "blink_per_min",
              "luminance", "contrast", "saturation"):
        s = out[k]
        print(f"  {k:<15} median={s['median']:8.2f}  band=[{s['p10']:.2f}, {s['p90']:.2f}]")
    print("[ok] wrote", a.out)


if __name__ == "__main__":
    main()
