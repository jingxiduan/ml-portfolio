"""Calibrate what NATURAL head-motion amplitude looks like, from real human video.

Why this exists: OPSD reaches ~21x the base model's low-band pose power, and that
overshoot is what looks uncanny. But no one measured what a real person actually
does, so there was no target to aim at.

This script estimates per-frame head pose (yaw/pitch/roll, degrees) from real
talking-head video with MediaPipe + solvePnP -- the SAME signal the reward uses --
and reports the low-band ("amp") and high-band ("jitter") power, giving an
empirical natural band to compare base / OPSD against.

Writes outputs/natural_amp.json
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
LP, HP = 0.3, 3.0          # low band -> "amp"
JH_LO, JH_HI = 6.0, 12.5   # high band -> "jitter"

MODEL_POINTS = np.array([
    [0.0, 0.0, 0.0], [0.0, -330.0, -65.0],
    [-225.0, 170.0, -135.0], [225.0, 170.0, -135.0],
    [-150.0, -150.0, -125.0], [150.0, -150.0, -125.0],
], dtype=np.float64)
LANDMARK_IDS = [1, 152, 33, 263, 61, 291]


def _landmarker():
    import mediapipe as mp
    opts = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=os.path.join(
            ROOT, "third_party", "mediapipe", "face_landmarker.task")),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1)
    return mp.tasks.vision.FaceLandmarker.create_from_options(opts)


def pose_from_bgr(img, lm, ts_ms):
    import mediapipe as mp
    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts_ms)
    if not res.face_landmarks:
        return None
    fl = res.face_landmarks[0]
    pts = np.array([[fl[i].x * w, fl[i].y * h] for i in LANDMARK_IDS], dtype=np.float64)
    cam = np.array([[w, 0, w / 2.0], [0, w, h / 2.0], [0, 0, 1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(MODEL_POINTS, pts, cam, np.zeros((4, 1)),
                               flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    rmat, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    yaw = np.degrees(np.arctan2(rmat[1, 0], rmat[0, 0]))
    pitch = np.degrees(np.arctan2(-rmat[2, 0], sy))
    roll = np.degrees(np.arctan2(rmat[2, 1], rmat[2, 2]))
    if yaw > 90:
        yaw -= 180
    elif yaw < -90:
        yaw += 180
    return yaw, pitch, roll


def band_power(x, lo, hi, fs):
    n = len(x)
    if n < 8:
        return 0.0
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    spec = np.abs(np.fft.rfft(x - np.mean(x)))
    m = (freqs >= lo) & (freqs <= hi)
    if not m.any():
        return 0.0
    return float(spec[m].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-dir", default=os.path.join(ROOT, "data", "vfhq", "video_original"))
    ap.add_argument("--fallback-dir", default=os.path.join(ROOT, "data", "hdtf", "videos"))
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--max-frames", type=int, default=250)
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "natural_amp.json"))
    a = ap.parse_args()

    import imageio.v3 as iio
    vids = sorted(glob.glob(os.path.join(a.video_dir, "*.mp4")))[: a.n]
    if not vids:
        vids = sorted(glob.glob(os.path.join(a.fallback_dir, "*.mp4")))[: a.n]
    print(f"[calib] {len(vids)} videos")

    lm = _landmarker()
    rows = []
    ts_ms = 0  # must increase monotonically across ALL videos for one landmarker
    for v in vids:
        cap = cv2.VideoCapture(v)
        poses, idx = [], 0
        while idx < a.max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            p = pose_from_bgr(frame, lm, ts_ms)
            ts_ms += int(1000.0 / FPS)
            if p is not None:
                poses.append(p)
            idx += 1
        cap.release()
        if len(poses) < 30:
            print(f"  [skip] {os.path.basename(v)}: only {len(poses)} frames")
            continue
        P = np.asarray(poses)
        amp = np.mean([band_power(P[:, i], LP, HP, FPS) for i in range(3)])
        jit = np.mean([band_power(P[:, i], JH_LO, JH_HI, FPS) for i in range(3)])
        std_low = np.mean([np.std(P[:, i]) for i in range(3)])
        rows.append({"video": os.path.basename(v), "frames": len(poses),
                     "amp_power": amp, "jitter_power": jit,
                     "pose_std_deg": float(std_low)})
        print(f"  {os.path.basename(v):<34} amp={amp:8.2f} jitter={jit:7.3f} "
              f"pose_std={std_low:5.2f}deg")

    if not rows:
        print("[warn] nothing measured")
        return
    amps = [r["amp_power"] for r in rows]
    jits = [r["jitter_power"] for r in rows]
    sds = [r["pose_std_deg"] for r in rows]
    out = {
        "n_videos": len(rows),
        "amp_power": {"mean": float(np.mean(amps)), "median": float(np.median(amps)),
                      "p10": float(np.percentile(amps, 10)), "p90": float(np.percentile(amps, 90))},
        "jitter_power": {"mean": float(np.mean(jits)), "median": float(np.median(jits))},
        "pose_std_deg": {"mean": float(np.mean(sds)), "median": float(np.median(sds)),
                         "p10": float(np.percentile(sds, 10)), "p90": float(np.percentile(sds, 90))},
        "per_video": rows,
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"\n[ok] NATURAL: amp median={np.median(amps):.2f}, "
          f"jitter median={np.median(jits):.3f}, pose_std median={np.median(sds):.2f} deg")
    print("[ok] wrote", a.out)


if __name__ == "__main__":
    main()
