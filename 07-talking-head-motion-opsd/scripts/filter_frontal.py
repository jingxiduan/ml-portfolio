"""Keep only near-frontal portraits so the demos show the face, not a profile.

Estimates head pose (yaw / pitch / roll) with MediaPipe FaceLandmarker + solvePnP
and keeps images whose |yaw| and |pitch| are within a threshold.

Writes data/portraits_frontal/*.png and a json report of the angles.
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

# Approximate 3-D face model points matching MediaPipe's canonical face mesh.
MODEL_POINTS = np.array([
    [0.0, 0.0, 0.0],        # nose tip (1)
    [0.0, -330.0, -65.0],   # chin (152)
    [-225.0, 170.0, -135.0],  # left eye outer (33)
    [225.0, 170.0, -135.0],   # right eye outer (263)
    [-150.0, -150.0, -125.0],  # left mouth (61)
    [150.0, -150.0, -125.0],   # right mouth (291)
], dtype=np.float64)
LANDMARK_IDS = [1, 152, 33, 263, 61, 291]


def _image_landmarker():
    """video_metrics._landmarker() is configured in VIDEO mode, which rejects
    single images. Build our own IMAGE-mode landmarker for still portraits."""
    import mediapipe as mp
    base = mp.tasks.BaseOptions
    opts = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=base(model_asset_path=os.path.join(
            ROOT, "third_party", "mediapipe", "face_landmarker.task")),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1,
    )
    return mp.tasks.vision.FaceLandmarker.create_from_options(opts)


def pose_angles(img_bgr, landmarker=None):
    """Return (yaw, pitch, roll) in degrees, or None if no face."""
    import mediapipe as mp
    lm = landmarker or _image_landmarker()
    h, w = img_bgr.shape[:2]
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    res = lm.detect(mp_image)
    if not res.face_landmarks:
        return None
    fl = res.face_landmarks[0]
    pts = []
    for i in LANDMARK_IDS:
        p = fl[i]
        pts.append([p.x * w, p.y * h])
    image_points = np.array(pts, dtype=np.float64)
    focal = w
    cam = np.array([[focal, 0, w / 2.0], [0, focal, h / 2.0], [0, 0, 1]], dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(MODEL_POINTS, image_points, cam,
                                  np.zeros((4, 1), dtype=np.float64), flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    rmat, _ = cv2.Rodrigues(rvec)
    # Standard Euler extraction (RQDecomp3x3 uses a different convention and
    # produced pitch near -180, which dropped every image).
    sy = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    if sy > 1e-6:
        roll = np.degrees(np.arctan2(rmat[2, 1], rmat[2, 2]))
        pitch = np.degrees(np.arctan2(-rmat[2, 0], sy))
        yaw = np.degrees(np.arctan2(rmat[1, 0], rmat[0, 0]))
    else:
        roll = np.degrees(np.arctan2(-rmat[1, 2], rmat[1, 1]))
        pitch = np.degrees(np.arctan2(-rmat[2, 0], sy))
        yaw = 0.0
    # solvePnP with this point convention reports some frontal faces as yaw ~±180
    # (a sign flip), so wrap into [-90, 90] before thresholding.
    if yaw > 90.0:
        yaw -= 180.0
    elif yaw < -90.0:
        yaw += 180.0
    return float(yaw), float(pitch), float(roll)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", default=os.path.join(ROOT, "data", "portraits_public"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "data", "portraits_frontal"))
    ap.add_argument("--max-yaw", type=float, default=15.0)
    ap.add_argument("--max-pitch", type=float, default=15.0)
    ap.add_argument("--report", default=os.path.join(ROOT, "outputs", "frontal_filter.json"))
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    imgs = sorted(glob.glob(os.path.join(a.src_dir, "*.png")))
    report, kept = [], []
    for p in imgs:
        img = cv2.imread(p)
        if img is None:
            continue
        ang = pose_angles(img)
        if ang is None:
            report.append({"file": os.path.basename(p), "status": "no_face"})
            continue
        yaw, pitch, roll = ang
        ok = abs(yaw) <= a.max_yaw and abs(pitch) <= a.max_pitch
        report.append({"file": os.path.basename(p), "yaw": round(yaw, 1),
                       "pitch": round(pitch, 1), "roll": round(roll, 1),
                       "kept": bool(ok)})
        if ok:
            import shutil
            dst = os.path.join(a.out_dir, os.path.basename(p))
            shutil.copy2(p, dst)
            kept.append(dst)
        print(f"  {os.path.basename(p):<34} yaw={yaw:6.1f} pitch={pitch:6.1f} "
              f"{'KEEP' if ok else 'drop'}")

    os.makedirs(os.path.dirname(a.report), exist_ok=True)
    with open(a.report, "w") as f:
        json.dump({"max_yaw": a.max_yaw, "max_pitch": a.max_pitch,
                   "n_input": len(imgs), "n_kept": len(kept), "per_file": report}, f, indent=1)
    print(f"[ok] kept {len(kept)}/{len(imgs)} frontal portraits -> {a.out_dir}")


if __name__ == "__main__":
    main()
