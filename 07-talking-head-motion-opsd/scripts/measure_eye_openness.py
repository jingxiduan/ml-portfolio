""""Are the eyes open?" -- measured, not eyeballed.

The first industry round rendered personas with half-closed eyes: the blink term's
target had been raised (0.18-0.21 vs v4's validated 0.15), so OPSD parked the
expression coefficients near the closed state. This script quantifies eye opening
straight off the rendered video, so the fix can be verified instead of assumed.

For each video it reports, from MediaPipe FaceLandmarker landmarks:
    ear_median          median eye-aspect-ratio over frames
    ear_p10             low percentile -- how closed the eyes get
    closed_frac         fraction of frames with EAR < 0.6 * that video's median
    blinks_per_min      EAR dips below (median - 0.04) lasting >= 2 frames

A "natural" talking head sits near the base model's opening with a human blink rate
(10-20/min). Much lower EAR / much higher blink rate = the training forced the eyes
shut.
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
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


def _landmarker():
    import mediapipe as mp
    opts = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=os.path.join(
            ROOT, "third_party", "mediapipe", "face_landmarker.task")),
        running_mode=mp.tasks.vision.RunningMode.VIDEO, num_faces=1)
    return mp.tasks.vision.FaceLandmarker.create_from_options(opts)


def _ear(fl, ids):
    p = np.array([[fl[i].x, fl[i].y] for i in ids])
    v = np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])
    h = np.linalg.norm(p[0] - p[3])
    return float(v / (2.0 * h + 1e-8))


def measure(path, lm, max_frames=300, clock=None):
    import mediapipe as mp
    cap = cv2.VideoCapture(path)
    ears, ts = [], clock if clock is not None else 0
    idx = 0
    while idx < max_frames:
        ok, fr = cap.read()
        if not ok:
            break
        res = lm.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB,
                     data=cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)), ts)
        ts += int(1000.0 / FPS)
        if res.face_landmarks:
            fl = res.face_landmarks[0]
            ears.append((_ear(fl, LEFT_EYE) + _ear(fl, RIGHT_EYE)) / 2.0)
        idx += 1
    cap.release()
    if len(ears) < 30:
        return None, ts
    E = np.asarray(ears)
    med = float(np.median(E))
    thr = med - 0.04
    below = E < thr
    blinks = int(np.sum(np.diff(below.astype(int)) == 1))
    return {"video": os.path.basename(path), "frames": len(E),
            "ear_median": med, "ear_p10": float(np.percentile(E, 10)),
            "closed_frac": float(np.mean(E < 0.6 * med)),
            "blinks_per_min": blinks / (len(E) / FPS) * 60.0}, ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", nargs="+", required=True,
                    help="glob(s) of mp4s, e.g. 'outputs/industry_demo_v2/finance/*.mp4'")
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "eye_openness.json"))
    a = ap.parse_args()

    lm = _landmarker()
    rows, clock = [], 0
    for pat in a.pattern:
        for p in sorted(glob.glob(os.path.join(ROOT, pat))):
            r, clock = measure(p, lm, clock=clock)
            clock += 1000
            if r is None:
                print(f"  [skip] {os.path.basename(p)}")
                continue
            r["group"] = os.path.basename(os.path.dirname(p))
            rows.append(r)
            print(f"  {r['group']:<24} {r['video'][:52]:<52} "
                  f"EAR={r['ear_median']:.3f} closed={r['closed_frac']:.2f} "
                  f"blink/min={r['blinks_per_min']:.1f}", flush=True)

    if not rows:
        print("[warn] nothing measured")
        return
    out = {"per_video": rows, "by_group": {}}
    for g in sorted(set(r["group"] for r in rows)):
        sub = [r for r in rows if r["group"] == g]
        out["by_group"][g] = {
            "n": len(sub),
            "ear_median": float(np.median([r["ear_median"] for r in sub])),
            "closed_frac": float(np.mean([r["closed_frac"] for r in sub])),
            "blinks_per_min": float(np.median([r["blinks_per_min"] for r in sub])),
        }
    json.dump(out, open(a.out, "w"), indent=1)
    print("\n[by group]")
    for g, s in out["by_group"].items():
        print(f"  {g:<24} EAR={s['ear_median']:.3f} closed={s['closed_frac']:.2f} "
              f"blink/min={s['blinks_per_min']:.1f}")
    print("[ok] wrote", a.out)


if __name__ == "__main__":
    main()
