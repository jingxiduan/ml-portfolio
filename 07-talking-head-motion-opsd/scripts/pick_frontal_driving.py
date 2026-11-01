"""Pick real clips whose speaker actually FACES THE CAMERA.

Why this exists
---------------
Driving a portrait with a real video's coefficients copies that person's gaze
behaviour too (the renderer's expression term is relative, but gaze *variation*
still transfers). A news anchor reading a teleprompter looks off-axis, so the
digital human ends up staring past the lens -- which is exactly what the user
saw. This scores every candidate clip on frontal-ness so we only drive from
people who are actually looking at the camera.

Reuses the MediaPipe + solvePnP measurement from calibrate_industry_bands.py so
the numbers stay in the same units as everything else in the project.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(ROOT, "src")))
sys.path.insert(0, os.path.abspath(os.path.join(ROOT, "scripts")))

from calibrate_industry_bands import _landmarker, measure  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--max-frames", type=int, default=150)
    ap.add_argument("--max-abs-yaw", type=float, default=15.0)
    ap.add_argument("--max-abs-pitch", type=float, default=15.0)
    ap.add_argument("--min-seconds", type=float, default=9.0)
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "frontal_clips.json"))
    a = ap.parse_args()

    import cv2
    lm = _landmarker()
    rows, ts = [], 0
    paths = []
    for d in a.dirs:
        paths += sorted(glob.glob(os.path.join(d, "*.mp4")))

    for p in paths:
        cap = cv2.VideoCapture(p)
        nf = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()
        if nf / fps < a.min_seconds:
            continue
        try:
            r, ts_out = measure(p, lm, a.max_frames, ts0=ts)
            ts = ts_out + 10
        except Exception as e:
            print(f"  [err] {os.path.basename(p)}: {type(e).__name__}: {e}", flush=True)
            continue
        if r is None:
            continue
        r["clip"] = p
        r["seconds"] = nf / fps
        # "faces the camera" score: mean |yaw| and |pitch| both small
        r["frontal_score"] = float(max(r["mean_abs_yaw"], r["mean_abs_pitch"]))
        rows.append(r)

    rows.sort(key=lambda r: r["frontal_score"])
    keep = [r for r in rows if r["mean_abs_yaw"] <= a.max_abs_yaw
            and r["mean_abs_pitch"] <= a.max_abs_pitch]
    print(f"[pick] {len(rows)} measurable, {len(keep)} pass "
          f"yaw<={a.max_abs_yaw} pitch<={a.max_abs_pitch}", flush=True)
    for r in rows[:12]:
        print(f"  {os.path.basename(r['clip']):34s} yaw={r['mean_abs_yaw']:5.1f} "
              f"pitch={r['mean_abs_pitch']:5.1f} low={r['low_std_deg']:5.2f}deg "
              f"blink={r['blink_per_min']:5.1f}/min  {r['seconds']:.1f}s", flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"criteria": {"max_abs_yaw": a.max_abs_yaw, "max_abs_pitch": a.max_abs_pitch},
               "kept": [{k: v for k, v in r.items() if k != "clip"} | {"clip": r["clip"]}
                        for r in keep],
               "all_scored": len(rows)},
              open(a.out, "w"), indent=1)
    print(f"[ok] -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
