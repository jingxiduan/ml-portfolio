"""Pick demo portraits that will actually render as natural talking heads.

Round 1 rendered personas with half-closed eyes for two separate reasons, and this
script fixes both by screening candidates on measurements rather than vibes:

  1. the blink reward target was too high (fixed in training: 0.12, penalty 8)
  2. some source portraits simply have small / half-closed eyes to begin with --
     if the source EAR is low, no amount of training produces an open-eyed video

So a candidate must pass all of:
    resolution >= --min-px          (the "HD" requirement)
    |yaw| <= --max-yaw, |pitch| <= --max-pitch   (near frontal, or the demo hides
                                                 exactly the behaviour we measure)
    ear >= --min-ear                (eyes genuinely open in the still)
    smile > 0                       (mouth corners lifted -> reads friendly)

Writes data/portraits_picked/<label>/*.png + data/audio_picked/<label>/*.wav and
data/portraits_picked.json.
"""
import argparse
import glob
import json
import os
import random
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(ROOT, "src")))

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


def _landmarker():
    import mediapipe as mp
    opts = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=os.path.join(
            ROOT, "third_party", "mediapipe", "face_landmarker.task")),
        running_mode=mp.tasks.vision.RunningMode.IMAGE, num_faces=1)
    return mp.tasks.vision.FaceLandmarker.create_from_options(opts)


def _ear(fl, ids):
    p = np.array([[fl[i].x, fl[i].y] for i in ids])
    v = np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])
    h = np.linalg.norm(p[0] - p[3])
    return float(v / (2.0 * h + 1e-8))


def score(img, lm, prep):
    """(yaw, pitch, smile, ear) or None."""
    import mediapipe as mp
    ang = prep.score_portrait(img, lm)          # (yaw, pitch, smile)
    if ang is None:
        return None
    res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                             data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    if not res.face_landmarks:
        return None
    fl = res.face_landmarks[0]
    ear = (_ear(fl, LEFT_EYE) + _ear(fl, RIGHT_EYE)) / 2.0
    return ang[0], ang[1], ang[2], ear


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default=os.path.join(ROOT, "data", "videos_public"))
    ap.add_argument("--scan", type=int, default=220)
    ap.add_argument("--per-industry", type=int, default=2)
    ap.add_argument("--min-px", type=int, default=800)
    ap.add_argument("--max-yaw", type=float, default=15.0)
    ap.add_argument("--max-pitch", type=float, default=15.0)
    ap.add_argument("--min-ear", type=float, default=0.28)
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()

    import importlib.util
    spec = importlib.util.spec_from_file_location("prep", os.path.join(ROOT, "scripts", "prep_industry_demo.py"))
    prep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prep)
    lm = _landmarker()

    vids = sorted(glob.glob(os.path.join(a.videos, "**", "*.mp4"), recursive=True))
    random.Random(a.seed).shuffle(vids)
    cands, seen_src = [], 0
    for v in vids[: a.scan]:
        for tmp in ("/tmp/pick.png", "/tmp/pick.wav"):
            if os.path.exists(tmp):
                os.remove(tmp)
        okp, okw = prep.extract(v, "/tmp/pick.png", "/tmp/pick.wav")
        if not okp:
            continue
        img = cv2.imread("/tmp/pick.png")
        if img is None or min(img.shape[:2]) < a.min_px:
            continue
        s = score(img, lm, prep)
        if s is None:
            continue
        seen_src += 1
        yaw, pitch, smile, ear = s
        if abs(yaw) > a.max_yaw or abs(pitch) > a.max_pitch:
            continue
        if ear < a.min_ear or smile <= 0:
            continue
        rank = min(img.shape[:2]) / 1024.0 + 0.6 * smile + 2.0 * (ear - a.min_ear) \
            - 0.01 * (abs(yaw) + abs(pitch))
        cands.append({"rank": rank, "video": v, "w": img.shape[1], "h": img.shape[0],
                      "yaw": yaw, "pitch": pitch, "smile": smile, "ear": ear})
    cands.sort(key=lambda c: -c["rank"])
    print(f"[pick] scanned {min(a.scan, len(vids))} clips, {seen_src} with a face, "
          f"{len(cands)} passed all gates")

    pdir = os.path.join(ROOT, "data", "portraits_picked")
    adir = os.path.join(ROOT, "data", "audio_picked")
    os.makedirs(pdir, exist_ok=True)
    os.makedirs(adir, exist_ok=True)
    kept = []
    for i, c in enumerate(cands[: a.per_industry * 3]):
        base = f"p{i:02d}_{os.path.splitext(os.path.basename(c['video']))[0]}"
        png = os.path.join(pdir, base + ".png")
        wav = os.path.join(adir, base + ".wav")
        prep.extract(c["video"], png, wav)
        if not os.path.exists(wav):
            continue
        kept.append({"portrait": os.path.relpath(png, ROOT), "audio": os.path.relpath(wav, ROOT),
                     "source_clip": os.path.basename(c["video"]),
                     "px": f"{c['w']}x{c['h']}", "yaw": round(c["yaw"], 1),
                     "pitch": round(c["pitch"], 1), "smile": round(c["smile"], 3),
                     "ear": round(c["ear"], 3)})
        print(f"  {base:<44} {c['w']}x{c['h']} yaw={c['yaw']:5.1f} "
              f"smile={c['smile']:+.3f} EAR={c['ear']:.3f}")

    json.dump({"gates": {"min_px": a.min_px, "max_yaw": a.max_yaw, "max_pitch": a.max_pitch,
                         "min_ear": a.min_ear}, "kept": kept},
              open(os.path.join(ROOT, "data", "portraits_picked.json"), "w"), indent=1)
    print(f"[ok] {len(kept)} portraits -> data/portraits_picked")


if __name__ == "__main__":
    main()
