"""Prepare portrait + driving-audio pairs for the three industry demos.

Portraits are the FIRST FRAME of real public footage, driving audio is that same
clip's real human speech track -- no TTS anywhere.

Crucially, only clips that already passed the motion quality gate in
`outputs/industry_bands.json` are eligible. Those clips were verified to be
near-frontal with head motion inside the human band, so the faces are usable as
talking-head sources and the audio is real speech.

Each candidate is additionally scored for
  - resolution        (bigger is better for an HD demo)
  - smile             (mouth-corner lift, normalised by mouth width)
  - frontal-ness      (|yaw|, |pitch| from solvePnP)
and the top `--per-industry` are kept.

Writes data/portraits_ind/<industry>/*.png, data/audio_ind/<industry>/*.wav
and data/industry_demo_manifest.json.
"""
import argparse
import json
import os
import subprocess
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(ROOT, "src")))

FFMPEG = None  # resolved lazily from imageio-ffmpeg (no system ffmpeg here)

MODEL_POINTS = np.array([
    [0.0, 0.0, 0.0], [0.0, -330.0, -65.0],
    [-225.0, 170.0, -135.0], [225.0, 170.0, -135.0],
    [-150.0, -150.0, -125.0], [150.0, -150.0, -125.0]], dtype=np.float64)
POSE_IDS = [1, 152, 33, 263, 61, 291]
# mouth corners + vertical lip centre, for a scale-free smile score
MOUTH_L, MOUTH_R, LIP_TOP, LIP_BOT = 61, 291, 0, 17


def ffmpeg():
    global FFMPEG
    if FFMPEG is None:
        import imageio_ffmpeg
        FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
    return FFMPEG


def _image_landmarker():
    import mediapipe as mp
    opts = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=os.path.join(
            ROOT, "third_party", "mediapipe", "face_landmarker.task")),
        running_mode=mp.tasks.vision.RunningMode.IMAGE, num_faces=1)
    return mp.tasks.vision.FaceLandmarker.create_from_options(opts)


def score_portrait(img_bgr, lm):
    """Return (yaw, pitch, smile) or None. Smile > 0 means the corners lift."""
    import mediapipe as mp
    h, w = img_bgr.shape[:2]
    res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                             data=cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)))
    if not res.face_landmarks:
        return None
    fl = res.face_landmarks[0]
    pts = np.array([[fl[i].x * w, fl[i].y * h] for i in POSE_IDS], dtype=np.float64)
    cam = np.array([[w, 0, w / 2.0], [0, w, h / 2.0], [0, 0, 1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(MODEL_POINTS, pts, cam, np.zeros((4, 1)),
                               flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    rmat, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    yaw = np.degrees(np.arctan2(rmat[1, 0], rmat[0, 0]))
    pitch = np.degrees(np.arctan2(-rmat[2, 0], sy))
    if yaw > 90:
        yaw -= 180
    elif yaw < -90:
        yaw += 180

    lc = np.array([fl[MOUTH_L].x * w, fl[MOUTH_L].y * h])
    rc = np.array([fl[MOUTH_R].x * w, fl[MOUTH_R].y * h])
    top = np.array([fl[LIP_TOP].x * w, fl[LIP_TOP].y * h])
    bot = np.array([fl[LIP_BOT].x * w, fl[LIP_BOT].y * h])
    width = float(np.linalg.norm(rc - lc)) + 1e-6
    centre_y = 0.5 * (top[1] + bot[1])
    corner_y = 0.5 * (lc[1] + rc[1])
    smile = (centre_y - corner_y) / width      # > 0 => corners above centre => smiling
    return float(yaw), float(pitch), float(smile)


def extract(video, png, wav):
    ff = ffmpeg()
    os.makedirs(os.path.dirname(png), exist_ok=True)
    os.makedirs(os.path.dirname(wav), exist_ok=True)
    if not os.path.exists(png):
        subprocess.run([ff, "-y", "-loglevel", "error", "-i", video,
                        "-vf", "select=eq(n\\,0)", "-vframes", "1", "-q:v", "2", png],
                       capture_output=True)
    if not os.path.exists(wav):
        subprocess.run([ff, "-y", "-loglevel", "error", "-i", video, "-vn",
                        "-ac", "1", "-ar", "16000", "-f", "wav", wav],
                       capture_output=True)
    return os.path.exists(png), os.path.exists(wav)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bands", default=os.path.join(ROOT, "outputs", "industry_bands.json"))
    ap.add_argument("--hdtf-dir", default=os.path.join(ROOT, "data", "videos_public", "hdtf"))
    ap.add_argument("--celebv-dir", default=os.path.join(ROOT, "data", "videos_public", "celebv"))
    ap.add_argument("--portrait-dir", default=os.path.join(ROOT, "data", "portraits_ind"))
    ap.add_argument("--audio-dir", default=os.path.join(ROOT, "data", "audio_ind"))
    ap.add_argument("--per-industry", type=int, default=4)
    ap.add_argument("--max-yaw", type=float, default=20.0)
    ap.add_argument("--max-pitch", type=float, default=20.0)
    a = ap.parse_args()

    bands = json.load(open(a.bands))
    src_dir = {"finance": a.hdtf_dir, "realestate": a.celebv_dir, "shortvideo": a.celebv_dir}
    lm = _image_landmarker()

    out = {"per_industry": a.per_industry, "industries": {}}
    for ind, block in bands["industries"].items():
        cands = []
        for row in block.get("per_clip", []):
            name = row["video"]
            vpath = os.path.join(src_dir.get(ind, a.celebv_dir), name)
            if not os.path.exists(vpath):
                continue
            base = f"{ind}_{os.path.splitext(name)[0]}"
            png = os.path.join(a.portrait_dir, ind, base + ".png")
            wav = os.path.join(a.audio_dir, ind, base + ".wav")
            okp, okw = extract(vpath, png, wav)
            if not (okp and okw):
                continue
            img = cv2.imread(png)
            if img is None:
                continue
            sc = score_portrait(img, lm)
            if sc is None:
                continue
            yaw, pitch, smile = sc
            if abs(yaw) > a.max_yaw or abs(pitch) > a.max_pitch:
                continue
            # prefer HD, frontal, and smiling
            rank = min(img.shape[1], img.shape[0]) / 1024.0 + 0.6 * max(smile, 0.0) \
                - 0.01 * (abs(yaw) + abs(pitch))
            cands.append({"portrait": os.path.relpath(png, ROOT),
                          "audio": os.path.relpath(wav, ROOT),
                          "video": name, "w": img.shape[1], "h": img.shape[0],
                          "yaw": round(yaw, 1), "pitch": round(pitch, 1),
                          "smile": round(smile, 3), "rank": round(rank, 3)})
        cands.sort(key=lambda c: -c["rank"])
        keep = cands[: a.per_industry]
        out["industries"][ind] = keep
        print(f"[{ind}] {len(keep)}/{len(cands)} portraits kept")
        for c in keep:
            print(f"   {os.path.basename(c['portrait']):<44} {c['w']}x{c['h']} "
                  f"yaw={c['yaw']:5.1f} smile={c['smile']:+.3f}")

    mpath = os.path.join(ROOT, "data", "industry_demo_manifest.json")
    json.dump(out, open(mpath, "w"), indent=1)
    print("[ok] wrote", mpath)


if __name__ == "__main__":
    main()
