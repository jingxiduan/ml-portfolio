"""Measure the natural motion band of each industry from real public video.

The three personas (finance / real-estate / short-video) must not be defined by
guessed numbers. This script measures, on real human footage, exactly the three
quantities the reward scores -- in the reward's own units:

    low-band pose std (deg)    0.3-3.0 Hz   -> --amp-target-deg
    high-band pose std (deg)   6.0-12.5 Hz  -> --jitter-cap-deg
    blinks per minute          EAR-based    -> sanity check on --blink-target-mean

Units matter. `calibrate_reward_targets.py` reports FFT *power*, which is NOT
what the reward scores; the reward scores the RMS of a Butterworth-bandpassed
signal. So this script re-implements the measurement with the same filter the
reward uses (`src/motion_reward._bandpass` / `eval_naturalness.band_std`), which
is why its finance number can be plugged straight into `--amp-target-deg`.

Industry proxies (documented, not invented labels):
    finance      <- HDTF news-anchor footage: formal, restrained, authoritative.
    real-estate  <- CelebV-HQ clips in the middle of the measured amplitude
                    distribution: conversational, warm, moderate motion.
    short-video  <- CelebV-HQ clips in the upper part of the same distribution:
                    energetic, attention-grabbing -- but still clipped to the
                    human range so it stays natural rather than theatrical.

CelebV-HQ has no industry metadata; the split is by *measured* motion energy,
which is stated explicitly in the output JSON.
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
LP_LO, LP_HI = 0.3, 3.0        # low band  = deliberate head motion ("amplitude")
HP_LO, HP_HI = 6.0, 12.5       # high band = tremor / jitter

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


def band_std(x, lo, hi, fs=FPS):
    """RMS/std of a Butterworth band-passed signal -- same filter as the reward."""
    from scipy.signal import butter, filtfilt
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < 24:
        return 0.0
    ny = fs / 2.0
    b, a = butter(2, [lo / ny, min(hi / ny, 0.99)], btype="band")
    return float(np.std(filtfilt(b, a, x)))


def _ear(fl, ids):
    p = np.array([[fl[i].x, fl[i].y] for i in ids])
    v = np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])
    h = np.linalg.norm(p[0] - p[3])
    return float(v / (2.0 * h + 1e-8))


def measure(path, lm, max_frames=250, ts0=0):
    """Return per-clip motion statistics, or None if the face is not tracked.

    `ts0` carries the timestamp across clips: MediaPipe's VIDEO mode requires
    monotonically increasing timestamps for the *lifetime of one landmarker*,
    so a fresh landmarker would otherwise reject the second clip.
    """
    import mediapipe as mp
    cap = cv2.VideoCapture(path)
    poses, ears, ts, idx = [], [], ts0, 0
    while idx < max_frames:
        ok, fr = cap.read()
        if not ok:
            break
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
        idx += 1
    cap.release()

    if len(poses) < 60:                      # need >= 2.4 s of tracking
        return None, ts
    P = np.asarray(poses)
    T = len(P)
    low = np.mean([band_std(P[:, i], LP_LO, LP_HI) for i in range(3)])
    high = np.mean([band_std(P[:, i], HP_LO, HP_HI) for i in range(3)])

    E = np.asarray(ears)
    thr = np.median(E) - 0.04
    below = E < thr
    blinks = int(np.sum(np.diff(below.astype(int)) == 1))
    return ({"video": os.path.basename(path), "frames": T,
             "low_std_deg": low, "high_std_deg": high,
             "blink_per_min": blinks / (T / FPS) * 60.0,
             "mean_abs_yaw": float(np.mean(np.abs(P[:, 1]))),
             "mean_abs_pitch": float(np.mean(np.abs(P[:, 0])))}, ts)


# ---- quality gate -----------------------------------------------------------
# Raw per-clip measurements contain two failure modes that silently inflate the
# band: (a) profile / off-axis faces, where solvePnP is ill-conditioned, and
# (b) clips whose "motion" is actually tracking noise or camera movement.
# A real human talking to camera moves its head by a couple of degrees at 0.3-3 Hz
# and tremors by ~0.1 deg at 6-12.5 Hz; anything an order of magnitude above that
# is measurement failure, not behaviour. These clips are dropped, and the count is
# reported, rather than being allowed to define the target.
MAX_MEAN_ABS_YAW = 25.0
MAX_MEAN_ABS_PITCH = 25.0
LOW_RANGE = (0.2, 8.0)
MAX_HIGH_STD = 1.0
MAX_BLINK_PER_MIN = 60.0


def quality_ok(r):
    if not (LOW_RANGE[0] <= r["low_std_deg"] <= LOW_RANGE[1]):
        return False, f"low_std={r['low_std_deg']:.2f} outside {LOW_RANGE}"
    if r["high_std_deg"] > MAX_HIGH_STD:
        return False, f"high_std={r['high_std_deg']:.2f} > {MAX_HIGH_STD}"
    if r["mean_abs_yaw"] > MAX_MEAN_ABS_YAW or r["mean_abs_pitch"] > MAX_MEAN_ABS_PITCH:
        return False, f"off-axis yaw={r['mean_abs_yaw']:.0f} pitch={r['mean_abs_pitch']:.0f}"
    if r["blink_per_min"] > MAX_BLINK_PER_MIN:
        return False, f"blink={r['blink_per_min']:.0f}/min > {MAX_BLINK_PER_MIN}"
    return True, ""


def stats(rows, key):
    v = np.array([r[key] for r in rows], dtype=np.float64)
    return {"median": float(np.median(v)), "p25": float(np.percentile(v, 25)),
            "p75": float(np.percentile(v, 75)), "p10": float(np.percentile(v, 10)),
            "p90": float(np.percentile(v, 90)), "mean": float(np.mean(v))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdtf-dir", default=os.path.join(ROOT, "data", "videos_public", "hdtf"))
    ap.add_argument("--celebv-dir", default=os.path.join(ROOT, "data", "videos_public", "celebv"))
    ap.add_argument("--max-frames", type=int, default=250)
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "industry_bands.json"))
    a = ap.parse_args()

    lm = _landmarker()

    clock = [0]  # monotonic MediaPipe timestamp shared across clips

    def sweep(d, label):
        vids = sorted(glob.glob(os.path.join(d, "*.mp4")))
        rows = []
        for v in vids:
            r, ts_end = measure(v, lm, a.max_frames, clock[0])
            clock[0] = ts_end + 1000  # strictly increasing across clips
            if r is None:
                print(f"  [skip] {os.path.basename(v)}: face not tracked", flush=True)
                continue
            r["source"] = label
            rows.append(r)
        print(f"[{label}] {len(rows)}/{len(vids)} clips measured", flush=True)
        return rows

    def gate(rows, label):
        keep = []
        for r in rows:
            ok, why = quality_ok(r)
            if ok:
                keep.append(r)
            else:
                print(f"  [drop] {r['video']}: {why}", flush=True)
        print(f"[{label}] quality gate kept {len(keep)}/{len(rows)}", flush=True)
        return keep

    hdtf = gate(sweep(a.hdtf_dir, "hdtf"), "hdtf")
    celebv = gate(sweep(a.celebv_dir, "celebv"), "celebv")

    if not hdtf and not celebv:
        print("[warn] nothing measured")
        return

    # ---- finance: HDTF news anchors, directly measured -----------------------
    fin = hdtf if hdtf else celebv

    # ---- real-estate / short-video: split CelebV-HQ by measured amplitude ----
    pool = celebv if celebv else hdtf
    pool_sorted = sorted(pool, key=lambda r: r["low_std_deg"])
    n = len(pool_sorted)
    lo_end = max(int(n * 0.45), 1)
    hi_start = min(int(n * 0.65), n - 1)
    mid = pool_sorted[:lo_end] if n < 6 else pool_sorted[int(n * 0.15):lo_end]
    top = pool_sorted[hi_start:] if n < 6 else pool_sorted[hi_start:min(int(n * 0.95), n)]
    if not mid:
        mid = pool_sorted[:max(n // 3, 1)]
    if not top:
        top = pool_sorted[-max(n // 3, 1):]

    groups = {"finance": fin, "realestate": mid, "shortvideo": top}
    out = {"method": "MediaPipe FaceLandmarker + solvePnP; Butterworth band RMS "
                     "(identical filter to src/motion_reward._bandpass)",
           "bands_hz": {"low": [LP_LO, LP_HI], "high": [HP_LO, HP_HI]},
           "proxies": {
               "finance": "HDTF news-anchor footage (formal, restrained)",
               "realestate": "CelebV-HQ, mid measured-amplitude clips (conversational)",
               "shortvideo": "CelebV-HQ, upper measured-amplitude clips (energetic), "
                             "still clipped to the human range",
           },
           "industries": {}}
    for name, rows in groups.items():
        if not rows:
            continue
        out["industries"][name] = {
            "n_clips": len(rows),
            "low_std_deg": stats(rows, "low_std_deg"),
            "high_std_deg": stats(rows, "high_std_deg"),
            "blink_per_min": stats(rows, "blink_per_min"),
            "per_clip": rows,
        }
        s = out["industries"][name]
        print(f"\n[{name}] n={s['n_clips']}")
        print(f"   low  (amp)   median={s['low_std_deg']['median']:.3f} deg"
              f"  band=[{s['low_std_deg']['p25']:.3f}, {s['low_std_deg']['p75']:.3f}]")
        print(f"   high (jitter) median={s['high_std_deg']['median']:.4f} deg"
              f"  band=[{s['high_std_deg']['p25']:.4f}, {s['high_std_deg']['p75']:.4f}]")
        print(f"   blink/min    median={s['blink_per_min']['median']:.1f}")

    out["quality_gate"] = {
        "max_mean_abs_yaw": MAX_MEAN_ABS_YAW, "max_mean_abs_pitch": MAX_MEAN_ABS_PITCH,
        "low_std_range_deg": list(LOW_RANGE), "max_high_std_deg": MAX_HIGH_STD,
        "max_blink_per_min": MAX_BLINK_PER_MIN,
        "hdtf_kept": len(hdtf), "celebv_kept": len(celebv),
    }

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print("\n[ok] wrote", a.out)


if __name__ == "__main__":
    main()
