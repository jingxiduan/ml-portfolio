"""Engine-agnostic motion metrics for talking-head videos.

Works on any rendered mp4 (JoyVASA, another vendor's engine, a real recording), so the
same feature vector can describe any engine/config for routing. Uses MediaPipe
FaceLandmarker: 52 ARKit-style blendshapes (blinks, gaze, jaw) + a 4x4 facial
transformation matrix (head pose), plus the audio track for speech coupling.

Features (all per clip):
  head_pitch_std / head_yaw_std / head_roll_std    degrees, how much the head moves at all
  head_speed_mean                                  deg/s, mean angular speed
  head_jitter                                      deg/s^2 high-frequency energy (lower = steadier)
  nod_rate                                         pitch oscillations per second in the 0.5-3 Hz band
  speech_head_corr                                 corr(audio loudness, head speed), "moves when talking"
  still_ratio                                      fraction of speaking time with near-still head
  blink_rate                                       blinks per minute (humans ~10-20)
  gaze_std                                         std of horizontal+vertical gaze, "eyes not frozen"
  gaze_shift_rate                                  saccade-like gaze shifts per second
  jaw_open_mean / jaw_audio_corr                   mouth activity and its coupling to audio
"""
import json
import os
import subprocess
import tempfile
from typing import Dict

import cv2
import librosa
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, savgol_filter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MP_MODEL = os.path.join(ROOT, "third_party", "mediapipe", "face_landmarker.task")

_LANDMARKER = None


def _landmarker():
    global _LANDMARKER
    if _LANDMARKER is None:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        opts = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MP_MODEL),
            running_mode=vision.RunningMode.VIDEO,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1,
        )
        _LANDMARKER = vision.FaceLandmarker.create_from_options(opts)
    return _LANDMARKER


def _rot_to_euler_deg(R: np.ndarray):
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    pitch = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    yaw = np.degrees(np.arctan2(-R[2, 0], sy))
    roll = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    return pitch, yaw, roll


def extract_tracks(video_path: str) -> Dict[str, np.ndarray]:
    import mediapipe as mp

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    rows, idx = [], 0
    # a fresh landmarker per video keeps the VIDEO-mode timestamps monotonic
    global _LANDMARKER
    _LANDMARKER = None
    lm = _landmarker()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        res = lm.detect_for_video(img, int(idx * 1000 / fps))
        idx += 1
        if not res.face_blendshapes:
            rows.append(None)
            continue
        bs = {c.category_name: c.score for c in res.face_blendshapes[0]}
        p, y, r = _rot_to_euler_deg(np.array(res.facial_transformation_matrixes[0])[:3, :3])
        gaze_h = (bs["eyeLookOutLeft"] - bs["eyeLookInLeft"] + bs["eyeLookInRight"] - bs["eyeLookOutRight"]) / 2
        gaze_v = (bs["eyeLookUpLeft"] + bs["eyeLookUpRight"] - bs["eyeLookDownLeft"] - bs["eyeLookDownRight"]) / 2
        blink = (bs["eyeBlinkLeft"] + bs["eyeBlinkRight"]) / 2
        rows.append([p, y, r, blink, gaze_h, gaze_v, bs["jawOpen"]])
    cap.release()
    valid = [r for r in rows if r is not None]
    if not valid:
        raise RuntimeError(f"no face found in {video_path}")
    arr = np.array([r if r is not None else [np.nan] * 7 for r in rows], dtype=np.float64)
    for j in range(arr.shape[1]):  # fill sporadic misses
        col = arr[:, j]
        bad = np.isnan(col)
        if bad.any():
            col[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad), col[~bad])
    names = ["pitch", "yaw", "roll", "blink", "gaze_h", "gaze_v", "jaw"]
    out = {n: arr[:, i] for i, n in enumerate(names)}
    out["fps"] = np.array(fps)
    out["detect_rate"] = np.array(len(valid) / len(rows))
    return out


def audio_envelope(video_or_audio: str, n_frames: int, fps: float) -> np.ndarray:
    with tempfile.TemporaryDirectory() as td:
        wav = os.path.join(td, "a.wav")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", video_or_audio, "-ac", "1", "-ar", "16000", wav],
                       check=True)
        y, sr = librosa.load(wav, sr=16000)
    hop = int(sr / fps)
    rms = librosa.feature.rms(y=y, frame_length=hop * 2, hop_length=hop)[0]
    rms = np.interp(np.linspace(0, len(rms) - 1, n_frames), np.arange(len(rms)), rms)
    return rms


def _bandpass(x, fps, lo, hi):
    b, a = butter(2, [lo / (fps / 2), hi / (fps / 2)], btype="band")
    return filtfilt(b, a, x) if len(x) > 3 * max(len(a), len(b)) else x - x.mean()


def _corr(a, b):
    if a.std() < 1e-8 or b.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def compute_features(tr: Dict[str, np.ndarray], env: np.ndarray) -> Dict[str, float]:
    fps = float(tr["fps"])
    pyr_raw = np.stack([tr["pitch"], tr["yaw"], tr["roll"]], 1)
    pyr_raw = pyr_raw - np.median(pyr_raw, 0)
    # landmark tracking adds ~0.2 deg/frame noise; differentiate a smoothed track instead
    win = 7 if len(pyr_raw) > 7 else (len(pyr_raw) // 2) * 2 - 1
    pyr = savgol_filter(pyr_raw, win, 2, axis=0) if win >= 3 else pyr_raw
    vel = np.diff(pyr, axis=0) * fps
    speed = np.linalg.norm(vel, axis=1)
    dur = len(pyr) / fps

    pitch_bp = _bandpass(pyr[:, 0], fps, 0.5, 3.0)
    nod_peaks, _ = find_peaks(pitch_bp, prominence=0.6, distance=int(fps * 0.25))
    acc = np.diff(vel, axis=0) * fps

    env_s = env[1:]
    speaking = env_s > (0.3 * np.percentile(env_s, 95))
    still = speed < 3.0
    blink = tr["blink"]
    blink_peaks, _ = find_peaks(blink, height=max(0.35, float(np.median(blink)) + 0.2), prominence=0.2,
                                distance=int(fps * 0.15))
    gaze = np.stack([tr["gaze_h"], tr["gaze_v"]], 1)
    gaze_speed = np.linalg.norm(np.diff(gaze, axis=0), axis=1) * fps
    gaze_shifts, _ = find_peaks(gaze_speed, height=1.5, distance=int(fps * 0.2))

    # head motion typically trails speech energy by 100-300 ms; take the best lag in [0, 400] ms
    env_sm = np.convolve(env_s, np.ones(5) / 5, mode="same")
    max_lag = int(0.4 * fps)
    speech_head = max(_corr(env_sm[: len(env_sm) - k], speed[k:]) for k in range(max_lag + 1))

    return {
        "duration_s": dur,
        "detect_rate": float(tr["detect_rate"]),
        "head_pitch_std": float(pyr[:, 0].std()),
        "head_yaw_std": float(pyr[:, 1].std()),
        "head_roll_std": float(pyr[:, 2].std()),
        "head_speed_mean": float(speed.mean()),
        "head_jitter": float(np.linalg.norm(acc, axis=1).mean()),
        "nod_rate": float(len(nod_peaks) / dur),
        "speech_head_corr": speech_head,
        "still_ratio": float((still & speaking).sum() / max(speaking.sum(), 1)),
        "blink_rate": float(len(blink_peaks) / dur * 60),
        "gaze_std": float(gaze.std(0).mean()),
        "gaze_shift_rate": float(len(gaze_shifts) / dur),
        "jaw_open_mean": float(tr["jaw"].mean()),
        "jaw_audio_corr": _corr(env, tr["jaw"]),
    }


def evaluate_video(video_path: str, audio_path: str = None) -> Dict[str, float]:
    tr = extract_tracks(video_path)
    env = audio_envelope(audio_path or video_path, len(tr["pitch"]), float(tr["fps"]))
    return compute_features(tr, env)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    a = ap.parse_args()
    for v in a.videos:
        print(os.path.basename(v), json.dumps({k: round(x, 3) for k, x in evaluate_video(v).items()}))
