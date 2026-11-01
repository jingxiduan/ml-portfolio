"""Check every persona demo clip against the four things the user cares about.

  1. speaking, not singing   -> voiced_run_p90 / harmonic_ratio (librosa)
  2. English speech          -> whisper-tiny transcription
  3. voice gender == face gender
       face  : InsightFace `genderage.onnx` (buffalo_l), median over frames
       voice : median F0 from librosa.pyin (<165 Hz = male, >165 Hz = female)
  4. looking at the camera   -> mean |yaw| / |pitch| (MediaPipe + solvePnP)

Nothing is silently accepted: every number is printed and written to
outputs/persona_clip_checks.json.
"""
import argparse
import glob
import json
import os
import subprocess
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(ROOT, "src")))
sys.path.insert(0, os.path.abspath(os.path.join(ROOT, "scripts")))

GENDERAGE = os.path.join(ROOT, "third_party", "JoyVASA", "pretrained_weights",
                         "insightface", "models", "buffalo_l", "genderage.onnx")


def audio_stats(path, seconds=10.0):
    import librosa
    wav = "/tmp/_chk.wav"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-t", str(seconds),
                    "-i", path, "-ac", "1", "-ar", "16000", "-vn", wav], check=True)
    y, sr = librosa.load(wav, sr=16000)
    if len(y) < sr:
        return None
    rms = librosa.feature.rms(y=y, frame_length=640, hop_length=320)[0]
    thr = np.percentile(rms, 60) * 1.15
    voiced = rms > thr
    runs, cur = [], 0
    for v in voiced:
        if v:
            cur += 1
        elif cur:
            runs.append(cur * 320 / 16000.0)
            cur = 0
    if cur:
        runs.append(cur * 320 / 16000.0)
    harm = librosa.effects.harmonic(y)
    f0, _, _ = librosa.pyin(y, fmin=60, fmax=400, sr=sr, frame_length=2048)
    f0 = f0[~np.isnan(f0)] if f0 is not None else np.array([])
    return {"voiced_frac": float(voiced.mean()),
            "voiced_run_p90": float(np.percentile(runs, 90)) if runs else 0.0,
            "harmonic_ratio": float(np.sum(harm ** 2) / (np.sum(y ** 2) + 1e-9)),
            "f0_median": float(np.median(f0)) if len(f0) else 0.0}


def face_gender(path, n=8):
    """Median InsightFace genderage score over n frames. Returns (sex, conf)."""
    import onnxruntime as ort
    sess = ort.InferenceSession(GENDERAGE, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    name = inp.name
    size = tuple(inp.shape[2:4][::-1]) if len(inp.shape) == 4 else (96, 96)
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    scores = []
    for i in range(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * (i + 0.5) / n))
        ok, fr = cap.read()
        if not ok:
            continue
        aimg = cv2.resize(fr, size, interpolation=cv2.INTER_LINEAR)
        blob = cv2.dnn.blobFromImage(aimg, 1.0 / 127.5, size,
                                     (127.5, 127.5, 127.5), swapRB=True)
        out = sess.run(None, {name: blob})[0]
        v = np.array(out).reshape(-1)
        # buffalo_l genderage: [gender_logit, age]; fall back to first element
        scores.append(float(v[0]))
    cap.release()
    if not scores:
        return "?", 0.0
    m = float(np.median(scores))
    return ("M" if m > 0.5 else "F"), abs(m - 0.5) * 2


def pose_stats(path):
    import video_metrics as vm
    tr = vm.extract_tracks(path)
    if len(tr.get("yaw", [])) == 0:
        return {}
    yaw, pitch = np.abs(tr["yaw"]), np.abs(tr["pitch"])
    return {"mean_abs_yaw": float(yaw.mean()), "mean_abs_pitch": float(pitch.mean()),
            "detect_rate": float(tr.get("detect_rate", 0.0))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(ROOT, "outputs", "industry_demo_v3"))
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "persona_clip_checks.json"))
    a = ap.parse_args()

    rows = []
    for p in sorted(glob.glob(os.path.join(a.dir, "*", "*__base.mp4"))):
        d, fn = os.path.split(p)
        industry = os.path.basename(d)
        stem = fn[:-len("__base.mp4")]
        # <industry>_<idx>_<portrait>_<clip>
        parts = stem.split("_", 3)
        idx, portrait, clip = parts[1], parts[2], parts[3] if len(parts) > 3 else ""
        row = {"industry": industry, "idx": idx, "portrait": portrait, "clip": clip,
               "base": p, "persona": os.path.join(d, stem + f"__{industry}.mp4")}
        rows.append(row)

    from transformers import pipeline
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")

    for r in rows:
        au = audio_stats(r["base"])
        if au:
            r.update(au)
        r["voice_sex"] = "M" if r.get("f0_median", 0) and r["f0_median"] < 165 else "F"
        try:
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-t", "8", "-i", r["base"],
                            "-ac", "1", "-ar", "16000", "-vn", "/tmp/_chk8.wav"], check=True)
            r["text"] = asr("/tmp/_chk8.wav")["text"].strip()
        except Exception as e:
            r["text"] = f"ERR {e}"
        try:
            r["face_sex"], r["face_sex_conf"] = face_gender(r["base"])
        except Exception as e:
            r["face_sex"], r["face_sex_conf"] = "?", 0.0
            print("  gender err", e)
        try:
            r.update(pose_stats(r["base"]))
        except Exception as e:
            print("  pose err", e)
        r["gender_match"] = (r["face_sex"] == r["voice_sex"])
        print(f"{r['industry'][:10]:10s} {r['idx']} {r['portrait']:6s} "
              f"f0={r.get('f0_median', 0):6.1f} voice={r['voice_sex']} "
              f"face={r['face_sex']}({r['face_sex_conf']:.2f}) "
              f"match={str(r['gender_match']):5s} "
              f"run90={r.get('voiced_run_p90', -1):4.1f} harm={r.get('harmonic_ratio', -1):.2f} "
              f"yaw={r.get('mean_abs_yaw', -1):4.1f} pitch={r.get('mean_abs_pitch', -1):4.1f} "
              f"| {r['text'][:40]}", flush=True)

    json.dump(rows, open(a.out, "w"), indent=1)
    print(f"\n[ok] {len(rows)} -> {a.out}")


if __name__ == "__main__":
    main()
