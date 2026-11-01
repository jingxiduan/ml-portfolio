"""Screen candidate driving clips on the things that actually ruined demos.

Every check below exists because it already bit us once:

  * `resolution`  -- HDTF is only 512x512, so anything rendered from it looks soft.
  * `brightness`  -- one picked clip had luma mean 34.6 and range 69; it looked grey.
  * `speech`      -- one picked clip turned out to be someone SINGING. A talking
                     head must be talking.
  * `english`     -- required by the user; verified by whisper transcription, not guessed.
  * `frontal`     -- news anchors read teleprompters; gaze variation transfers and the
                     digital human ends up staring past the lens.

Speech vs singing is a heuristic, reported as numbers so it can be judged:
singing holds long steady voiced runs over a musical background, speech is
choppy with frequent short pauses. Both `voiced_run_p90` and `harmonic_ratio`
are printed; nothing is silently accepted.
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


def frame_stats(path, n=6):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    W, H = int(cap.get(3)), int(cap.get(4))
    vals = []
    for i in range(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * (i + 0.5) / n))
        ok, fr = cap.read()
        if not ok:
            continue
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        vals.append(g)
    cap.release()
    if not vals:
        return None
    g = np.concatenate([v.reshape(-1) for v in vals])
    return {"w": W, "h": H, "luma_mean": float(g.mean()),
            "luma_p5": float(np.percentile(g, 5)), "luma_p95": float(np.percentile(g, 95))}


def audio_stats(path, seconds=10.0):
    import librosa
    wav = "/tmp/_screen.wav"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-t", str(seconds),
                    "-i", path, "-ac", "1", "-ar", "16000", "-vn", wav], check=True)
    y, sr = librosa.load(wav, sr=16000)
    if len(y) < sr:
        return None
    rms = librosa.feature.rms(y=y, frame_length=640, hop_length=320)[0]
    thr = np.percentile(rms, 60) * 1.15
    voiced = rms > thr
    # run lengths of voiced frames
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
    return {"voiced_frac": float(voiced.mean()),
            "voiced_run_p90": float(np.percentile(runs, 90)) if runs else 0.0,
            "voiced_run_max": float(np.max(runs)) if runs else 0.0,
            "harmonic_ratio": float(np.sum(harm ** 2) / (np.sum(y ** 2) + 1e-9))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", nargs="+", required=True, help="frontal_clips json files")
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "screened_clips.json"))
    a = ap.parse_args()

    paths, meta = [], {}
    for j in a.json:
        d = json.load(open(j))
        for r in d.get("kept", []):
            paths.append(r["clip"])
            meta[r["clip"]] = {"yaw": r["mean_abs_yaw"], "pitch": r["mean_abs_pitch"],
                               "low_deg": r["low_std_deg"]}
    paths += list(a.extra)
    paths = sorted(set(paths))

    from transformers import pipeline
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")

    rows = []
    for p in paths:
        row = {"clip": p, "name": os.path.basename(p)}
        row.update(meta.get(p, {}))
        fs = frame_stats(p)
        if fs:
            row.update(fs)
        au = audio_stats(p)
        if au:
            row.update(au)
        try:
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-t", "8", "-i", p,
                            "-ac", "1", "-ar", "16000", "-vn", "/tmp/_screen8.wav"], check=True)
            row["text"] = asr("/tmp/_screen8.wav")["text"].strip()
        except Exception as e:
            row["text"] = f"ERR {e}"
        rows.append(row)
        print(f"{row['name'][:30]:30s} {row.get('w', 0)}x{row.get('h', 0)} "
              f"luma={row.get('luma_mean', -1):5.1f} rng={row.get('luma_p95', 0) - row.get('luma_p5', 0):5.0f} "
              f"voiced={row.get('voiced_frac', -1):.2f} run90={row.get('voiced_run_p90', -1):4.1f}s "
              f"harm={row.get('harmonic_ratio', -1):.2f} | {row['text'][:44]}", flush=True)

    json.dump(rows, open(a.out, "w"), indent=1)
    print(f"\n[ok] {len(rows)} -> {a.out}")


if __name__ == "__main__":
    main()
