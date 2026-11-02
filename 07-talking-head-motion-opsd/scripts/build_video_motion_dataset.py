"""Learn from REAL VIDEO instead of audio only.

Until now the trainer saw only audio (AISHELL-1) plus a hand-written reward.
This script builds a supervised audio->motion dataset from public talking-head
video: for every clip we run LivePortrait's own motion extractor over the real
frames, so each 4-second window is a pair

    (audio wav, real human 73-d motion coefficients)

which lets the model *imitate how real people actually move* (behaviour
cloning) instead of only chasing a reward. The reward still runs on top.

Filters (all measured, nothing guessed): luminance, speech-not-singing,
English (whisper), face found by the cropper, enough windows.
"""
import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(ROOT, "src")))
sys.path.insert(0, os.path.abspath(os.path.join(ROOT, "scripts")))

import cv2  # noqa: E402
from render_real_driven import (  # noqa: E402
    FPS, read_frames, extract_audio, real_coefficients, stabilize_scale_translation)


def luma_ok(path, lo=45.0, hi=215.0):
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    vals = []
    for f in (0.1, 0.4, 0.7):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * f))
        ok, fr = cap.read()
        if ok:
            vals.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).mean())
    cap.release()
    if not vals:
        return False, 0.0
    m = float(np.mean(vals))
    return (lo <= m <= hi), m


def speech_stats(wav, seconds=8.0):
    import librosa
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
            runs.append(cur * 320 / 16000.0); cur = 0
    if cur:
        runs.append(cur * 320 / 16000.0)
    harm = librosa.effects.harmonic(y)
    return {"voiced_run_p90": float(np.percentile(runs, 90)) if runs else 0.0,
            "harmonic_ratio": float(np.sum(harm ** 2) / (np.sum(y ** 2) + 1e-9))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos-dir", default=os.path.join(ROOT, "data", "videos_public"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "motion_ds"))
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--win-seconds", type=float, default=4.0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-english", action="store_true")
    ap.add_argument("--max-run90", type=float, default=0.6, help="above this = probably singing")
    ap.add_argument("--max-harm", type=float, default=0.75)
    ap.add_argument("--stride-frames", type=int, default=50,
                    help="window stride; clips are only 3-8 s so we overlap to get more windows")
    a = ap.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(a.gpu))
    os.makedirs(a.out, exist_ok=True)

    from src.config.argument_config import ArgumentConfig  # noqa: E402
    from src.config.crop_config import CropConfig          # noqa: E402
    from src.utils.cropper import Cropper                  # noqa: E402
    from engine import TalkingHeadEngine                   # noqa: E402

    eng = TalkingHeadEngine(load_renderer=False)
    args = ArgumentConfig()
    crop_cfg = CropConfig(**{k: v for k, v in args.__dict__.items() if hasattr(CropConfig, k)})
    cropper = Cropper(crop_cfg=crop_cfg)
    W = eng.n_motions
    print(f"[ds] window = {W} frames ({W / FPS:.2f}s)", flush=True)

    vids = sorted(glob.glob(os.path.join(a.videos_dir, "**", "*.mp4"), recursive=True))
    if a.limit:
        vids = vids[:a.limit]
    print(f"[ds] {len(vids)} candidate clips", flush=True)

    asr = None
    if not a.no_english:
        from transformers import pipeline
        asr = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")

    rows, kept = [], 0
    for vi, v in enumerate(vids, 1):
        name = os.path.splitext(os.path.basename(v))[0]
        row = {"video": os.path.relpath(v, ROOT), "name": name, "kept": False}
        try:
            ok, lum = luma_ok(v)
            row["luma"] = lum
            if not ok:
                row["drop"] = "luma"; rows.append(row); continue
            wav = os.path.join("/tmp", f"_ds_{name}.wav")
            extract_audio(v, wav)
            au = speech_stats(wav)
            if au:
                row.update(au)
                if au["voiced_run_p90"] > a.max_run90 or au["harmonic_ratio"] > a.max_harm:
                    row["drop"] = "singing"; rows.append(row); continue
            if asr is not None:
                subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-t", "8", "-i", wav,
                                "-ac", "1", "-ar", "16000", "/tmp/_ds8.wav"], check=True)
                row["text"] = asr("/tmp/_ds8.wav")["text"].strip()
            frames = read_frames(v, int(a.seconds * FPS))
            coef = real_coefficients(eng, frames, cropper, crop_cfg)
            if coef.shape[0] < W:
                row["drop"] = "too_short"; rows.append(row); continue
            if coef.shape[1] < 73:
                coef = torch.cat([coef, torch.zeros(coef.shape[0], 73 - coef.shape[1])], dim=-1)
            coef = stabilize_scale_translation(coef)      # kill the scale pulsing
            coef = torch.nan_to_num(coef, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20, 20)
            starts = list(range(0, max(coef.shape[0] - W, 0) + 1, a.stride_frames))
            if not starts:
                row["drop"] = "no_window"; rows.append(row); continue
            import soundfile as sf
            y, sr = sf.read(wav)
            spf = int(sr / FPS)                            # samples per frame
            n_win = 0
            for w, st in enumerate(starts):
                c = coef[st:st + W].clone()
                s0, s1 = st * spf, (st + W) * spf
                seg = y[s0:s1]
                if len(seg) < W * spf:
                    break
                stem = f"{name}__w{w:03d}"
                sf.write(os.path.join(a.out, stem + ".wav"), seg, sr)
                torch.save(c, os.path.join(a.out, stem + ".coef.pt"))
                n_win += 1
            row["windows"] = n_win
            row["kept"] = True
            kept += 1
        except Exception as e:
            row["drop"] = f"err:{type(e).__name__}"
        rows.append(row)
        print(f"[{vi}/{len(vids)}] {name[:34]:34s} kept={row['kept']} "
              f"{row.get('drop','')} win={row.get('windows',0)}", flush=True)

    json.dump(rows, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
    nwin = len(glob.glob(os.path.join(a.out, "*.coef.pt")))
    print(f"\n[ok] {kept}/{len(vids)} clips, {nwin} windows -> {a.out}")


if __name__ == "__main__":
    main()
