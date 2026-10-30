"""Build 8-15 s training/eval audio segments from AISHELL-1.

Rules (per task book section 5):
  - Speaker split: first N speakers -> train, next M speakers -> eval.
    NO overlap between train and eval speakers.
  - AISHELL-1 utterances average ~5 s; the motion trainer uses 4 s windows with
    1 s stride. Concatenate ADJACENT utterances OF THE SAME SPEAKER (by numeric
    id order) until the segment is >= 8 s, inserting 0.3 s silence at joins.
    Never concatenate across speakers.
  - Output: 16 kHz mono wav.

Writes data/audio_manifest_large.json + data/audio_public/train|eval/*.wav
Also writes data/DATA_SOURCES.md
"""
import argparse
import json
import os
import random
import subprocess
import sys

import numpy as np
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WAV_ROOT = os.path.join(ROOT, "data", "audio_public", "data_aishell", "wav", "train")

TARGET_SR = 16000
MIN_SEC = 8.0
MAX_SEC = 15.0
GAP_SEC = 0.3


def read_wav(path):
    x, sr = sf.read(path, dtype="float32")
    if getattr(x, "ndim", 1) > 1:
        x = x.mean(axis=1)
    if sr != TARGET_SR:
        # resample with soxr/ffmpeg via soundfile is not available; use linear interp
        n = int(round(len(x) * TARGET_SR / float(sr)))
        idx = np.linspace(0, len(x) - 1, n).astype(np.float64)
        x = np.interp(idx, np.arange(len(x)), x.astype(np.float64)).astype(np.float32)
    return x


def build_segments(speakers, target_n, seed):
    """Concatenate adjacent same-speaker utterances into 8-15 s segments."""
    rng = random.Random(seed)
    gap = np.zeros(int(GAP_SEC * TARGET_SR), dtype=np.float32)
    segs = []
    for spk in speakers:
        d = os.path.join(WAV_ROOT, spk)
        if not os.path.isdir(d):
            continue
        files = sorted(f for f in os.listdir(d) if f.endswith(".wav"))
        # sort by numeric utterance id so "adjacent" is deterministic
        files.sort(key=lambda f: int("".join(ch for ch in f if ch.isdigit()) or 0))
        buf, parts = [], []
        for f in files:
            try:
                x = read_wav(os.path.join(d, f))
            except Exception:
                continue
            if buf:
                buf.append(gap)
            buf.append(x)
            parts.append(f)
            total = sum(b.size for b in buf) / float(TARGET_SR)
            if total >= MIN_SEC:
                y = np.concatenate(buf)
                if y.size / TARGET_SR > MAX_SEC:
                    y = y[: int(MAX_SEC * TARGET_SR)]
                segs.append((spk, parts, y))
                buf, parts = [], []
    rng.shuffle(segs)
    return segs[:target_n]


def write_wav(path, y):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, y, TARGET_SR, subtype="PCM_16")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-eval", type=int, default=100)
    ap.add_argument("--n-train-speakers", type=int, default=60)
    ap.add_argument("--n-eval-speakers", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    all_spk = sorted(d for d in os.listdir(WAV_ROOT) if os.path.isdir(os.path.join(WAV_ROOT, d)))
    print(f"[data] {len(all_spk)} speakers available")
    train_spk = all_spk[: a.n_train_speakers]
    eval_spk = all_spk[a.n_train_speakers: a.n_train_speakers + a.n_eval_speakers]
    print(f"[data] train speakers={len(train_spk)} eval speakers={len(eval_spk)}")
    assert not (set(train_spk) & set(eval_spk)), "speaker overlap!"

    tr = build_segments(train_spk, a.n_train, a.seed)
    ev = build_segments(eval_spk, a.n_eval, a.seed + 1)
    print(f"[data] train segments={len(tr)} eval segments={len(ev)}")

    manifest = {"source": "AISHELL-1", "license": "Apache 2.0",
                "train_speakers": train_spk, "eval_speakers": eval_spk,
                "train": [], "eval": []}
    for split, segs in (("train", tr), ("eval", ev)):
        for i, (spk, parts, y) in enumerate(segs):
            rel = f"data/audio_public/{split}/seg_{i:05d}.wav"
            write_wav(os.path.join(ROOT, rel), y)
            manifest[split].append({"path": rel, "speaker": spk,
                                    "sources": parts, "seconds": round(y.size / TARGET_SR, 2)})
        print(f"[data] wrote {len(segs)} {split} wavs")

    with open(os.path.join(ROOT, "data", "audio_manifest_large.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print("[ok] manifest written")


if __name__ == "__main__":
    main()
