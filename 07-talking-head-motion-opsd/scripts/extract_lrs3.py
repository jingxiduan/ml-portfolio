"""Extract LRS3 (BBC news / TED talk) clips to mp4 + wav on disk.

LRS3 is the right reference for `professional陈述`: news anchors and lecturers
speaking to camera, not theatrical performance. This is what the reward target
band should be calibrated against.
"""
import argparse
import os

import imageio.v3 as iio
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=os.path.join(
        ROOT, "data", "lrs3", "data", "train-00000-of-00002-6d235766a16b101a.parquet"))
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--video-out", default=os.path.join(ROOT, "data", "lrs3", "clips"))
    ap.add_argument("--audio-out", default=os.path.join(ROOT, "data", "audio_public", "lrs3"))
    ap.add_argument("--fps", type=int, default=25)
    a = ap.parse_args()

    os.makedirs(a.video_out, exist_ok=True)
    os.makedirs(a.audio_out, exist_ok=True)
    df = pd.read_parquet(a.parquet)
    print(f"[lrs3] {len(df)} rows, extracting {a.n} from {a.start}")

    nv = na = 0
    for i in range(a.start, min(a.start + a.n, len(df))):
        r = df.iloc[i]
        raw = r["video"]
        # LRS3 (this HF build) is a ragged object array: frames -> rows -> 96 px.
        # Each frame reconstructs to (96, 96) GRAYSCALE -- low-res mouth crop,
        # so it is only usable for rough statistics, not for HD demos.
        frames = np.stack([np.stack([np.asarray(x, dtype=np.uint8) for x in fr])
                           for fr in raw])
        if frames.ndim == 3:
            frames = np.repeat(frames[..., None], 3, axis=-1)
        frames = frames[..., :3].astype(np.uint8)
        vp = os.path.join(a.video_out, f"lrs3_{i:04d}.mp4")
        ap_ = os.path.join(a.audio_out, f"lrs3_{i:04d}.wav")
        try:
            iio.imwrite(vp, list(frames), fps=a.fps, codec="libx264", quality=8)
            nv += 1
        except Exception as e:
            print(f"  [video skip] {i}: {str(e)[:60]}")
            continue
        try:
            import soundfile as sf
            aud = np.asarray(r["audio"], dtype=np.float32)
            sr = 16000
            if aud.ndim > 1:
                aud = aud.mean(axis=-1)
            mx = float(np.abs(aud).max()) or 1.0
            sf.write(ap_, aud / mx * 0.9, sr)
            na += 1
        except Exception as e:
            print(f"  [audio skip] {i}: {str(e)[:60]}")
    print(f"[ok] {nv} videos -> {a.video_out}, {na} audio -> {a.audio_out}")
    print("[ok] sample label:", df.iloc[a.start]["label"][:80])


if __name__ == "__main__":
    main()
