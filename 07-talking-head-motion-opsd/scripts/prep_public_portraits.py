"""Extract HD portraits + driving audio from public talking-head datasets.

Sources (both public, see data/DATA_SOURCES.md):
  - CelebA-HQ  (mattymchen/celeba-hq)      : 1024x1024 still portraits
  - CelebV-HQ  (saeed-5959/celebv_hq_head_talking) : real talking-head videos

From CelebV-HQ we take the FIRST frame as a portrait (a real person, not a
synthetic render) and the audio track as a driving signal, giving demos that use
real speech instead of only TTS.

Outputs:
  data/portraits_public/*.png   HD portraits
  data/audio_public/celebv/*.wav  real-speech driving audio
"""
import argparse
import io
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FFMPEG = "/usr/local/bin/ffmpeg"


def extract_celeba(n, out_dir):
    import pandas as pd
    from PIL import Image
    import glob
    shards = sorted(glob.glob(os.path.join(ROOT, "data", "celeba_hq", "data", "*.parquet")))
    if not shards:
        print("[celeba] no parquet found, skipping")
        return []
    df = pd.read_parquet(shards[0])
    col = "image" if "image" in df.columns else df.columns[0]
    os.makedirs(out_dir, exist_ok=True)
    saved = []
    for i in range(min(n, len(df))):
        v = df.iloc[i][col]
        if isinstance(v, dict):
            v = v.get("bytes") or v.get("path")
        if isinstance(v, bytes):
            im = Image.open(io.BytesIO(v))
        elif isinstance(v, str) and os.path.exists(v):
            im = Image.open(v)
        else:
            continue
        im = im.convert("RGB")
        p = os.path.join(out_dir, f"celebahq_{i:03d}.png")
        im.save(p)
        saved.append(p)
        if len(saved) >= n:
            break
    print(f"[celeba] saved {len(saved)} portraits -> {out_dir}")
    return saved


def extract_celebv(videos, portrait_dir, audio_dir):
    """First frame -> portrait; audio track -> wav."""
    os.makedirs(portrait_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)
    saved_p, saved_a = [], []
    for vp in videos:
        base = os.path.splitext(os.path.basename(vp))[0]
        png = os.path.join(portrait_dir, f"celebv_{base}.png")
        wav = os.path.join(audio_dir, f"celebv_{base}.wav")
        if not os.path.exists(png):
            r = subprocess.run([FFMPEG, "-y", "-i", vp, "-vf", "select=eq(n\\,0)",
                                "-vframes", "1", "-q:v", "2", png],
                               capture_output=True, text=True)
            if r.returncode != 0 or not os.path.exists(png):
                print(f"  [skip frame] {base}: {r.stderr[-120:]}")
                continue
        saved_p.append(png)
        if not os.path.exists(wav):
            r = subprocess.run([FFMPEG, "-y", "-i", vp, "-vn", "-ac", "1",
                                "-ar", "16000", "-f", "wav", wav],
                               capture_output=True, text=True)
            if r.returncode != 0 or not os.path.exists(wav):
                print(f"  [skip audio] {base}")
                continue
        saved_a.append(wav)
    print(f"[celebv] {len(saved_p)} portraits, {len(saved_a)} audio clips")
    return saved_p, saved_a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-celeba", type=int, default=12)
    ap.add_argument("--celebv-dir", default=os.path.join(ROOT, "data", "celebv_raw", "head_talk"))
    ap.add_argument("--portrait-dir", default=os.path.join(ROOT, "data", "portraits_public"))
    ap.add_argument("--audio-dir", default=os.path.join(ROOT, "data", "audio_public", "celebv"))
    a = ap.parse_args()

    import glob as _g
    vids = sorted(_g.glob(os.path.join(a.celebv_dir, "*.mp4")))
    print(f"[prep] celeba={a.n_celeba} celebv videos={len(vids)}")

    extract_celeba(a.n_celeba, a.portrait_dir)
    if vids:
        extract_celebv(vids, a.portrait_dir, a.audio_dir)


if __name__ == "__main__":
    main()
