"""Fetch the public datasets this project needs.

Three things are downloaded, all from Hugging Face, all public:

  1. AISHELL-1 (Apache 2.0)  -- multi-speaker Chinese speech -> training audio.
     Only the speaker archives we need are pulled, never the full 178 h.
  2. HDTF                    -- news-anchor footage. Used to calibrate the
     *restrained* (finance) motion band and as demo portraits.
  3. CelebV-HQ               -- 23 k real talking-head clips covering many
     YouTube genres. Used to calibrate the *mid* and *energetic* bands
     (real-estate walkthrough / short-video hook) and as demo portraits.

Everything is written under data/ and is reproducible from this file alone.
"""
import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

AISHELL_BASE = "https://huggingface.co/datasets/AISHELL/AISHELL-1/resolve/main"
HDTF_BASE = "https://huggingface.co/datasets/saeed-5959/hdtf/resolve/main"
CELEBV_BASE = "https://huggingface.co/datasets/saeed-5959/celebv_hq_head_talking/resolve/main"


def _curl(url, dest, timeout=600):
    """Download url -> dest unless dest already exists and is non-empty."""
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest, "cached"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    cmd = ["curl", "-L", "--retry", "3", "--retry-delay", "2",
           "--max-time", str(timeout), "-o", tmp, url]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(tmp) or os.path.getsize(tmp) == 0:
        if os.path.exists(tmp):
            os.remove(tmp)
        return dest, f"FAILED rc={r.returncode} {r.stderr[-200:]}"
    os.replace(tmp, dest)
    return dest, "ok"


def _run(jobs, workers, label):
    done, failed = 0, []
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for dest, status in ex.map(lambda j: _curl(*j), jobs):
            done += 1
            if status not in ("ok", "cached"):
                failed.append((dest, status))
                print(f"  [fail] {os.path.basename(dest)}: {status}", flush=True)
            elif done % 10 == 0:
                print(f"  [{label}] {done}/{len(jobs)}  {time.time()-t0:.0f}s", flush=True)
    print(f"[{label}] {len(jobs)-len(failed)}/{len(jobs)} downloaded in {time.time()-t0:.0f}s",
          flush=True)
    return failed


def fetch_aishell(n_speakers, out_dir):
    """Speaker archives S0002.. ; each is a tar.gz of one speaker's wavs."""
    jobs = []
    spk_ids = [f"S{i:04d}" for i in range(2, 2 + n_speakers)]
    for s in spk_ids:
        jobs.append((f"{AISHELL_BASE}/data_aishell/wav/{s}.tar.gz",
                     os.path.join(out_dir, "data_aishell", "wav", f"{s}.tar.gz")))
    print(f"[aishell] {len(jobs)} speaker archives", flush=True)
    return _run(jobs, 8, "aishell"), spk_ids


def extract_aishell(out_dir):
    """Untar into data_aishell/wav/train/<spk>/ (layout build_audio_segments.py expects)."""
    wav_dir = os.path.join(out_dir, "data_aishell", "wav")
    train_dir = os.path.join(wav_dir, "train")
    os.makedirs(train_dir, exist_ok=True)
    tars = sorted(f for f in os.listdir(wav_dir) if f.endswith(".tar.gz"))
    n = 0
    for t in tars:
        spk = t.replace(".tar.gz", "")
        dst = os.path.join(train_dir, spk)
        if os.path.isdir(dst) and any(f.endswith(".wav") for f in os.listdir(dst)):
            continue
        # archive layout is  train/<SPK>/<utt>.wav  -> untar into wav/ so that the
        # result is wav/train/<SPK>/<utt>.wav, which build_audio_segments.py expects.
        subprocess.run(["tar", "-xzf", os.path.join(wav_dir, t), "-C", wav_dir],
                       capture_output=True)
        n += 1
    print(f"[aishell] extracted {n} archives -> {train_dir}", flush=True)
    return train_dir


def _manifest_list(repo, prefix, ext):
    """List files under `prefix` in a HF dataset repo via the API."""
    url = f"https://huggingface.co/api/datasets/{repo}"
    r = subprocess.run(["curl", "-s", "--max-time", "60", url], capture_output=True, text=True)
    try:
        sib = json.loads(r.stdout)["siblings"]
    except Exception:
        print(f"[manifest] could not list {repo}: {r.stdout[:200]}", flush=True)
        return []
    return [s["rfilename"] for s in sib
            if s["rfilename"].startswith(prefix) and s["rfilename"].endswith(ext)]


def fetch_videos(repo, base, prefix, n, start, out_dir, label, workers=8):
    files = _manifest_list(repo, prefix, ".mp4")
    if not files:
        return []
    sel = files[start:start + n]
    jobs = [(f"{base}/{f}", os.path.join(out_dir, os.path.basename(f))) for f in sel]
    print(f"[{label}] {len(jobs)} of {len(files)} available clips", flush=True)
    failed = _run(jobs, workers, label)
    return [os.path.join(out_dir, os.path.basename(f)) for f in sel
            if os.path.exists(os.path.join(out_dir, os.path.basename(f)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aishell-speakers", type=int, default=75)
    ap.add_argument("--hdtf", type=int, default=40, help="news-anchor clips (finance band)")
    ap.add_argument("--celebv", type=int, default=90, help="genre-diverse clips (other bands)")
    ap.add_argument("--skip-aishell", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "audio_public"))
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    videos_dir = os.path.join(ROOT, "data", "videos_public")

    if not a.skip_aishell:
        failed, spk = fetch_aishell(a.aishell_speakers, a.out)
        extract_aishell(a.out)
        if failed:
            print(f"[aishell] {len(failed)} archives failed", flush=True)

    hdtf_dir = os.path.join(videos_dir, "hdtf")
    fetch_videos("saeed-5959/hdtf", HDTF_BASE, "videos/", a.hdtf, 0, hdtf_dir, "hdtf")

    celebv_dir = os.path.join(videos_dir, "celebv")
    fetch_videos("saeed-5959/celebv_hq_head_talking", CELEBV_BASE, "head_talk/",
                 a.celebv, 0, celebv_dir, "celebv")

    print("[ok] fetch_public_data done", flush=True)


if __name__ == "__main__":
    main()
