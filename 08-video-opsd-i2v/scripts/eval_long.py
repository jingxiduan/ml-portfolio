"""Evaluation for the 2000-step long runs (T3).

Evaluates base + V2/NoNat at 500 / 1000 / 2000 checkpoints on the held-out
eval prompts, and reports PAIRED differences with bootstrap 95% CIs
(V2 - NoNat, V2 - Base). Point estimates alone are not enough, so every
comparison is paired per prompt and bootstrapped.

Does NOT modify scripts/eval_full.py; this is a new file.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

EVAL_PROMPTS = "data/prompts/tier2_eval_40.json"
EVAL_FRAMES = "data/first_frames_eval"
BASE_OUT = "outputs/eval_long"

CONFIGS = {
    "base": {"desc": "DynamiCrafter base (no LoRA)", "lora_dir": None},
    "v2_500": {"desc": "V2 (sigmoid naturalness) @500", "lora_dir": "outputs/opsd_long_v2/lora_step00500"},
    "v2_1000": {"desc": "V2 @1000", "lora_dir": "outputs/opsd_long_v2/lora_step01000"},
    "v2_2000": {"desc": "V2 @2000", "lora_dir": "outputs/opsd_long_v2/lora_step02000"},
    "nonat_500": {"desc": "NoNat @500", "lora_dir": "outputs/opsd_long_no_nat/lora_step00500"},
    "nonat_1000": {"desc": "NoNat @1000", "lora_dir": "outputs/opsd_long_no_nat/lora_step01000"},
    "nonat_2000": {"desc": "NoNat @2000", "lora_dir": "outputs/opsd_long_no_nat/lora_step02000"},
}

METRIC_KEYS = ["appeal", "nat", "dynamic"]


def render_videos(tag, lora_dir, out_dir, num_prompts=40, gpu=0):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    existing = list(out.glob("seed_00/*.mp4"))
    if len(existing) >= num_prompts:
        print(f"  [{tag}] already rendered {len(existing)}, skipping")
        return True
    if lora_dir and not Path(lora_dir).exists():
        print(f"  [{tag}] checkpoint missing: {lora_dir}, skipping")
        return False
    cmd = [sys.executable, "-m", "src.render_i2v",
           "--frames_dir", EVAL_FRAMES, "--prompts", EVAL_PROMPTS,
           "--out_dir", str(out), "--seeds", "0",
           "--num_frames", "16", "--fps", "8", "--limit", str(num_prompts)]
    if lora_dir:
        cmd.extend(["--i2v_lora_dir", lora_dir])
    env = {**os.environ, "PYTHONPATH": ".", "CUDA_VISIBLE_DEVICES": str(gpu)}
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        print(f"  [{tag}] FAILED ({time.time()-t0:.0f}s): {r.stderr[-400:]}")
        return False
    print(f"  [{tag}] rendered {len(list(out.glob('seed_00/*.mp4')))} in {time.time()-t0:.0f}s")
    return True


def compute_metrics(tag, video_dir):
    import imageio.v3 as iio
    from src.eval.naturalness_v2 import naturalness_v2_video
    videos = sorted(Path(video_dir).glob("seed_00/*.mp4"))
    res = []
    for vp in videos:
        try:
            frames = np.stack(list(iio.imiter(str(vp))))
            m = naturalness_v2_video(frames)
            res.append({"file": vp.name, "appeal": m["appeal"], "nat": m["nat"], "dynamic": m["dynamic"]})
        except Exception as e:
            print(f"  [{tag}] error {vp.name}: {e}")
    return res


def bootstrap_ci(diffs, n_boot=10000, seed=0):
    """95% percentile CI of the mean of paired differences."""
    d = np.asarray(diffs, dtype=np.float64)
    if len(d) < 2:
        return {"mean": float(d.mean()) if len(d) else None, "lo": None, "hi": None}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return {"mean": float(d.mean()), "lo": float(np.percentile(means, 2.5)),
            "hi": float(np.percentile(means, 97.5))}


def paired(a, b, key):
    """Pair per-video metrics by sorted filename."""
    ma = {x["file"]: x[key] for x in a}
    mb = {x["file"]: x[key] for x in b}
    common = sorted(set(ma) & set(mb))
    return [ma[f] - mb[f] for f in common], len(common)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render-only", action="store_true")
    ap.add_argument("--metrics-only", action="store_true")
    ap.add_argument("--gpus", default="0,1")
    a = ap.parse_args()

    gpus = [int(g) for g in a.gpus.split(",")]
    Path(BASE_OUT).mkdir(parents=True, exist_ok=True)

    tags = list(CONFIGS)
    per_tag = {}
    for i, tag in enumerate(tags):
        cfg = CONFIGS[tag]
        vid_dir = f"{BASE_OUT}/videos_{tag}"
        print(f"\n=== {tag}: {cfg['desc']} ===")
        if not a.metrics_only:
            render_videos(tag, cfg["lora_dir"], vid_dir, gpu=gpus[i % len(gpus)])
        if not a.render_only:
            per_tag[tag] = compute_metrics(tag, vid_dir)
            if per_tag[tag]:
                arr = {k: [x[k] for x in per_tag[tag]] for k in METRIC_KEYS}
                print("  " + "  ".join(f"{k}={np.mean(v):.4f}" for k, v in arr.items())
                      + f"  (n={len(per_tag[tag])})")

    if a.render_only or not per_tag:
        return

    summary = {}
    for tag, rows in per_tag.items():
        summary[tag] = {k: float(np.mean([x[k] for x in rows])) for k in METRIC_KEYS}
        summary[tag]["n"] = len(rows)

    comparisons = {}
    for key in METRIC_KEYS:
        for tag_a, tag_b in [("v2_2000", "nonat_2000"), ("v2_2000", "base"),
                             ("v2_1000", "base"), ("v2_500", "base")]:
            if tag_a not in per_tag or tag_b not in per_tag:
                continue
            if not per_tag[tag_a] or not per_tag[tag_b]:
                continue
            diffs, n = paired(per_tag[tag_a], per_tag[tag_b], key)
            ci = bootstrap_ci(diffs)
            ci["n_pairs"] = n
            comparisons[f"{tag_a} - {tag_b} [{key}]"] = ci
            print(f"  {tag_a} - {tag_b} [{key}]: mean={ci['mean']:+.4f} "
                  f"CI=[{ci['lo']:+.4f}, {ci['hi']:+.4f}] (n={n})")

    out = {"summary": summary, "comparisons": comparisons, "per_video": per_tag}
    with open(f"{BASE_OUT}/metrics_long.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[ok] wrote {BASE_OUT}/metrics_long.json")


if __name__ == "__main__":
    main()
