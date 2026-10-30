"""Full evaluation pipeline for OPSD-I2V experiments.

For each config (base, v1, v2, ablation_no_nat):
  1. Render 40 held-out eval prompts
  2. Compute naturalness_v2 metrics
  3. Generate comparison gallery

Produces:
  - outputs/eval/metrics_all.json
  - outputs/eval/metrics_table.md
  - outputs/eval/gallery/*.gif
"""
import json, os, sys, time, subprocess
from pathlib import Path

import numpy as np

# ---- Configuration ----
EVAL_PROMPTS = "data/prompts/tier2_eval_40.json"
EVAL_FRAMES  = "data/first_frames_eval"
BASE_OUT     = "outputs/eval"

CONFIGS = {
    "base": {
        "desc": "DynamiCrafter Base (no LoRA)",
        "lora_dir": None,
    },
    "v1_nat_clamp": {
        "desc": "OPSD v1 (naturalness clamped → dead)",
        "lora_dir": "outputs/opsd_tier2/lora_step00500",
    },
    "v2_nat_sigmoid": {
        "desc": "OPSD v2 (naturalness sigmoid → active)",
        "lora_dir": "outputs/opsd_tier2_v2/lora_step00500",
    },
    "ablation_no_nat": {
        "desc": "Ablation: no naturalness penalty",
        "lora_dir": "outputs/opsd_tier2_no_nat/lora_step00500",
    },
}


def render_videos(tag: str, lora_dir, out_dir: str, num_prompts: int = 40):
    """Render videos using render_i2v.py."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    # Check if already rendered
    existing = list(out.glob("seed_00/*.mp4"))
    if len(existing) >= num_prompts:
        print(f"  [{tag}] Already rendered {len(existing)} videos, skipping")
        return
    
    cmd = [
        sys.executable, "-m", "src.render_i2v",
        "--frames_dir", EVAL_FRAMES,
        "--prompts", EVAL_PROMPTS,
        "--out_dir", str(out),
        "--seeds", "0",
        "--num_frames", "16",
        "--fps", "8",
        "--limit", str(num_prompts),
    ]
    if lora_dir and Path(lora_dir).exists():
        cmd.extend(["--i2v_lora_dir", lora_dir])
    
    print(f"  [{tag}] Rendering {num_prompts} videos...")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": "."})
    elapsed = time.time() - t0
    
    if result.returncode != 0:
        print(f"  [{tag}] FAILED ({elapsed:.0f}s):")
        print(result.stderr[-500:])
        return
    
    rendered = list(Path(out).glob("seed_00/*.mp4"))
    print(f"  [{tag}] Rendered {len(rendered)} videos in {elapsed:.0f}s")


def compute_metrics(tag: str, video_dir: str):
    """Compute naturalness_v2 metrics for all videos."""
    import imageio.v3 as iio
    from src.eval.naturalness_v2 import naturalness_v2_video
    
    videos = sorted(Path(video_dir).glob("seed_00/*.mp4"))
    if not videos:
        print(f"  [{tag}] No videos found in {video_dir}/seed_00/")
        return []
    
    results = []
    for vp in videos:
        try:
            frames = np.stack(list(iio.imiter(str(vp))))
            m = naturalness_v2_video(frames)
            results.append({
                "file": vp.name,
                "nat": round(m["nat"], 4),
                "dynamic": round(m["dynamic"], 4),
                "appeal": round(m["appeal"], 4),
            })
        except Exception as e:
            print(f"  [{tag}] Error on {vp.name}: {e}")
    
    return results


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--render-only", action="store_true")
    ap.add_argument("--metrics-only", action="store_true")
    ap.add_argument("--tags", nargs="+", default=list(CONFIGS.keys()))
    args = ap.parse_args()
    
    Path(BASE_OUT).mkdir(parents=True, exist_ok=True)
    
    all_metrics = {}
    
    for tag in args.tags:
        if tag not in CONFIGS:
            print(f"Unknown tag: {tag}, skipping")
            continue
        
        cfg = CONFIGS[tag]
        vid_dir = f"{BASE_OUT}/videos_{tag}"
        print(f"\n{'='*60}")
        print(f"  {tag}: {cfg['desc']}")
        print(f"{'='*60}")
        
        if not args.metrics_only:
            render_videos(tag, cfg["lora_dir"], vid_dir)
        
        if not args.render_only:
            metrics = compute_metrics(tag, vid_dir)
            all_metrics[tag] = {
                "desc": cfg["desc"],
                "lora_dir": cfg["lora_dir"],
                "n_videos": len(metrics),
                "per_video": metrics,
                "avg_nat": round(np.mean([m["nat"] for m in metrics]), 4) if metrics else None,
                "avg_dyn": round(np.mean([m["dynamic"] for m in metrics]), 4) if metrics else None,
                "avg_appeal": round(np.mean([m["appeal"] for m in metrics]), 4) if metrics else None,
            }
    
    if not args.render_only and all_metrics:
        # Save JSON
        json_path = f"{BASE_OUT}/metrics_all.json"
        with open(json_path, "w") as f:
            json.dump(all_metrics, f, indent=2)
        print(f"\nMetrics saved to {json_path}")
        
        # Generate markdown table
        md = "# OPSD-I2V Evaluation — Held-out Metrics (n=40 prompts)\n\n"
        md += "| Config | Description | Avg Naturalness | Avg Dynamic | Avg Appeal |\n"
        md += "|--------|-------------|-----------------|-------------|------------|\n"
        for tag, m in all_metrics.items():
            md += f"| {tag} | {m['desc']} | {m.get('avg_nat', '-')} | {m.get('avg_dyn', '-')} | {m.get('avg_appeal', '-')} |\n"
        
        md_path = f"{BASE_OUT}/metrics_table.md"
        with open(md_path, "w") as f:
            f.write(md)
        print(f"Table saved to {md_path}")
        
        # Print summary
        print(f"\n{'='*60}")
        print("  SUMMARY")
        print(f"{'='*60}")
        for tag, m in all_metrics.items():
            print(f"  {tag:25s}  appeal={m.get('avg_appeal', '-'):>8}  nat={m.get('avg_nat', '-'):>8}  dyn={m.get('avg_dyn', '-'):>8}")


if __name__ == "__main__":
    main()
