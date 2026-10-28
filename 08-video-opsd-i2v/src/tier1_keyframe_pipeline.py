"""Tier 1 pipeline — training-free keyframe carryover study.

Orchestrates the end-to-end Tier 1 experiment:
    1. generate_first_frames.py --mode base  (if frames_base/ missing)
    2. generate_first_frames.py --mode opsd  (if frames_opsd/ missing)
    3. for method in {base, opsd}:
         render videos with frozen DynamiCrafter from frames_{method}/seed_SS/i.png
    4. VBench-I2V on both video dirs
    5. Qwen2-VL pairwise judge
    6. collect_metrics into tier1_main_table.{csv, md}
    7. paired Wilcoxon per dim → stats.json

This script does **not** train anything. Everything is inference.

Usage:
    python -m src.tier1_keyframe_pipeline \
        --config configs/tier1_carryover.yaml

If `--skip_t2i` is set we assume frames are already generated.
If `--skip_i2v` is set we assume videos are already rendered.
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# =============================================================================
# steps
# =============================================================================


def step_t2i(cfg: Dict, mode: str) -> Path:
    """Call generate_first_frames.py as a subprocess (so each mode owns the GPU memory)."""
    out_dir = Path(cfg["frames_dir"][mode])
    manifest = out_dir / "manifest.json"
    if manifest.is_file() and not cfg.get("force_t2i", False):
        logger.info(f"[t2i:{mode}] skipping, manifest exists at {manifest}")
        return out_dir
    cmd = [
        sys.executable, "-m", "src.t2i.generate_first_frames",
        "--prompts", cfg["prompts_json"],
        "--out_dir", str(out_dir),
        "--mode", mode,
        "--seeds", *[str(s) for s in cfg["seeds"]],
        "--height", str(cfg["frame_height"]),
        "--width", str(cfg["frame_width"]),
        "--steps", str(cfg["t2i_steps"]),
        "--cfg", str(cfg["t2i_cfg"]),
        "--dtype", cfg["t2i_dtype"],
    ]
    if cfg.get("limit"):
        cmd += ["--limit", str(cfg["limit"])]
    if cfg.get("cpu_offload"):
        cmd += ["--cpu_offload"]
    logger.info(f"[t2i:{mode}] launching: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return out_dir


def step_i2v(cfg: Dict, mode: str) -> Path:
    """Render videos from frames using frozen DynamiCrafter. One subprocess per mode."""
    frames_dir = Path(cfg["frames_dir"][mode])
    videos_dir = Path(cfg["videos_dir"][mode])
    if (videos_dir / "manifest.json").is_file() and not cfg.get("force_i2v", False):
        logger.info(f"[i2v:{mode}] skipping, manifest exists at {videos_dir}/manifest.json")
        return videos_dir
    cmd = [
        sys.executable, "-m", "src.render_i2v",
        "--frames_dir", str(frames_dir),
        "--prompts", cfg["prompts_json"],
        "--out_dir", str(videos_dir),
        "--seeds", *[str(s) for s in cfg["seeds"]],
        "--num_frames", str(cfg["num_frames"]),
        "--fps", str(cfg["fps"]),
    ]
    if cfg.get("limit"):
        cmd += ["--limit", str(cfg["limit"])]
    logger.info(f"[i2v:{mode}] launching: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return videos_dir


def step_vbench(cfg: Dict, mode: str) -> Path:
    """Run the lite VBench-I2V evaluator on the rendered videos.

    Falls back from the official VBench CLI (which needs decord → Python 3.11)
    to our own 4-dim re-implementation. See src/eval/vbench_lite_eval.py.
    """
    videos_dir = Path(cfg["videos_dir"][mode])
    results_dir = Path(cfg["vbench_dir"][mode])
    results_file = results_dir / "eval_results.json"
    if results_file.is_file() and not cfg.get("force_vbench", False):
        logger.info(f"[vbench:{mode}] skipping, results exist at {results_file}")
        return results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "src.eval.vbench_lite_eval",
        "--videos_dir", str(videos_dir),
        "--out_json", str(results_file),
        "--dims", "subject_consistency", "motion_smoothness",
                  "dynamic_degree", "aesthetic_quality",
    ]
    logger.info(f"[vbench:{mode}] launching: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return results_dir


def step_judge(cfg: Dict) -> Path:
    """Qwen2-VL pairwise judge on OPSD vs base."""
    out_path = Path(cfg["judge_json"])
    if out_path.is_file() and not cfg.get("force_judge", False):
        logger.info(f"[judge] skipping, exists at {out_path}")
        return out_path
    cmd = [
        sys.executable, "-m", "src.eval.llm_judge",
        "--a", str(Path(cfg["videos_dir"]["opsd"])),
        "--b", str(Path(cfg["videos_dir"]["base"])),
        "--out", str(out_path),
        "--n_pairs", str(cfg["judge_n_pairs"]),
        "--seed", str(cfg["seeds"][0]),
    ]
    logger.info(f"[judge] launching: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return out_path


# =============================================================================
# stats
# =============================================================================


def _load_vbench_per_prompt(path: Path) -> Dict[str, Dict[str, float]]:
    """Return {dim: {prompt_id: score}}. VBench v2 saves per-sample scores alongside the
    aggregate. If only aggregates are available, we fall back to those (no p-values)."""
    results = json.loads((path / "eval_results.json").read_text())
    out: Dict[str, Dict[str, float]] = {}
    for dim, payload in results.items():
        if isinstance(payload, list) and len(payload) >= 2 and isinstance(payload[1], dict):
            # payload = [aggregate_score, {sample_id: score, ...}]
            out[dim] = {k: float(v) for k, v in payload[1].items()}
        elif isinstance(payload, dict):
            out[dim] = {k: float(v) for k, v in payload.items()}
        elif isinstance(payload, (int, float)):
            out[dim] = {"_agg": float(payload)}
    return out


def paired_stats(
    base: Dict[str, Dict[str, float]], opsd: Dict[str, Dict[str, float]]
) -> Dict[str, Dict[str, float]]:
    """Paired Wilcoxon per dim, with Holm-Bonferroni over dims."""
    try:
        from scipy import stats as sstats
    except ImportError:
        logger.warning("scipy not available; skipping p-values.")
        sstats = None

    rows: Dict[str, Dict[str, float]] = {}
    for dim in sorted(set(base) & set(opsd)):
        b_scores = base[dim]
        o_scores = opsd[dim]
        keys = sorted(set(b_scores) & set(o_scores))
        if "_agg" in keys and len(keys) == 1:
            rows[dim] = {
                "base_mean": b_scores["_agg"],
                "opsd_mean": o_scores["_agg"],
                "delta": o_scores["_agg"] - b_scores["_agg"],
                "n": 1, "p": float("nan"),
            }
            continue
        b_arr = np.array([b_scores[k] for k in keys])
        o_arr = np.array([o_scores[k] for k in keys])
        delta = float(np.mean(o_arr - b_arr))
        row = {
            "base_mean": float(b_arr.mean()),
            "opsd_mean": float(o_arr.mean()),
            "delta": delta,
            "n": len(keys),
        }
        if sstats is not None:
            try:
                _, p = sstats.wilcoxon(o_arr - b_arr, alternative="greater")
                row["p"] = float(p)
            except Exception:
                row["p"] = float("nan")
        rows[dim] = row

    # Holm-Bonferroni
    ordered = sorted(rows.items(), key=lambda kv: kv[1].get("p", 1.0))
    m = len([r for _, r in ordered if not np.isnan(r.get("p", float("nan")))])
    for i, (dim, r) in enumerate(ordered):
        if np.isnan(r.get("p", float("nan"))):
            r["p_adj"] = float("nan")
        else:
            r["p_adj"] = min(1.0, r["p"] * (m - i))
    return rows


def write_table(rows: Dict[str, Dict[str, float]], out_dir: Path) -> None:
    import csv
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "tier1_main_table.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dim", "base_mean", "opsd_mean", "delta", "n", "p", "p_adj"])
        for dim, r in rows.items():
            w.writerow([dim, f"{r['base_mean']:.4f}", f"{r['opsd_mean']:.4f}",
                        f"{r['delta']:+.4f}", r["n"],
                        f"{r.get('p', float('nan')):.4g}",
                        f"{r.get('p_adj', float('nan')):.4g}"])
    with (out_dir / "tier1_main_table.md").open("w") as f:
        f.write("| dim | base | OPSD | Δ | n | p | p_adj |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for dim, r in rows.items():
            star = "★" if r.get("p_adj", 1) < 0.05 else ""
            f.write(
                f"| {dim} | {r['base_mean']:.3f} | **{r['opsd_mean']:.3f}** "
                f"| {r['delta']:+.3f}{star} | {r['n']} "
                f"| {r.get('p', float('nan')):.3g} | {r.get('p_adj', float('nan')):.3g} |\n"
            )
    logger.info(f"[table] wrote {out_dir}/tier1_main_table.{{csv,md}}")


# =============================================================================
# main
# =============================================================================


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--skip_t2i", action="store_true")
    ap.add_argument("--skip_i2v", action="store_true")
    ap.add_argument("--skip_vbench", action="store_true")
    ap.add_argument("--skip_judge", action="store_true")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # 1 + 2. T2I: generate first frames for both modes.
    if not args.skip_t2i:
        step_t2i(cfg, "base")
        step_t2i(cfg, "opsd")

    # 3. I2V: render videos with frozen DynamiCrafter for both.
    if not args.skip_i2v:
        step_i2v(cfg, "base")
        step_i2v(cfg, "opsd")

    # 4. VBench on both.
    if not args.skip_vbench:
        step_vbench(cfg, "base")
        step_vbench(cfg, "opsd")

    # 5. Qwen2-VL judge (OPSD vs base).
    if not args.skip_judge:
        step_judge(cfg)

    # 6 + 7. stats + table
    base_per = _load_vbench_per_prompt(Path(cfg["vbench_dir"]["base"]))
    opsd_per = _load_vbench_per_prompt(Path(cfg["vbench_dir"]["opsd"]))
    rows = paired_stats(base_per, opsd_per)

    out_dir = Path(cfg["out_dir"])
    (out_dir / "stats.json").parent.mkdir(parents=True, exist_ok=True)
    (out_dir / "stats.json").write_text(json.dumps(rows, indent=2))
    write_table(rows, out_dir)

    logger.info("[tier1] complete.")


if __name__ == "__main__":
    main()
