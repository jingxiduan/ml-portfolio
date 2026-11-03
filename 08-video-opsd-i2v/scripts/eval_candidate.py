"""Paired n>=40 held-out evaluation for one I2V candidate against base."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

METRICS = ("appeal", "nat", "dynamic")


def render(
    tag: str,
    lora_dir: Optional[Path],
    out_dir: Path,
    frames_dir: Path,
    prompts: Path,
    limit: int,
) -> None:
    target = out_dir / f"videos_{tag}"
    target.mkdir(parents=True, exist_ok=True)
    existing = list((target / "seed_00").glob("*.mp4"))
    if len(existing) >= limit:
        print(f"[render] {tag}: reuse {len(existing)} videos")
        return
    if lora_dir is not None and not lora_dir.is_dir():
        raise SystemExit(f"missing candidate checkpoint: {lora_dir}")
    command = [
        sys.executable,
        "-m",
        "src.render_i2v",
        "--frames_dir",
        str(frames_dir),
        "--prompts",
        str(prompts),
        "--out_dir",
        str(target),
        "--seeds",
        "0",
        "--num_frames",
        "16",
        "--fps",
        "8",
        "--limit",
        str(limit),
    ]
    if lora_dir is not None:
        command.extend(["--i2v_lora_dir", str(lora_dir)])
    start = time.time()
    subprocess.run(command, check=True, env={**os.environ, "PYTHONPATH": "."})
    count = len(list((target / "seed_00").glob("*.mp4")))
    if count != limit:
        raise SystemExit(f"{tag}: expected {limit} videos, rendered {count}")
    print(f"[render] {tag}: {count} videos in {time.time() - start:.0f}s")


def metrics(video_dir: Path) -> list[dict[str, float | str]]:
    import imageio.v3 as iio
    from src.eval.naturalness_v2 import naturalness_v2_video

    rows = []
    for video in sorted((video_dir / "seed_00").glob("*.mp4")):
        frames = np.stack(list(iio.imiter(video)))
        values = naturalness_v2_video(frames)
        rows.append({"file": video.name, **{key: float(values[key]) for key in METRICS}})
    return rows


def bootstrap(values: np.ndarray, seed: int, draws: int = 10000) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "lo": float(np.quantile(means, 0.025)),
        "hi": float(np.quantile(means, 0.975)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-tag", required=True)
    parser.add_argument("--candidate-lora", type=Path, required=True)
    parser.add_argument("--frames-dir", type=Path, default=Path("data/first_frames_eval"))
    parser.add_argument("--prompts", type=Path, default=Path("data/prompts/tier2_eval_40.json"))
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--out", type=Path, default=Path("outputs/eval_candidate"))
    parser.add_argument("--metrics-only", action="store_true")
    args = parser.parse_args()
    if args.limit < 40:
        raise SystemExit("confirmatory evaluation requires at least 40 paired prompts")

    args.out.mkdir(parents=True, exist_ok=True)
    if not args.metrics_only:
        render("base", None, args.out, args.frames_dir, args.prompts, args.limit)
        render(args.candidate_tag, args.candidate_lora, args.out, args.frames_dir, args.prompts, args.limit)

    base = metrics(args.out / "videos_base")
    candidate = metrics(args.out / f"videos_{args.candidate_tag}")
    base_map = {row["file"]: row for row in base}
    candidate_map = {row["file"]: row for row in candidate}
    common = sorted(set(base_map) & set(candidate_map))
    if len(common) < 40:
        raise SystemExit(f"only {len(common)} paired videos available")

    comparisons = {}
    summaries = {}
    for tag, rows in (("base", base), (args.candidate_tag, candidate)):
        summaries[tag] = {key: float(np.mean([float(row[key]) for row in rows])) for key in METRICS}
        summaries[tag]["n"] = len(rows)
    for index, key in enumerate(METRICS):
        delta = np.array([float(candidate_map[name][key]) - float(base_map[name][key]) for name in common])
        comparisons[key] = bootstrap(delta, seed=500 + index)
        comparisons[key]["n_pairs"] = len(common)
        comparisons[key]["favorable_ci_excludes_zero"] = bool(comparisons[key]["lo"] > 0)

    result = {
        "candidate": args.candidate_tag,
        "candidate_lora": str(args.candidate_lora),
        "paired_generation_seed": 0,
        "summaries": summaries,
        "candidate_minus_base": comparisons,
        "per_video": {"base": base, args.candidate_tag: candidate},
    }
    output = args.out / f"metrics_{args.candidate_tag}.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"summaries": summaries, "candidate_minus_base": comparisons}, indent=2))
    print(f"saved {output}")


if __name__ == "__main__":
    main()
