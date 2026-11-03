"""Select one V4 checkpoint/scale on the fixed six-pair development split."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import imageio.v3 as iio
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.naturalness_v2 import naturalness_v2_video  # noqa: E402
from src.t2i.lora_scale import prepare_scaled_lora  # noqa: E402

METRICS = ("appeal", "nat", "dynamic")


def render(tag: str, adapter: Optional[Path], out_root: Path, frames: Path, prompts: Path, count: int) -> Path:
    target = out_root / tag
    videos = target / "seed_00"
    if len(list(videos.glob("*.mp4"))) >= count:
        return videos
    command = [
        sys.executable,
        "-m",
        "src.render_i2v",
        "--frames_dir",
        str(frames),
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
        str(count),
    ]
    if adapter is not None:
        command.extend(["--i2v_lora_dir", str(adapter)])
    subprocess.run(command, cwd=ROOT, check=True, env={**os.environ, "PYTHONPATH": str(ROOT)})
    return videos


def evaluate(directory: Path) -> dict[str, float]:
    rows = []
    for video in sorted(directory.glob("*.mp4")):
        frames = np.stack(list(iio.imiter(video)))
        rows.append(naturalness_v2_video(frames))
    if len(rows) != 6:
        raise SystemExit(f"expected six development videos in {directory}, found {len(rows)}")
    return {key: float(np.mean([float(row[key]) for row in rows])) for key in METRICS}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=ROOT / "outputs" / "opsd_long_v4_gradclip")
    parser.add_argument("--frames", type=Path, default=ROOT / "data" / "first_frames_dev_coco6")
    parser.add_argument("--prompts", type=Path, default=ROOT / "data" / "prompts" / "coco_dev_6.json")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "v4_dev_selection")
    parser.add_argument("--checkpoints", type=int, nargs="+", default=[500, 1000, 1500, 2000])
    parser.add_argument("--scales", type=float, nargs="+", default=[0.01, 0.02, 0.05])
    parser.add_argument("--nat-floor", type=float, default=-0.02)
    parser.add_argument("--appeal-floor", type=float, default=-0.02)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    base = evaluate(render("base", None, args.out, args.frames, args.prompts, 6))
    candidates = []
    for step in args.checkpoints:
        checkpoint = args.run_dir / f"lora_step{step:05d}"
        if not checkpoint.is_dir():
            raise SystemExit(f"missing pre-registered checkpoint: {checkpoint}")
        for scale in args.scales:
            adapter = prepare_scaled_lora(checkpoint, scale=scale, dst_root=args.out / "scaled_adapters")
            tag = f"step{step:04d}_scale{scale:.2f}"
            values = evaluate(render(tag, adapter, args.out, args.frames, args.prompts, 6))
            delta = {key: values[key] - base[key] for key in METRICS}
            eligible = delta["nat"] >= args.nat_floor and delta["appeal"] >= args.appeal_floor
            score = delta["appeal"] + 0.5 * delta["nat"] + 0.5 * delta["dynamic"]
            candidates.append(
                {
                    "tag": tag,
                    "step": step,
                    "scale": scale,
                    "adapter": str(adapter),
                    "metrics": values,
                    "delta_vs_base": delta,
                    "eligible": eligible,
                    "selection_score": score,
                }
            )
            print(tag, json.dumps(candidates[-1], sort_keys=True), flush=True)

    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    if not eligible:
        raise SystemExit("no candidate satisfies the frozen naturalness/appeal floors")
    selected = sorted(eligible, key=lambda item: (-item["selection_score"], item["scale"], item["step"]))[0]
    result = {
        "development_only": True,
        "n_pairs": 6,
        "base": base,
        "grid": {"checkpoints": args.checkpoints, "scales": args.scales},
        "eligibility": {"delta_nat_min": args.nat_floor, "delta_appeal_min": args.appeal_floor},
        "score": "delta_appeal + 0.5*delta_nat + 0.5*delta_dynamic",
        "selected": selected,
        "candidates": candidates,
    }
    output = args.out / "selection.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"selected {selected['tag']} -> {selected['adapter']}")
    print(f"saved {output}")


if __name__ == "__main__":
    main()
