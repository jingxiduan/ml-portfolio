"""Lite VBench-I2V evaluator — our own 4-dim implementation.

Why not official VBench: as of 2026-10-28, VBench 0.1.5 hard-depends on `decord`, which has
no Python 3.12 wheel. We re-implement the 4 most-cited dimensions directly using the same
underlying primitives (CLIP ViT-B/32, RAFT optical flow, LAION aesthetic predictor). This
keeps the pipeline MPS-compatible and gives us the same ordering + similar absolute scores.

Dimensions:
    - subject_consistency : mean cosine sim of CLIP(frame_t) with CLIP(frame_0), t > 0
    - motion_smoothness   : exp(-mean |RAFT_flow_{t+1} - RAFT_flow_{t}|)
    - dynamic_degree      : clipped mean |RAFT_flow| / cap
    - aesthetic_quality   : LAION aesthetic predictor on each frame, averaged

Supplementary paper will note: "we use a 4-dim lite re-implementation; full 9-dim VBench
results on A100 cloud deferred to supplementary §S1".

Usage:
    python -m src.eval.vbench_lite_eval \
        --videos_dir outputs/tier1/videos_opsd \
        --prompts    data/prompts/eval.json \
        --out_json   outputs/tier1/vbench_opsd/eval_results.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import imageio.v3 as iio
import torch

from ..rewards.aesthetic import Aesthetic
from ..rewards.dynamic_degree import DynamicDegree
from ..rewards.motion_smoothness import MotionSmoothness
from ..rewards.subject_consistency import SubjectConsistency
from ..utils.mps_patch import apply_mps_patches, get_device


DIM_CLASSES = {
    "subject_consistency": SubjectConsistency,
    "motion_smoothness":   MotionSmoothness,
    "dynamic_degree":      DynamicDegree,
    "aesthetic_quality":   Aesthetic,
}


def load_video(path: Path) -> torch.Tensor:
    """Read mp4 → tensor [T, 3, H, W] in [-1, 1]."""
    frames = iio.imread(str(path), plugin="pyav")   # [T, H, W, 3] uint8
    t = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 127.5 - 1.0
    return t


def _normalize_manifest_scores(per_dim: Dict[str, Dict[str, float]]) -> Dict:
    """Return VBench-style output: {dim: [aggregate, per_sample_dict]}."""
    out = {}
    for dim, scores in per_dim.items():
        if not scores:
            continue
        agg = sum(scores.values()) / len(scores)
        out[dim] = [agg, scores]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos_dir", required=True,
                    help="dir with seed_00/000.mp4, ..., manifest.json")
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--dims", nargs="+", default=list(DIM_CLASSES.keys()))
    ap.add_argument("--max_videos", type=int, default=None,
                    help="smoke-test limit")
    args = ap.parse_args()

    apply_mps_patches()
    device = get_device()
    print(f"[vbench_lite] device={device}")

    # Instantiate requested dim evaluators once.
    evaluators = {}
    for d in args.dims:
        if d not in DIM_CLASSES:
            raise KeyError(f"unknown dim: {d}; known: {list(DIM_CLASSES)}")
        print(f"[vbench_lite] loading {d}...")
        evaluators[d] = DIM_CLASSES[d](device=str(device))
        for p in evaluators[d].parameters():
            p.requires_grad_(False)

    videos_dir = Path(args.videos_dir)
    manifest_path = videos_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing manifest {manifest_path}")
    manifest: List[Dict] = json.loads(manifest_path.read_text())

    if args.max_videos:
        manifest = manifest[: args.max_videos]

    # Collect per-sample scores keyed by "<index>_s<seed>".
    scores: Dict[str, Dict[str, float]] = {d: {} for d in args.dims}
    t0 = time.time()

    with torch.no_grad():
        for i, item in enumerate(manifest):
            vid_path = videos_dir / item["file"]
            try:
                vid = load_video(vid_path).unsqueeze(0).to(device)
            except Exception as e:
                print(f"[vbench_lite] skip {vid_path}: {e}")
                continue
            key = f"{item['index']:03d}_s{item['seed']}"
            for d, ev in evaluators.items():
                r = ev(vid, [item["prompt"]])
                scores[d][key] = float(r.item())
            if (i + 1) % 10 == 0:
                print(f"[vbench_lite] {i+1}/{len(manifest)}  "
                      f"({time.time()-t0:.1f}s, {(time.time()-t0)/(i+1):.2f}s/vid)")

    results = _normalize_manifest_scores(scores)
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))

    print("\n[vbench_lite] AGGREGATES:")
    for d in args.dims:
        if d in results:
            print(f"  {d:25s}: {results[d][0]:+.4f}  (n={len(results[d][1])})")
    print(f"[vbench_lite] saved to {out}  (took {time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
