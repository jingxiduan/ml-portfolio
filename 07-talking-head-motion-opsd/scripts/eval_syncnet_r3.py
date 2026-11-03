"""Independent SyncNet LSE-C/LSE-D evaluation for the rendered R3 set."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SYNCNET = ROOT / "third_party" / "syncnet_python"
sys.path.insert(0, str(SYNCNET))

from SyncNetInstance import SyncNetInstance  # noqa: E402


def bootstrap_ci(values: np.ndarray, seed: int, draws: int = 10000) -> list[float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def prepare_face_video(source: Path, destination: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise SystemExit("ffmpeg is required")
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-vf",
            "crop='min(iw,ih)*0.72':'min(iw,ih)*0.72':(iw-ow)/2:(ih-oh)/2,scale=224:224,fps=25",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:v",
            "mpeg4",
            "-q:v",
            "2",
            "-c:a",
            "pcm_s16le",
            str(destination),
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "outputs" / "expression_render_set_r3" / "manifest.json",
    )
    parser.add_argument("--model", type=Path, default=ROOT / "weights" / "syncnet" / "syncnet_v2.model")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "syncnet_eval_r3.json")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--vshift", type=int, default=15)
    parser.add_argument("--margin", type=float, default=0.5)
    args = parser.parse_args()

    if not args.manifest.is_file() or not args.model.is_file():
        raise SystemExit("render manifest or SyncNet model is missing")
    manifest = json.loads(args.manifest.read_text())
    entries = manifest["entries"]
    expected = int(manifest["n_clips"]) * 4
    if len(entries) != expected:
        raise SystemExit(f"expected {expected} rendered entries, found {len(entries)}")

    model = SyncNetInstance()
    model.loadParameters(str(args.model))
    rows = []
    with tempfile.TemporaryDirectory(prefix="syncnet-r3-") as temp:
        temp_path = Path(temp)
        for index, entry in enumerate(entries):
            source = ROOT / entry["video"]
            prepared = temp_path / f"prepared-{index:03d}.avi"
            prepare_face_video(source, prepared)
            opt = SimpleNamespace(
                tmp_dir=str(temp_path / "work"),
                reference=f"clip-{index:03d}",
                batch_size=args.batch_size,
                vshift=args.vshift,
            )
            offset, confidence, distances = model.evaluate(opt, videofile=str(prepared))
            mean_distances = np.asarray(distances).mean(axis=0)
            row = {
                "clip_index": int(entry["clip_index"]),
                "variant": entry["variant"],
                "offset_frames": int(offset),
                "lse_c": float(confidence),
                "lse_d": float(mean_distances.min()),
            }
            rows.append(row)
            print(
                f"[syncnet] {index + 1}/{len(entries)} {row['variant']} "
                f"offset={row['offset_frames']} LSE-C={row['lse_c']:.3f} LSE-D={row['lse_d']:.3f}",
                flush=True,
            )

    by_variant = {
        name: sorted([row for row in rows if row["variant"] == name], key=lambda row: row["clip_index"])
        for name in ("base", "seed0", "seed1", "seed2")
    }
    base_c = np.array([row["lse_c"] for row in by_variant["base"]])
    base_d = np.array([row["lse_d"] for row in by_variant["base"]])
    summaries = {}
    for seed_index, (name, values) in enumerate(by_variant.items()):
        lse_c = np.array([row["lse_c"] for row in values])
        lse_d = np.array([row["lse_d"] for row in values])
        delta_c = lse_c - base_c
        delta_d = lse_d - base_d
        ci_c = bootstrap_ci(delta_c, seed=300 + seed_index)
        ci_d = bootstrap_ci(delta_d, seed=400 + seed_index)
        summaries[name] = {
            "n": len(values),
            "lse_c_mean": float(lse_c.mean()),
            "lse_d_mean": float(lse_d.mean()),
            "offset_abs_mean": float(np.mean(np.abs([row["offset_frames"] for row in values]))),
            "delta_lse_c_vs_base_mean": float(delta_c.mean()),
            "delta_lse_c_vs_base_ci95": ci_c,
            "delta_lse_d_vs_base_mean": float(delta_d.mean()),
            "delta_lse_d_vs_base_ci95": ci_d,
            "noninferiority_gate": bool(name == "base" or (ci_c[0] >= -args.margin and ci_d[1] <= args.margin)),
        }

    result = {
        "metric": "SyncNet LSE-C/LSE-D",
        "implementation": "joonson/syncnet_python@907c0b5",
        "model": args.model.name,
        "crop": "center 72%, 224x224, 25 fps",
        "paired": True,
        "noninferiority_margin": args.margin,
        "summaries": summaries,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(summaries, indent=2))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
