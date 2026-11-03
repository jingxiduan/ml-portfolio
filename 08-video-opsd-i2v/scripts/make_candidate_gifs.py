"""Create visible base/candidate GIF demos from paired held-out I2V renders."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from make_eval_gifs import make_sidebyside_gif

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, default=ROOT / "outputs/eval_candidate/videos_base/seed_00")
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--candidate-label", default="Candidate")
    parser.add_argument("--prompts", type=Path, default=ROOT / "data" / "prompts" / "coco_eval_40.json")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "docs/figures/eval_candidate")
    parser.add_argument("--indices", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    args = parser.parse_args()
    args.base_dir = args.base_dir.resolve()
    args.candidate_dir = args.candidate_dir.resolve()
    args.prompts = args.prompts.resolve()
    args.out_dir = args.out_dir.resolve()

    prompts = json.loads(args.prompts.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for index in args.indices:
        base = args.base_dir / f"{index:03d}.mp4"
        candidate = args.candidate_dir / f"{index:03d}.mp4"
        if not base.is_file() or not candidate.is_file():
            raise SystemExit(f"missing paired videos for index {index}")
        output = args.out_dir / f"eval_{index:03d}_base_vs_candidate.gif"
        make_sidebyside_gif(base, candidate, output, labels=("Base", args.candidate_label))
        prompt = prompts[index]["prompt"] if isinstance(prompts[index], dict) else prompts[index]
        manifest.append({"index": index, "prompt": prompt, "gif": str(output.relative_to(ROOT))})
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"saved {len(manifest)} visible demos to {args.out_dir}")


if __name__ == "__main__":
    main()
