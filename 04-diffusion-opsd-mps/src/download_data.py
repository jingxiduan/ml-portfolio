"""
Download the DrawBench prompt list (public, from the original worldbench/DiffusionOPSD
repo's `data/drawbench/test.txt`, itself derived from the DrawBench benchmark released
with the Imagen paper) and split it into a small train / eval subset sized for a
single-GPU-less (MPS) run.
"""
import argparse
import random
from pathlib import Path

import requests

DRAWBENCH_URL = (
    "https://raw.githubusercontent.com/worldbench/DiffusionOPSD/main/"
    "data/drawbench/test.txt"
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def download_raw(dest: Path) -> list[str]:
    if dest.exists():
        print(f"[skip] {dest} already exists")
    else:
        resp = requests.get(DRAWBENCH_URL, timeout=30)
        resp.raise_for_status()
        dest.write_text(resp.text)
        print(f"[ok] downloaded {dest}")
    lines = [l.strip() for l in dest.read_text().splitlines() if l.strip()]
    print(f"[info] {len(lines)} prompts total")
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-train", type=int, default=40,
                         help="number of prompts used as the OPSD training pool")
    parser.add_argument("--n-eval", type=int, default=20,
                         help="number of held-out prompts used only for CLIPScore evaluation")
    parser.add_argument("--n-demo", type=int, default=4,
                         help="number of prompts used for the before/after visual demo")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = DATA_DIR / "drawbench_raw.txt"
    lines = download_raw(raw_path)

    rng = random.Random(args.seed)
    shuffled = lines[:]
    rng.shuffle(shuffled)

    total_needed = args.n_train + args.n_eval + args.n_demo
    assert total_needed <= len(shuffled), "not enough DrawBench prompts for requested split"

    train = shuffled[: args.n_train]
    eval_ = shuffled[args.n_train: args.n_train + args.n_eval]
    demo = shuffled[args.n_train + args.n_eval: total_needed]

    (DATA_DIR / "train_prompts.txt").write_text("\n".join(train) + "\n")
    (DATA_DIR / "eval_prompts.txt").write_text("\n".join(eval_) + "\n")
    (DATA_DIR / "demo_prompts.txt").write_text("\n".join(demo) + "\n")

    print(f"[ok] train={len(train)} eval={len(eval_)} demo={len(demo)} "
          f"(disjoint subsets of DrawBench, seed={args.seed})")


if __name__ == "__main__":
    main()
