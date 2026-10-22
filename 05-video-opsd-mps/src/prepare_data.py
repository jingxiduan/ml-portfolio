"""
Prepare the motion-prompt dataset for the video OPSD project.

Unlike `04-diffusion-opsd-mps` (which reuses the original DiffusionOPSD repo's
public DrawBench prompt list), DrawBench prompts describe static scenes and
are a poor fit for *video* generation -- there is nothing for a video model
to animate. So this project uses a small, hand-authored list of 30 prompts
that each describe a visually verifiable motion (water flowing, something
spinning/falling/flying/rising, etc.), stored in `data/prompts_all.txt`.
This script only splits that list into disjoint train / eval / demo subsets.
"""
import argparse
import random
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-train", type=int, default=18)
    parser.add_argument("--n-eval", type=int, default=9)
    parser.add_argument("--n-demo", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    lines = [l.strip() for l in (DATA_DIR / "prompts_all.txt").read_text().splitlines() if l.strip()]
    total_needed = args.n_train + args.n_eval + args.n_demo
    assert total_needed <= len(lines), "not enough prompts for requested split"

    rng = random.Random(args.seed)
    shuffled = lines[:]
    rng.shuffle(shuffled)

    train = shuffled[: args.n_train]
    eval_ = shuffled[args.n_train: args.n_train + args.n_eval]
    demo = shuffled[args.n_train + args.n_eval: total_needed]

    (DATA_DIR / "train_prompts.txt").write_text("\n".join(train) + "\n")
    (DATA_DIR / "eval_prompts.txt").write_text("\n".join(eval_) + "\n")
    (DATA_DIR / "demo_prompts.txt").write_text("\n".join(demo) + "\n")

    print(f"[ok] train={len(train)} eval={len(eval_)} demo={len(demo)} "
          f"(disjoint subsets of the hand-authored motion-prompt list, seed={args.seed})")


if __name__ == "__main__":
    main()
