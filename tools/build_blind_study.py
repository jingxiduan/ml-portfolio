"""Build a blinded, balanced A/B study package for projects 07 and 08."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import secrets
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def opaque_id(project: str, index: int, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{project}:{index}".encode()).hexdigest()[:12]


def copy_pair(
    project: str,
    index: int,
    base: Path,
    candidate: Path,
    prompt: str,
    candidate_left: bool,
    salt: str,
    media_dir: Path,
) -> tuple[dict, dict]:
    trial_id = opaque_id(project, index, salt)
    left_source, right_source = (candidate, base) if candidate_left else (base, candidate)
    left_name = f"{trial_id}_A.mp4"
    right_name = f"{trial_id}_B.mp4"
    shutil.copyfile(left_source, media_dir / left_name)
    shutil.copyfile(right_source, media_dir / right_name)
    public = {
        "trial_id": trial_id,
        "project": project,
        "source_index": index,
        "prompt": prompt,
        "video_a": f"media/{left_name}",
        "video_b": f"media/{right_name}",
    }
    private = {
        "trial_id": trial_id,
        "project": project,
        "source_index": index,
        "candidate_side": "A" if candidate_left else "B",
        "base_side": "B" if candidate_left else "A",
        "base_source": str(base.relative_to(ROOT)),
        "candidate_source": str(candidate.relative_to(ROOT)),
    }
    return public, private


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "human-study")
    parser.add_argument("--seed", type=int, default=None, help="private random seed; generated when omitted")
    parser.add_argument("--salt", default=None, help="private trial-ID salt; generated when omitted")
    args = parser.parse_args()
    seed = args.seed if args.seed is not None else secrets.randbits(63)
    salt = args.salt if args.salt is not None else secrets.token_hex(16)
    out = args.out.resolve()
    media = out / "media"
    out.mkdir(parents=True, exist_ok=True)
    if media.exists():
        shutil.rmtree(media)
    for generated in ("study.json", "private_key.json", "responses_template.csv"):
        (out / generated).unlink(missing_ok=True)
    media.mkdir(parents=True)
    rng = random.Random(seed)
    public_trials, private_key = [], []
    sides = {}
    for project in ("07", "08"):
        assignment = [True] * 20 + [False] * 20
        rng.shuffle(assignment)
        sides[project] = assignment

    manifest07 = json.loads((ROOT / "07-talking-head-motion-opsd/outputs/expression_render_set_r3/manifest.json").read_text())
    entries07 = {(int(item["clip_index"]), item["variant"]): item for item in manifest07["entries"]}
    for index in range(40):
        public, private = copy_pair(
            "07",
            index,
            ROOT / "07-talking-head-motion-opsd" / entries07[(index, "base")]["video"],
            ROOT / "07-talking-head-motion-opsd" / entries07[(index, "seed0")]["video"],
            "Which video has more natural and appropriate facial/head motion while preserving lip synchronization?",
            sides["07"][index],
            salt,
            media,
        )
        public_trials.append(public)
        private_key.append(private)

    base08 = ROOT / "08-video-opsd-i2v/outputs/eval_v4/videos_base/seed_00"
    candidate08 = ROOT / "08-video-opsd-i2v/outputs/eval_v4/videos_step1000_scale0.05/seed_00"
    prompts08 = json.loads((ROOT / "08-video-opsd-i2v/data/prompts/coco_eval_40.json").read_text())
    for index, item in enumerate(prompts08):
        public, private = copy_pair(
            "08",
            index,
            base08 / f"{index:03d}.mp4",
            candidate08 / f"{index:03d}.mp4",
            item["prompt"],
            sides["08"][index],
            salt,
            media,
        )
        public_trials.append(public)
        private_key.append(private)

    rng.shuffle(public_trials)
    study = {
        "version": 1,
        "n_trials": len(public_trials),
        "instructions": "For each pair choose A, B, or Tie. Judge overall visible quality for project 08 and natural motion/lip preservation for project 07.",
        "trials": public_trials,
    }
    (out / "study.json").write_text(json.dumps(study, indent=2) + "\n")
    (out / "private_key.json").write_text(
        json.dumps({"version": 1, "seed": seed, "salt": salt, "key": private_key}, indent=2) + "\n"
    )
    template = "participant_id,trial_id,choice\n" + "\n".join(
        f",{trial['trial_id']}," for trial in public_trials
    ) + "\n"
    (out / "responses_template.csv").write_text(template)
    print(f"built {len(public_trials)} blinded trials in {out}")


if __name__ == "__main__":
    main()
