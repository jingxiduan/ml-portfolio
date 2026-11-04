"""Prepare frozen, leakage-audited scene and portrait prompt manifests."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import unicodedata
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PICKAPIC_ID = "sayakpaul/pick-a-pic-v2-unique-prompts"
PICKAPIC_REVISION = "ec6ade5fc615ec90152a78140492cec60da9680c"
PARTI_ID = "nateraw/parti-prompts"
SEED = 3407

PORTRAIT_TERMS = {
    "portrait", "headshot", "close up", "close-up", "face", "person", "people",
    "woman", "women", "man", "men", "girl", "boy", "lady", "gentleman",
    "elderly", "smiling", "model", "photographer", "chef", "artist", "musician",
}
SCENE_TERMS = {
    "landscape", "environment", "forest", "mountain", "valley", "city", "street",
    "room", "interior", "beach", "lake", "river", "building", "garden", "house",
    "sunset", "sunrise", "sky", "field", "ocean", "waterfall", "village", "canyon",
    "architecture", "alley", "harbor", "desert", "meadow", "coast", "cabin",
}
BLOCKED_TERMS = {
    "nude", "naked", "porn", "erotic", "sexual", "gore", "guro", "blood",
    "weapon", "gun", "rifle", "pistol", "bomb", "explosive", "murder", "corpse",
    "president", "prime minister", "senator", "politician", "election", "government",
}
SHOWCASE_PROMPTS = {
    "scene": [
        "A misty mountain valley at sunrise, layered pine forests, a winding river, cinematic natural light, highly detailed landscape photography.",
        "A quiet minimalist hotel lobby with travertine walls, indoor olive trees, soft skylight, warm neutral interior photography.",
        "A rain-soaked neon alley at blue hour, reflections on wet pavement, atmospheric urban photography, no people.",
        "A sunlit Mediterranean courtyard with white stone arches, a small fountain, climbing jasmine, serene architectural photography.",
    ],
    "portrait": [
        "Studio headshot of a middle-aged architect, neutral gray background, soft window light, natural skin texture, 85mm portrait photography.",
        "Cinematic close-up portrait of a jazz musician backstage, warm practical lights, expressive eyes, realistic editorial photography.",
        "Editorial portrait of a pastry chef in a bright professional kitchen, calm expression, soft daylight, realistic skin detail.",
        "Outdoor environmental portrait of a ceramic artist beside a workshop window, overcast light, natural colors, documentary photography.",
    ],
}


def normalize_prompt(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower().strip()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def shingles(text: str, width: int = 8) -> set[tuple[str, ...]]:
    tokens = normalize_prompt(text).split()
    if len(tokens) < width:
        return set()
    return {tuple(tokens[i : i + width]) for i in range(len(tokens) - width + 1)}


def safe_english_prompt(text: str) -> bool:
    normalized = normalize_prompt(text)
    if len(normalized.split()) < 3 or len(text) > 300:
        return False
    if any(term in normalized for term in BLOCKED_TERMS):
        return False
    ascii_letters = sum(ch.isascii() and ch.isalpha() for ch in text)
    letters = sum(ch.isalpha() for ch in text)
    return letters > 0 and ascii_letters / letters >= 0.9


def row(source: str, source_id: str, domain: str, prompt: str, category: str) -> dict:
    return {
        "source": source,
        "source_id": source_id,
        "domain": domain,
        "category": category,
        "prompt": prompt.strip(),
        "normalized_sha256": hashlib.sha256(normalize_prompt(prompt).encode()).hexdigest(),
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in rows))


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def overlaps(left: list[dict], right: list[dict]) -> tuple[set[str], set[tuple[str, ...]]]:
    left_norm = {normalize_prompt(item["prompt"]) for item in left}
    right_norm = {normalize_prompt(item["prompt"]) for item in right}
    left_shingles = set().union(*(shingles(item["prompt"]) for item in left)) if left else set()
    right_shingles = set().union(*(shingles(item["prompt"]) for item in right)) if right else set()
    return left_norm & right_norm, left_shingles & right_shingles


def select_parti(rows: list[dict], domain: str, dev_n: int, test_n: int) -> tuple[list[dict], list[dict]]:
    categories = {"People"} if domain == "portrait" else {"Outdoor Scenes", "Indoor Scenes"}
    candidates = []
    seen = set()
    for idx, item in enumerate(rows):
        prompt = str(item["Prompt"]).strip()
        norm = normalize_prompt(prompt)
        if item["Category"] not in categories or not safe_english_prompt(prompt) or norm in seen:
            continue
        seen.add(norm)
        candidates.append(row(PARTI_ID, str(idx), domain, prompt, str(item["Category"])))
    needed = dev_n + test_n
    if len(candidates) < needed:
        raise RuntimeError(f"only {len(candidates)} safe PartiPrompts rows for {domain}, need {needed}")
    return candidates[:dev_n], candidates[dev_n:needed]


def select_training(
    prompts: list[str], domain: str, count: int, forbidden: list[dict], rng: random.Random
) -> list[dict]:
    terms = PORTRAIT_TERMS if domain == "portrait" else SCENE_TERMS
    forbidden_norm = {normalize_prompt(item["prompt"]) for item in forbidden}
    forbidden_shingles = set().union(*(shingles(item["prompt"]) for item in forbidden))
    candidates = []
    seen = set()
    for idx, prompt in enumerate(prompts):
        prompt = str(prompt).strip()
        norm = normalize_prompt(prompt)
        if not safe_english_prompt(prompt) or not any(term in norm for term in terms):
            continue
        if norm in seen or norm in forbidden_norm or shingles(prompt) & forbidden_shingles:
            continue
        seen.add(norm)
        candidates.append(row(PICKAPIC_ID, str(idx), domain, prompt, "domain-filtered"))
    rng.shuffle(candidates)
    if len(candidates) < count:
        raise RuntimeError(f"only {len(candidates)} safe, disjoint Pick-a-Pic prompts for {domain}, need {count}")
    return candidates[:count]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-per-domain", type=int, default=768)
    parser.add_argument("--dev-per-domain", type=int, default=16)
    parser.add_argument("--test-per-domain", type=int, default=40)
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    parti = load_dataset(PARTI_ID, split="train")
    parti_rows = list(parti)
    portrait_dev, portrait_test = select_parti(
        parti_rows, "portrait", args.dev_per_domain, args.test_per_domain
    )
    scene_dev, scene_test = select_parti(
        parti_rows, "scene", args.dev_per_domain, args.test_per_domain
    )

    pickapic = load_dataset(
        PICKAPIC_ID,
        revision=PICKAPIC_REVISION,
        split="train",
        token=False,
    )
    rng = random.Random(SEED)
    portrait_train = select_training(
        list(pickapic["prompt"]),
        "portrait",
        args.train_per_domain,
        portrait_dev + portrait_test + scene_dev + scene_test,
        rng,
    )
    scene_train = select_training(
        list(pickapic["prompt"]),
        "scene",
        args.train_per_domain,
        portrait_dev + portrait_test + scene_dev + scene_test + portrait_train,
        rng,
    )

    showcase_scene = [
        row("portfolio-authored", str(index), "scene", prompt, "showcase")
        for index, prompt in enumerate(SHOWCASE_PROMPTS["scene"])
    ]
    showcase_portrait = [
        row("portfolio-authored", str(index), "portrait", prompt, "showcase")
        for index, prompt in enumerate(SHOWCASE_PROMPTS["portrait"])
    ]
    manifests = {
        "train_scene.jsonl": scene_train,
        "train_portrait.jsonl": portrait_train,
        "dev_scene.jsonl": scene_dev,
        "dev_portrait.jsonl": portrait_dev,
        "test_scene.jsonl": scene_test,
        "test_portrait.jsonl": portrait_test,
        "showcase_scene.jsonl": showcase_scene,
        "showcase_portrait.jsonl": showcase_portrait,
    }
    for name, items in manifests.items():
        write_jsonl(DATA_DIR / name, items)

    split_groups = {
        "train": scene_train + portrait_train,
        "dev": scene_dev + portrait_dev,
        "test": scene_test + portrait_test,
        "showcase": showcase_scene + showcase_portrait,
    }
    audit = {}
    for left_name, right_name in (
        ("train", "dev"),
        ("train", "test"),
        ("dev", "test"),
        ("train", "showcase"),
        ("dev", "showcase"),
        ("test", "showcase"),
    ):
        exact, near = overlaps(split_groups[left_name], split_groups[right_name])
        audit[f"{left_name}_vs_{right_name}"] = {
            "exact_overlap": len(exact),
            "shared_8_token_shingles": len(near),
        }
        if exact or near:
            raise RuntimeError(f"leakage detected: {left_name} vs {right_name}")

    source_info = HfApi().dataset_info(PARTI_ID)
    report = {
        "seed": SEED,
        "normalization": "Unicode NFKC, lowercase, punctuation/whitespace normalization",
        "near_duplicate_rule": "no shared contiguous 8-token shingle across train/dev/test/showcase",
        "sources": {
            "training": {"dataset": PICKAPIC_ID, "revision": PICKAPIC_REVISION},
            "evaluation": {"dataset": PARTI_ID, "revision": source_info.sha},
        },
        "counts": {name: len(items) for name, items in manifests.items()},
        "audit": audit,
        "manifest_sha256": {name: file_sha256(DATA_DIR / name) for name in manifests},
        "showcase_policy": "four portfolio-authored prompts per domain, frozen before generation and never selected from generated output",
    }
    (DATA_DIR / "split_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
