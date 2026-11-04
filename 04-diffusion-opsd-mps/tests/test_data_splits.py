from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower().strip()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def shingles(text: str, width: int = 8):
    words = normalize(text).split()
    return {tuple(words[i : i + width]) for i in range(max(0, len(words) - width + 1))}


def load(name: str):
    return [json.loads(line) for line in (DATA / name).read_text().splitlines() if line]


def test_train_dev_test_are_disjoint_and_unique():
    groups = {
        split: load(f"{split}_scene.jsonl") + load(f"{split}_portrait.jsonl")
        for split in ("train", "dev", "test")
    }
    for rows in groups.values():
        normalized = [normalize(row["prompt"]) for row in rows]
        assert len(normalized) == len(set(normalized))
    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        left_norm = {normalize(row["prompt"]) for row in groups[left]}
        right_norm = {normalize(row["prompt"]) for row in groups[right]}
        assert not left_norm & right_norm
        left_shingles = set().union(*(shingles(row["prompt"]) for row in groups[left]))
        right_shingles = set().union(*(shingles(row["prompt"]) for row in groups[right]))
        assert not left_shingles & right_shingles


def test_demo_rows_are_frozen_first_test_rows():
    for domain in ("scene", "portrait"):
        assert load(f"demo_{domain}.jsonl") == load(f"test_{domain}.jsonl")[:4]
