"""Prepare a 6-pair dev split and untouched 40-pair COCO validation I2V split."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.parse
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

META_DATASET = "phiyodr/coco2017"
IMAGE_DATASET = "lance-format/coco-captions-2017-lance"
ALLOWED_LICENSES = {
    4: "CC BY 2.0",
    5: "CC BY-SA 2.0",
    7: "No known copyright restrictions",
    8: "United States Government Work",
}


def fetch_rows(dataset: str, offset: int, length: int = 100) -> list[dict]:
    query = urllib.parse.urlencode(
        {"dataset": dataset, "config": "default", "split": "validation", "offset": offset, "length": length}
    )
    request = urllib.request.Request(
        f"https://datasets-server.huggingface.co/rows?{query}",
        headers={"User-Agent": "ml-portfolio-heldout-eval/1.0"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)["rows"]


def download(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        destination.write_bytes(response.read())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-count", type=int, default=6)
    parser.add_argument("--eval-count", type=int, default=40)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/coco_eval46_cache"))
    parser.add_argument("--dev-dir", type=Path, default=Path("data/first_frames_dev_coco6"))
    parser.add_argument("--eval-dir", type=Path, default=Path("data/first_frames_eval_coco40"))
    parser.add_argument("--dev-prompts", type=Path, default=Path("data/prompts/coco_dev_6.json"))
    parser.add_argument("--eval-prompts", type=Path, default=Path("data/prompts/coco_eval_40.json"))
    args = parser.parse_args()
    total = args.dev_count + args.eval_count

    metadata_parquet = args.cache_dir / "coco_validation_metadata.parquet"
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    if not metadata_parquet.is_file():
        download(
            "https://huggingface.co/api/datasets/phiyodr/coco2017/parquet/default/validation/0.parquet",
            metadata_parquet,
        )
    table = pq.read_table(metadata_parquet)
    metadata_by_id = {int(row["image_id"]): row for row in table.to_pylist()}

    selected = []
    offset = 0
    while len(selected) < total:
        for wrapped in fetch_rows(IMAGE_DATASET, offset):
            row = wrapped["row"]
            image_id = int(Path(row["filename"]).stem)
            metadata = metadata_by_id.get(image_id)
            if metadata is None or int(metadata["license"]) not in ALLOWED_LICENSES:
                continue
            selected.append((row, metadata))
            if len(selected) == total:
                break
        offset += 100

    cache_images = args.cache_dir / "images"
    cache_images.mkdir(exist_ok=True)
    split_specs = [
        ("development", selected[: args.dev_count], args.dev_dir, args.dev_prompts),
        ("confirmatory", selected[args.dev_count :], args.eval_dir, args.eval_prompts),
    ]
    for split_name, items, split_dir, prompt_path in split_specs:
        image_dir = split_dir / "seed_00"
        image_dir.mkdir(parents=True, exist_ok=True)
        prompts = []
        sources = []
        for split_index, (row, metadata) in enumerate(items):
            image_id = int(metadata["image_id"])
            cached = cache_images / f"{image_id}.png"
            if not cached.is_file():
                temporary = cached.with_suffix(".download")
                download(row["image"]["src"], temporary)
                with Image.open(temporary) as image:
                    image.convert("RGB").save(cached, format="PNG")
                temporary.unlink()
            output = image_dir / f"{split_index:03d}.png"
            shutil.copyfile(cached, output)
            caption = str(row["caption"]).rstrip(".")
            prompt = f"{caption}, gentle natural motion, stable cinematic camera"
            prompts.append({"id": f"coco-{split_name}-{split_index:03d}", "prompt": prompt})
            license_id = int(metadata["license"])
            sources.append(
                {
                    "index": split_index,
                    "image_id": image_id,
                    "file": str(output),
                    "caption": caption,
                    "coco_url": metadata["coco_url"],
                    "flickr_url": metadata["flickr_url"],
                    "license_id": license_id,
                    "license": ALLOWED_LICENSES[license_id],
                    "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                }
            )
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(json.dumps(prompts, indent=2) + "\n")
        (split_dir / "SOURCES.json").write_text(
            json.dumps(
                {
                    "metadata_dataset": META_DATASET,
                    "image_dataset": IMAGE_DATASET,
                    "split": "validation",
                    "role": split_name,
                    "selection": "dataset order after modification-compatible license filter 4/5/7/8",
                    "used_for_training": False,
                    "items": sources,
                },
                indent=2,
            )
            + "\n"
        )
        print(f"prepared {len(items)} {split_name} pairs in {split_dir}")


if __name__ == "__main__":
    main()
