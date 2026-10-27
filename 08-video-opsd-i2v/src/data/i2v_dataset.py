"""I2V dataset: pairs (first_frame_image, prompt) with on-the-fly conditioning."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class I2VDataset(Dataset):
    """Returns ``{first_frame: [3,H,W] in [-1,1], prompt: str, cond: dict}``.

    Prompt list format (JSON array):
        [{"image": "000.png", "prompt": "a cat on a windowsill"}, ...]

    If ``image`` is missing we try ``<index:03d>.png`` fallback.
    """

    def __init__(
        self,
        first_frame_dir: str | Path,
        prompts_json: str | Path,
        resolution: Tuple[int, int] = (320, 512),
        num_frames: int = 16,
        fps: int = 8,
    ) -> None:
        super().__init__()
        self.root = Path(first_frame_dir)
        self.resolution = resolution
        self.num_frames = num_frames
        self.fps = fps

        with open(prompts_json) as f:
            self.entries: List[Dict] = json.load(f)
        if not self.entries:
            raise RuntimeError(f"Empty prompts file: {prompts_json}")

        H, W = resolution
        self.tf = transforms.Compose(
            [
                transforms.Resize((H, W), antialias=True),
                transforms.CenterCrop((H, W)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # -> [-1, 1]
            ]
        )

    def __len__(self) -> int:
        return len(self.entries)

    def _resolve_image(self, entry: Dict, idx: int) -> Path:
        if "image" in entry:
            p = self.root / entry["image"]
            if p.is_file():
                return p
        p = self.root / f"{idx:03d}.png"
        if p.is_file():
            return p
        p = self.root / f"{idx:03d}.jpg"
        if p.is_file():
            return p
        raise FileNotFoundError(
            f"No image for entry {idx}; tried {entry.get('image')}, {idx:03d}.png/.jpg"
        )

    def __getitem__(self, idx: int) -> Dict:
        entry = self.entries[idx]
        img_path = self._resolve_image(entry, idx)
        img = Image.open(img_path).convert("RGB")
        tensor = self.tf(img)                                 # [3, H, W] in [-1, 1]

        prompt: str = entry["prompt"]

        cond = {
            "text": prompt,
            "video_length": self.num_frames,
            "fps": self.fps,
        }
        return {"first_frame": tensor, "prompt": prompt, "cond": cond}


def collate_i2v(batch: List[Dict]) -> Dict:
    """Simple collate that stacks images and keeps prompts as a list."""
    first_frame = torch.stack([b["first_frame"] for b in batch], dim=0)
    prompts = [b["prompt"] for b in batch]
    cond_keys = batch[0]["cond"].keys()
    cond = {k: [b["cond"][k] for b in batch] for k in cond_keys}
    # scalars that are the same across batch: unwrap
    for k in ("video_length", "fps"):
        if k in cond and all(c == cond[k][0] for c in cond[k]):
            cond[k] = cond[k][0]
    return {"first_frame": first_frame, "prompt": prompts, "cond": cond}
