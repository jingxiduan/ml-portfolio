"""LAION aesthetic predictor (differentiable).

Thin MLP on top of CLIP image embeddings. Weights downloaded by scripts/02_download_weights.sh
to ``weights/aesthetic_predictor_v2.pth``.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import BaseReward, register

try:
    import open_clip  # type: ignore
except ImportError as e:  # pragma: no cover
    raise ImportError("open_clip_torch required") from e


class _AestheticMLP(nn.Module):
    def __init__(self, d_in: int = 768) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(d_in, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x).squeeze(-1)


@register("aesthetic")
class Aesthetic(BaseReward):
    def __init__(
        self,
        weights_path: str = "weights/aesthetic_predictor_v2.pth",
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        model, _, _ = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
        self.clip = model.visual.eval()
        for p in self.clip.parameters():
            p.requires_grad_(False)

        self.mlp = _AestheticMLP(d_in=768).eval()
        wp = Path(weights_path)
        if wp.is_file():
            state = torch.load(wp, map_location="cpu")
            self.mlp.load_state_dict(state, strict=False)
        else:
            print(f"[Aesthetic] WARNING: weights not found at {wp}; MLP is random. "
                  "Run scripts/02_download_weights.sh.")
        for p in self.mlp.parameters():
            p.requires_grad_(False)

        self.register_buffer(
            "mean", torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
        )
        self.to(device)

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = (x + 1.0) * 0.5
        x = F.interpolate(x, size=224, mode="bilinear", align_corners=False, antialias=True)
        return (x - self.mean) / self.std

    def forward(self, video: torch.Tensor, prompts: List[str]) -> torch.Tensor:
        B, T = video.shape[0], video.shape[1]
        x = self._preprocess(video.flatten(0, 1))
        feats = self.clip(x)                                  # [B*T, 768]
        feats = F.normalize(feats, dim=-1)
        scores = self.mlp(feats).view(B, T).mean(dim=1)       # [B]
        # Aesthetic predictor outputs ~ [1, 10]; map to [0, 1]:
        return (scores - 1.0) / 9.0
