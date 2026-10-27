"""Subject-consistency reward (CLIP-based, differentiable).

VBench defines subject consistency as the average cosine similarity of CLIP image features
between the first frame and each subsequent frame. We re-implement it in a way that admits
gradients w.r.t. the frames (OpenCLIP's forward is differentiable end-to-end).
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import BaseReward, register

try:
    import open_clip  # type: ignore
except ImportError as e:  # pragma: no cover
    raise ImportError("open_clip_torch required (`pip install open_clip_torch`)") from e


@register("subject_consistency")
class SubjectConsistency(BaseReward):
    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        # we re-implement the preprocess inline (differentiable); keep tokenizer only.
        self.model = model.visual
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.to(device)
        self._device = torch.device(device)

        # OpenAI CLIP mean/std
        self.register_buffer(
            "mean", torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
        )

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        # x in [-1, 1]  ->  [0, 1]  ->  CLIP-normalized
        x = (x + 1.0) * 0.5
        x = F.interpolate(x, size=224, mode="bilinear", align_corners=False, antialias=True)
        return (x - self.mean) / self.std

    def forward(self, video: torch.Tensor, prompts: List[str]) -> torch.Tensor:
        """video: [B, T, 3, H, W] -> [B] mean cosine similarity to frame 0."""
        B, T = video.shape[0], video.shape[1]
        x = video.flatten(0, 1)                              # [B*T, 3, H, W]
        x = self._normalize(x)
        feats = self.model(x)                                # [B*T, D]
        feats = F.normalize(feats, dim=-1).view(B, T, -1)
        anchor = feats[:, :1]                                # [B, 1, D]
        sims = (feats * anchor).sum(-1)                      # [B, T] cos sim
        return sims[:, 1:].mean(dim=1)                       # mean over t=1..T-1
