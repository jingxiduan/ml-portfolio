"""
Differentiable multimodal reward model: CLIP-based text-image alignment score
(the "CLIPScore" reward used as one of the 7 reward heads in the original
DiffusionOPSD paper, swapped in here instead of the 7B/8B HPSv3 / DeQA scorers
because those don't fit in 36GB of unified memory on a Mac).

Unlike `transformers.CLIPProcessor` (which operates on PIL images and breaks
the autograd graph), this wraps CLIP's image tower with a differentiable
resize + normalize so gradients can flow from the reward back into the
generated image tensor -- this is exactly the "differentiable reward
gradient" OPSD uses to build its positive/negative targets.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTokenizer

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
CLIP_IMAGE_SIZE = 224
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


class ClipRewardModel:
    """Differentiable text-image CLIPScore reward, frozen, eval-mode."""

    def __init__(self, device: str = "mps", dtype: torch.dtype = torch.float32):
        self.device = device
        self.dtype = dtype
        self.model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device=device, dtype=dtype)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.tokenizer = CLIPTokenizer.from_pretrained(CLIP_MODEL_NAME)
        self.mean = CLIP_MEAN.to(device=device, dtype=dtype)
        self.std = CLIP_STD.to(device=device, dtype=dtype)

    @torch.no_grad()
    def encode_text(self, prompts: list[str]) -> torch.Tensor:
        tokens = self.tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        feats = self.model.get_text_features(**tokens)
        return F.normalize(feats, dim=-1)

    def _preprocess(self, images: torch.Tensor) -> torch.Tensor:
        """images: (B, 3, H, W) in [0, 1], differentiable.

        Uses plain bilinear resize (no antialias): MPS does not yet implement
        the backward pass for antialiased bicubic/bilinear interpolation, and
        a differentiable resize is required here since gradients must flow
        from the reward back into the generated image/latent.
        """
        images = F.interpolate(images, size=(CLIP_IMAGE_SIZE, CLIP_IMAGE_SIZE),
                                mode="bilinear", align_corners=False)
        images = (images - self.mean) / self.std
        return images

    def score(self, images: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """Returns a differentiable per-sample CLIPScore (cosine sim * 100), shape (B,)."""
        pixel_values = self._preprocess(images.to(dtype=self.dtype))
        image_feats = self.model.get_image_features(pixel_values=pixel_values)
        image_feats = F.normalize(image_feats, dim=-1)
        cos_sim = (image_feats * text_features).sum(dim=-1)
        return cos_sim * 100.0
