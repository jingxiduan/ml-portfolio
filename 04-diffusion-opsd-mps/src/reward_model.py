"""Differentiable training reward for image OPSD target construction."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTokenizer

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
CLIP_IMAGE_SIZE = 224
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
QUALITY_POSITIVE = "a professional high-end photograph, natural skin texture, realistic lighting, sharp fine detail, balanced color and exposure"
QUALITY_NEGATIVE = "a low-quality artificial image, blurry details, oversaturated colors, crushed shadows, plastic skin, bad exposure"


class ClipRewardModel:
    """Frozen CLIP alignment with a small differentiable display-range penalty."""

    def __init__(self, device: str = "cuda", dtype: torch.dtype = torch.float32):
        self.device = torch.device(device)
        self.dtype = dtype
        self.model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(self.device, dtype=dtype)
        self.model.eval().requires_grad_(False)
        self.tokenizer = CLIPTokenizer.from_pretrained(CLIP_MODEL_NAME)
        self.mean = CLIP_MEAN.to(self.device, dtype=dtype)
        self.std = CLIP_STD.to(self.device, dtype=dtype)

    @torch.no_grad()
    def encode_text(self, prompts: list[str]) -> torch.Tensor:
        tokens = self.tokenizer(
            prompts, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)
        return F.normalize(self.model.get_text_features(**tokens), dim=-1)

    @torch.no_grad()
    def encode_quality_targets(self) -> tuple[torch.Tensor, torch.Tensor]:
        positive, negative = self.encode_text([QUALITY_POSITIVE, QUALITY_NEGATIVE])
        return positive.unsqueeze(0), negative.unsqueeze(0)

    def _preprocess(self, images: torch.Tensor) -> torch.Tensor:
        images = F.interpolate(
            images.to(dtype=self.dtype),
            size=(CLIP_IMAGE_SIZE, CLIP_IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        )
        return (images - self.mean) / self.std

    def score(
        self,
        images: torch.Tensor,
        text_features: torch.Tensor,
        quality_positive: torch.Tensor,
        quality_negative: torch.Tensor,
        quality_weight: float = 0.35,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        pixel_values = self._preprocess(images)
        image_features = F.normalize(
            self.model.get_image_features(pixel_values=pixel_values), dim=-1
        )
        alignment = (image_features * text_features).sum(dim=-1) * 100.0
        quality_positive_score = (image_features * quality_positive).sum(dim=-1) * 100.0
        quality_negative_score = (image_features * quality_negative).sum(dim=-1) * 100.0
        quality_margin = quality_positive_score - quality_negative_score

        luma = 0.2126 * images[:, 0] + 0.7152 * images[:, 1] + 0.0722 * images[:, 2]
        clipped_fraction = (
            F.relu(0.02 - luma).mean(dim=(1, 2))
            + F.relu(luma - 0.98).mean(dim=(1, 2))
        )
        saturation = images.amax(dim=1) - images.amin(dim=1)
        saturation_excess = F.relu(saturation - 0.95).mean(dim=(1, 2))
        display_penalty = clipped_fraction + 0.25 * saturation_excess
        reward = alignment + quality_weight * quality_margin - 10.0 * display_penalty
        return reward, {
            "alignment": alignment,
            "quality_margin": quality_margin,
            "display_penalty": display_penalty,
        }
