"""
Differentiable multimodal reward model for video OPSD.

Extends the CLIPScore reward used in `04-diffusion-opsd-mps` (a single-image
text-image alignment score) to video by averaging it over sampled frames, and
adds a *temporal consistency* term: the mean cosine similarity between
consecutive frames' CLIP image embeddings. Neither the per-frame averaging
nor the temporal-consistency term is from the original DiffusionOPSD paper --
it is a reasonable, small extension needed because the paper's reward heads
(HPSv3 / DeQA / CLIPScore) only ever score a single still image. Without some
notion of "do consecutive frames look like the same continuous scene", OPSD's
reward-gradient step would only ever push individual frames to look more
prompt-aligned in isolation, with no pressure towards temporally coherent
(non-flickering) motion.

Both terms stay strictly differentiable end-to-end (image tensor -> CLIP ->
scalar reward), which is what OPSD's positive/negative target construction
needs (it takes the gradient of the reward w.r.t. the generated latent).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTokenizer

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
CLIP_IMAGE_SIZE = 224
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


class ClipVideoRewardModel:
    """Differentiable text-video reward: frame-averaged CLIPScore + temporal consistency."""

    def __init__(self, device: str = "mps", dtype: torch.dtype = torch.float32,
                 temporal_weight: float = 0.3):
        self.device = device
        self.dtype = dtype
        self.temporal_weight = temporal_weight
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
        """images: (N, 3, H, W) in [0, 1], differentiable. Plain bilinear resize
        (no antialias): MPS has no backward pass for antialiased resize."""
        images = F.interpolate(images, size=(CLIP_IMAGE_SIZE, CLIP_IMAGE_SIZE),
                                mode="bilinear", align_corners=False)
        images = (images - self.mean) / self.std
        return images

    def score_video(self, frames: torch.Tensor, text_features: torch.Tensor):
        """frames: (B, n_frames, 3, H, W) in [0, 1], differentiable.
        text_features: (B, dim), L2-normalized.

        Returns (reward, align_score, consistency_score), each shape (B,):
          reward        = align_score + temporal_weight * consistency_score
          align_score   = mean over frames of CLIPScore(frame, prompt) * 100
          consistency   = mean cosine similarity between consecutive frames'
                          CLIP image embeddings * 100 (100 = identical embedding)
        """
        batch, n_frames, c, h, w = frames.shape
        flat = frames.reshape(batch * n_frames, c, h, w).to(dtype=self.dtype)
        pixel_values = self._preprocess(flat)
        image_feats = self.model.get_image_features(pixel_values=pixel_values)
        image_feats = F.normalize(image_feats, dim=-1)
        image_feats = image_feats.reshape(batch, n_frames, -1)

        text_feats_exp = text_features.unsqueeze(1)
        align_score = (image_feats * text_feats_exp).sum(dim=-1).mean(dim=1) * 100.0

        if n_frames > 1:
            consistency = (image_feats[:, 1:] * image_feats[:, :-1]).sum(dim=-1).mean(dim=1) * 100.0
        else:
            consistency = torch.zeros(batch, device=frames.device, dtype=align_score.dtype)

        reward = align_score + self.temporal_weight * consistency
        return reward, align_score, consistency
