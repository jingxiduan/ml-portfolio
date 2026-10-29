"""Unit test for the two helpers added to opsd_trainer_i2v.py.

Validates that `_call_unet` correctly unpacks a DynamiCrafter cond dict so
the raw UNetModel receives `context=` instead of silently dropping the
conditioning via `**kwargs`. Also sanity-checks `_vae_decode_video` shape
math on a mock VAE.
"""
import torch
import torch.nn as nn

from src.opsd_trainer_i2v import _call_unet, _vae_decode_video


class _MockUNet(nn.Module):
    """Records how it was called so we can assert that cond was unpacked."""
    def __init__(self):
        super().__init__()
        self.last_x_shape = None
        self.last_context_shape = None
        self.last_fs = None

    def forward(self, x, timesteps, context=None, features_adapter=None, fs=None, **kwargs):
        self.last_x_shape = tuple(x.shape)
        self.last_context_shape = None if context is None else tuple(context.shape)
        self.last_fs = fs
        # produce an output matching x (what DynamiCrafter UNet does)
        return torch.zeros_like(x)


class _MockVAE(nn.Module):
    """2-D VAE: 4-ch latent -> 3-ch RGB, 1x spatial for simplicity."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 3, 1)
    def decode(self, z):
        return self.conv(z)


def test_call_unet_unpacks_cond():
    unet = _MockUNet()
    # DynamiCrafter hybrid cond: c_crossattn (text+image emb) + c_concat (image latent)
    B, C, T, H, W = 1, 4, 16, 40, 64
    z = torch.randn(B, C, T, H, W)
    text_emb = torch.randn(B, 77, 1024)
    img_latent = torch.randn(B, C, T, H, W)
    fs = torch.tensor([8])
    cond = {
        "c_crossattn": [text_emb],
        "c_concat":    [img_latent],
        "fs":          fs,
    }
    t = torch.tensor([0.5])
    out = _call_unet(unet, z, t, cond)
    # c_concat must be cat'd to x on channel dim: 4+4 = 8 channels
    assert unet.last_x_shape == (B, 2 * C, T, H, W), f"got {unet.last_x_shape}"
    assert unet.last_context_shape == (B, 77, 1024), f"got {unet.last_context_shape}"
    assert unet.last_fs is fs, "fs not forwarded"
    assert out.shape == z.shape or out.shape == (B, 2 * C, T, H, W)  # mock returns zeros_like
    print(f"[ok] _call_unet unpacks cond:  x cat to {unet.last_x_shape[1]} ch, "
          f"context {unet.last_context_shape}, fs={unet.last_fs.tolist()}")


def test_call_unet_crossattn_only():
    """If c_concat is empty, x stays unchanged."""
    unet = _MockUNet()
    z = torch.randn(1, 4, 16, 40, 64)
    cond = {"c_crossattn": [torch.randn(1, 77, 1024)], "c_concat": []}
    _call_unet(unet, z, torch.tensor([0.5]), cond)
    assert unet.last_x_shape == (1, 4, 16, 40, 64)
    print("[ok] _call_unet crossattn-only: no concat, x stays 4-ch")


def test_vae_decode_video_shape():
    vae = _MockVAE()
    B, C, T, H, W = 2, 4, 16, 40, 64
    z = torch.randn(B, C, T, H, W)
    out = _vae_decode_video(vae, z, scale_factor=0.18215)
    assert out.shape == (B, T, 3, H, W), f"got {out.shape}"
    print(f"[ok] _vae_decode_video: {tuple(z.shape)} → {tuple(out.shape)}")


if __name__ == "__main__":
    test_call_unet_unpacks_cond()
    test_call_unet_crossattn_only()
    test_vae_decode_video_shape()
    print("[all pass]")
