"""Unit test for build_cond + CondStage dataclass.

Validates that build_cond assembles the DynamiCrafter cond dict with:
  - c_crossattn: text_emb concatenated with projected image_emb (dim=1)
  - c_concat:    first-frame latent repeated along time axis
  - fs:          long tensor on correct device

Uses mock encoders so we don't need the 9.7GB DynamiCrafter checkpoint.
"""
import torch
import torch.nn as nn

from src.dynamicrafter_wrapper import CondStage, build_cond


class _MockTextEncoder(nn.Module):
    """CLIP text encoder: list[str] → [B, 77, 1024]."""
    def encode(self, prompts):
        B = len(prompts)
        # deterministic: hash prompt to a scalar then broadcast
        return torch.stack([
            torch.full((77, 1024), float(abs(hash(p)) % 97) / 100.0) for p in prompts
        ])


class _MockImageEmbedder(nn.Module):
    """OpenCLIP image embedder: [B,3,H,W] → [B,L,1280]."""
    def __init__(self, L=257):
        super().__init__()
        self.L = L
    def forward(self, img):
        B = img.shape[0]
        return torch.randn(B, self.L, 1280, dtype=img.dtype, device=img.device)


class _MockImageProj(nn.Module):
    """Resampler [B,L,1280] → [B,L',1024]."""
    def __init__(self, L_out=16):
        super().__init__()
        self.L_out = L_out
        self.proj = nn.Linear(1280, 1024)
    def forward(self, x):
        # Resample along length by averaging over blocks, then project
        B, L, _ = x.shape
        step = L // self.L_out
        pooled = x[:, :self.L_out * step, :].view(B, self.L_out, step, 1280).mean(dim=2)
        return self.proj(pooled)


def test_build_cond_shapes():
    text = _MockTextEncoder()
    emb  = _MockImageEmbedder(L=257)
    proj = _MockImageProj(L_out=16)
    cs = CondStage(cond_stage_model=text, embedder=emb, image_proj_model=proj,
                   temporal_length=16, default_fs=8)

    B, C, Hp, Wp = 2, 4, 40, 64
    first_frame_lat = torch.randn(B, C, Hp, Wp)
    first_frame_px  = torch.randn(B, 3, 320, 512)

    cond = build_cond(cs, prompts=["a cat", "a dog"],
                      first_frame_latent=first_frame_lat,
                      first_frame_px=first_frame_px, T=16)

    # c_crossattn: cat(text, img_proj) → [B, 77+16, 1024]
    assert "c_crossattn" in cond and len(cond["c_crossattn"]) == 1
    xa = cond["c_crossattn"][0]
    assert xa.shape == (B, 77 + 16, 1024), f"c_crossattn shape {xa.shape}"

    # c_concat: first-frame latent repeated along T
    assert "c_concat" in cond and len(cond["c_concat"]) == 1
    cc = cond["c_concat"][0]
    assert cc.shape == (B, C, 16, Hp, Wp), f"c_concat shape {cc.shape}"
    # verify the T-repeat is deterministic (same frame copied)
    assert torch.allclose(cc[:, :, 0, :, :], cc[:, :, 15, :, :])
    assert torch.allclose(cc[:, :, 0, :, :], first_frame_lat)

    # fs: default long tensor
    assert "fs" in cond
    assert cond["fs"].dtype == torch.long
    assert cond["fs"].shape == (B,)
    assert (cond["fs"] == 8).all()

    print(f"[ok] build_cond:  c_crossattn={tuple(xa.shape)}, "
          f"c_concat={tuple(cc.shape)}, fs={cond['fs'].tolist()}")


def test_build_cond_custom_fs():
    text = _MockTextEncoder()
    emb  = _MockImageEmbedder(L=257)
    proj = _MockImageProj(L_out=16)
    cs = CondStage(cond_stage_model=text, embedder=emb, image_proj_model=proj,
                   temporal_length=16, default_fs=8)

    first_frame_lat = torch.randn(1, 4, 40, 64)
    first_frame_px  = torch.randn(1, 3, 320, 512)
    fs = torch.tensor([24], dtype=torch.long)

    cond = build_cond(cs, prompts=["a bird"],
                      first_frame_latent=first_frame_lat,
                      first_frame_px=first_frame_px, fs=fs)
    assert cond["fs"] is fs
    print(f"[ok] build_cond custom fs: fs={cond['fs'].tolist()}")


def test_build_cond_end_to_end_with_call_unet():
    """Smoke test: build_cond's output is accepted by _call_unet without KeyError."""
    from src.opsd_trainer_i2v import _call_unet

    class _MockUNet(nn.Module):
        def forward(self, x, t, context=None, features_adapter=None, fs=None, **kwargs):
            # Standard DynamiCrafter sig; must receive context & fs
            assert context is not None, "context was lost — cond unpack failed"
            assert fs is not None, "fs was lost"
            return torch.zeros_like(x[:, :4])  # drop the concat'd channels

    text = _MockTextEncoder()
    emb  = _MockImageEmbedder(L=257)
    proj = _MockImageProj(L_out=16)
    cs = CondStage(cond_stage_model=text, embedder=emb, image_proj_model=proj,
                   temporal_length=16, default_fs=8)

    B, C, T, Hp, Wp = 1, 4, 16, 40, 64
    z = torch.randn(B, C, T, Hp, Wp)
    first_frame_lat = torch.randn(B, C, Hp, Wp)
    first_frame_px  = torch.randn(B, 3, 320, 512)
    cond = build_cond(cs, prompts=["sun rays"],
                      first_frame_latent=first_frame_lat,
                      first_frame_px=first_frame_px)

    unet = _MockUNet()
    out = _call_unet(unet, z, torch.tensor([0.5]), cond)
    print(f"[ok] build_cond → _call_unet end-to-end: out shape {tuple(out.shape)}")


if __name__ == "__main__":
    test_build_cond_shapes()
    test_build_cond_custom_fs()
    test_build_cond_end_to_end_with_call_unet()
    print("[all pass]")
