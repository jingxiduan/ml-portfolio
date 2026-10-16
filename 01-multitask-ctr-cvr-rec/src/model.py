"""
RankMixer-style dense feature interaction + multi-task (CTR/CVR) tower model

Design:
  - Shared bottom user/ad/context representation -> each sparse feature has its own
    embedding, dense features are projected into tokens of the same dimension,
    replacing traditional FM/DCN explicit second-order crossing.
  - RankMixer dense feature interaction, shifting from sparse-lookup-dominant to
    Transformer-style dense-computation-dominant -> borrowing from MLP-Mixer, token-mixing
    across the token dimension (cross-feature) and channel-mixing across the embedding
    dimension (cross-channel), stacked into a dense-computation-dominant interaction network.
  - Multi-task structure with task-branch routing on top + sample mask -> the shared
    representation feeds into two independent towers, CTR and CVR; the CVR branch's loss
    backpropagates gradients only on clicked (click=1) samples via the label_convert == -1 mask.
"""
import torch
import torch.nn as nn


class MixerBlock(nn.Module):
    """token-mixing + channel-mixing, an MLP-Mixer-like dense interaction module."""

    def __init__(self, num_tokens: int, dim: int, token_hidden: int, channel_hidden: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.token_mix = nn.Sequential(
            nn.Linear(num_tokens, token_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_hidden, num_tokens),
        )
        self.norm2 = nn.LayerNorm(dim)
        self.channel_mix = nn.Sequential(
            nn.Linear(dim, channel_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channel_hidden, dim),
        )

    def forward(self, x):
        # x: (B, T, D)
        y = self.norm1(x)
        y = y.transpose(1, 2)              # (B, D, T)
        y = self.token_mix(y)              # interaction across feature tokens
        y = y.transpose(1, 2)              # (B, T, D)
        x = x + y

        y = self.norm2(x)
        y = self.channel_mix(y)            # interaction across embedding channels
        x = x + y
        return x


class Tower(nn.Module):
    def __init__(self, in_dim: int, hidden_dims=(128, 64)):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(0.1)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class RankMixerMTL(nn.Module):
    def __init__(self, vocab_size: dict, sparse_cols, dense_dim: int,
                 embed_dim: int = 16, num_mixer_blocks: int = 2, dropout: float = 0.1):
        super().__init__()
        self.sparse_cols = sparse_cols
        self.embed_dim = embed_dim

        self.embeddings = nn.ModuleDict({
            col: nn.Embedding(vocab_size[col], embed_dim) for col in sparse_cols
        })
        self.dense_proj = nn.Linear(dense_dim, embed_dim)

        num_tokens = len(sparse_cols) + 1  # +1 for the dense feature token
        self.blocks = nn.ModuleList([
            MixerBlock(num_tokens, embed_dim, token_hidden=num_tokens * 2,
                       channel_hidden=embed_dim * 4, dropout=dropout)
            for _ in range(num_mixer_blocks)
        ])
        self.final_norm = nn.LayerNorm(embed_dim)

        shared_dim = num_tokens * embed_dim
        self.ctr_tower = Tower(shared_dim)
        self.cvr_tower = Tower(shared_dim)

    def forward(self, sparse_x, dense_x):
        tokens = [self.embeddings[col](sparse_x[:, i]) for i, col in enumerate(self.sparse_cols)]
        tokens.append(self.dense_proj(dense_x))
        x = torch.stack(tokens, dim=1)  # (B, T, D)

        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)

        shared_repr = x.reshape(x.size(0), -1)
        ctr_logit = self.ctr_tower(shared_repr)
        cvr_logit = self.cvr_tower(shared_repr)
        return ctr_logit, cvr_logit
