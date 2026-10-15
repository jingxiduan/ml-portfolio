"""
RankMixer 风格的稠密特征交互 + 多任务(CTR/CVR)塔式模型

设计对应 resume 描述:
  - "底层共享的用户/广告/上下文表征" -> 各稀疏特征独立 Embedding + 稠密特征投影, 统一为
    一组等维 token, 替代传统 FM/DCN 的显式二阶交叉。
  - "RankMixer 稠密特征交互, 由稀疏查表主导改为 Transformer 式稠密计算主导" -> 借鉴
    MLP-Mixer 思路, 对 token 维度(跨特征)做 token-mixing, 对 embedding 维度(跨通道)做
    channel-mixing, 多层堆叠形成稠密计算为主的交互网络。
  - "上层按任务分支路由的多任务结构 + sample mask" -> 共享表征之后接 CTR / CVR 两个独立
    塔, CVR 分支的损失通过 label_convert == -1 的 mask 只在点击(click=1)样本上回传梯度。
"""
import torch
import torch.nn as nn


class MixerBlock(nn.Module):
    """token-mixing + channel-mixing, 类 MLP-Mixer 的稠密交互模块。"""

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
        y = self.token_mix(y)              # 跨特征 token 做交互
        y = y.transpose(1, 2)              # (B, T, D)
        x = x + y

        y = self.norm2(x)
        y = self.channel_mix(y)            # 跨 embedding 通道做交互
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

        num_tokens = len(sparse_cols) + 1  # + 1 个稠密特征 token
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
