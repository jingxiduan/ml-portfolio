# 08 — OPSD-I2V Video Generation · 项目状态

> 最后更新：2026-10-30 H20 session 到期前

## 已完成

### 1. Tier-1: 诊断 + Quick Fix（Mac 上跑的）
- 诊断了 pre-trained LoRA 的 reward hacking（2× 饱和度、1.7× 过曝）
- LoRA rescale sweep → scale=0.2 sweet spot
- Best-of-N seed selection（naturalness_v2 scorer）
- Lightness post-process: shadow lift + highlight protect + sat pullback
- **结果**: Wilcoxon p=0.016, appeal 从 0.578 → 0.680
- 6 组 GIF 对比（duel + fourway + lightness）: `docs/figures/duel/*.gif`（32个 GIF）

### 2. Tier-2: 三组消融训练（H20 GPU，各 500 步）

| 实验 | 配置 | 步数 | 耗时 | 最终 Reward | Naturalness |
|------|------|------|------|-------------|-------------|
| V1 (nat_clamp) | `tier2_cvpr.yaml` | 500 | 86min | +2.16 | 0.000 (梯度死亡) |
| V2 (nat_sigmoid) | `tier2_cvpr_v2.yaml` | 500 | 86min | -0.50 | -1.23 (有梯度) |
| NoNat (消融) | `tier2_cvpr_ablation_no_nat.yaml` | 500 | 86min | +2.36 | N/A |

**关键发现**: V1 的 ReLU+clamp 在 100% 的训练步中杀死了 naturalness 梯度。V2 用 sigmoid 修复后，naturalness penalty 从 -1.47 改善到 -0.79。

### 3. 评估（n=10 held-out prompts × 4 configs）

| Config | Appeal↑ | Naturalness↑ | Dynamic↑ |
|--------|---------|--------------|----------|
| Base (no LoRA) | 0.7295 | 0.9515 | 0.5075 |
| V1 (nat_clamp) | 0.7294 | 0.9515 | 0.5074 |
| V2 (nat_sigmoid) | 0.7291 | 0.9513 | 0.5069 |
| Ablation (no_nat) | 0.7296 | 0.9512 | 0.5079 |

> 500步训练 LoRA L2 norm 仅 ~27，不足以在 eval metrics 上看到显著变化。价值在训练曲线的动态差异。

### 4. 训练曲线 + 对比图
- 每组实验: reward curves / loss curves / diagnostics（PNG + PDF）
- 三组对比图: `outputs/figures_comparison/comparison_rewards.png`, `naturalness_evolution.png`, `reward_vs_gpu_hours.png`

### 5. 数据准备
- 200 diverse prompts（15个类别，160 train + 40 eval）
- 200 SDXL-Turbo 首帧（512×320）
- 40 held-out eval 首帧

## 产出文件清单

```
outputs/
├── opsd_tier2_cvpr/                 # V1: metrics.jsonl + figures + lora_step00500 (15MB)
├── opsd_tier2_cvpr_v2/              # V2: metrics.jsonl + figures + lora_step00500 (15MB)
├── opsd_tier2_cvpr_no_nat/          # NoNat: metrics.jsonl + figures + lora_step00500 (15MB)
├── figures_comparison/              # 三组对比图 (1.4MB)
├── eval_cvpr/                       # metrics_summary.json + metrics_table.md + 40 videos (49MB)
data/
├── first_frames_200/                # 200张训练首帧 (35MB)
├── first_frames_eval/               # 40张eval首帧 (8.7MB)
├── prompts/                         # JSON prompt文件
docs/figures/duel/                   # 32个 GIF 对比 (161MB)
```

## 尚未完成 / 下次要做

### 优先级高
1. **增加训练步数到 2000+**: 当前 500 步 LoRA 变化太小，需要至少 2000 步才能看到 eval metrics 变化
2. **增加 eval 样本**: 当前 n=10，需要 n=40+（已有 40 个 prompt，只渲染了 10 个）
3. **渲染更多 eval 视频生成 GIF 对比**: 当前只有 6 组 Tier-1 旧 GIF，需要用 Tier-2 训练的 LoRA 生成新对比

### 优先级中
4. **调整 dynamic_degree reward 阈值**: cap_tau=0.6 太低，所有 config 在 50 步内就饱和到 1.0
5. **motion_smoothness reward 在高噪声下无效**: RAFT 光流在噪声重建的 latent 上不可靠，需要只在低噪声 σ_q 时启用
6. **FVD / FID 标准视频指标**: 当前只有 naturalness_v2 自己的 scorer（scorer = reward 循环依赖问题）
7. **Human evaluation**: A/B 人工比较

### 优先级低
8. **更大 LoRA rank**: 当前 rank=16, 可以试 rank=32/64
9. **DPO-style 训练**: 用 naturalness_v2 scorer 做 preference pair 再训
10. **不同 base model**: 当前只用 DynamiCrafter 512×320, 可以试其他 I2V model

## 8 个 Bug Fix 记录（DynamiCrafter 适配）

| # | Bug | Fix |
|---|-----|-----|
| 1 | conditioning dropped in UNetModel.forward | `_call_unet` helper |
| 2 | `vae.decode(z).sample` crashed | `_vae_decode_video` helper |
| 3 | `encode(x).latent_dist.sample()` crashed | `encode(x).sample()` |
| 4 | PEFT target modules regex mismatch | Runtime enumeration |
| 5 | Missing `c_crossattn` + `c_concat` | `CondStage` dataclass |
| 6 | OpenCLIP hard-coded CUDA | Patch device attribute |
| 7 | `IndexError` in `timestep_embedding` | Auto-broadcast |
| 8 | Shape mismatch in `rollout_until` | Use `c_concat.shape[2]` |
