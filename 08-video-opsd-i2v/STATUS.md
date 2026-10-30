# 08 — OPSD-I2V · 项目状态与接续指南

> 最后更新：2026-10-30 10:20 — H20 到期前紧急备份

---

## ⚠️ 下次开机第一件事

```bash
cd ~/ml-portfolio/08-video-opsd-i2v
git pull
source .venv/bin/activate  # 如果 venv 还在；否则重建
export PYTHONPATH="$PWD"
```

---

## 一、已有数据盘点

### 训练数据（已生成，在 git 中）
| 数据 | 数量 | 路径 | 说明 |
|------|------|------|------|
| 训练 prompts | 160 | `data/prompts/tier2_train_200.json` | 15 个场景类别，GPT 生成 |
| 评估 prompts | 40 | `data/prompts/tier2_eval_40.json` | held-out，不参与训练 |
| 训练首帧 | 200 张 | `data/first_frames_200/*.png` | SDXL-Turbo 512×320 |
| 评估首帧 | 40 张 | `data/first_frames_eval/*.png` | SDXL-Turbo 512×320 |
| 旧 prompts（小规模） | 6 / 894 | `data/prompts/tier2_train.json` / `train.json` | Tier-1 时期用的 |

### LoRA Checkpoints（已训练，在 git 中）
| 实验 | 步数 | 路径 | 大小 |
|------|------|------|------|
| V1 (nat_clamp) | 500 | `outputs/opsd_tier2_cvpr/lora_step00500/` | 13MB |
| V2 (nat_sigmoid) | 500 | `outputs/opsd_tier2_cvpr_v2/lora_step00500/` | 13MB |
| NoNat (消融) | 500 | `outputs/opsd_tier2_cvpr_no_nat/lora_step00500/` | 13MB |

### 训练 Metrics（在 git 中）
- 每组 500 行 JSON (per-step)：`outputs/opsd_tier2_cvpr{,_v2,_no_nat}/train_metrics.jsonl`
- 字段：step, reward, loss, R_subject_consistency, R_dynamic_degree, R_motion_smoothness, R_naturalness_penalty, grad_norm, lr

### 训练曲线图（在 git 中）
- 单组：`outputs/opsd_tier2_cvpr{,_v2,_no_nat}/figures/fig_reward_curves.png` 等（各 3 张）
- 三组对比：`outputs/figures_comparison/comparison_rewards.png`, `naturalness_evolution.png`, `reward_vs_gpu_hours.png`

### Eval 视频（在本地，mp4 被 gitignore）
| 配置 | 已渲染 | 路径 |
|------|--------|------|
| Base（无 LoRA） | **40/40** ✅ | `outputs/eval_cvpr/videos_base/seed_00/` |
| V1 (nat_clamp) | **10/40** ❌ | `outputs/eval_cvpr/videos_v1_nat_clamp/seed_00/` |
| V2 (nat_sigmoid) | **10/40** ❌ | `outputs/eval_cvpr/videos_v2_nat_sigmoid/seed_00/` |
| NoNat (消融) | **10/40** ❌ | `outputs/eval_cvpr/videos_ablation_no_nat/seed_00/` |

> ⚠️ mp4 视频在 `.gitignore` 中，**服务器到期后丢失**。下次需要重新渲染。

### GIF 对比（在 git 中，但只有 6 个老场景）
- 32 个 GIF 在 `docs/figures/duel/`：duel × 6、fourway × 6、lightness × 6、tier2_lora × 6、tier2_bon × 6、triway × 2
- **全部基于旧的 6 个 prompt**（sun_rays, candle_dark, houses_hillside, fog_pine_dawn, pagoda_snowy, waterfall_zoom）
- **需要从 40 eval 新场景生成新 GIF**

### Eval Metrics（在 git 中）
- `outputs/eval_cvpr/metrics_summary.json` — 4 组 × 3 指标（仅基于 n=10）
- `outputs/eval_cvpr/metrics_table.md`

---

## 二、关键实验结论

### 三组消融训练结果
| 实验 | 最终 Reward | R_naturalness | 发现 |
|------|------------|---------------|------|
| V1 (ReLU+clamp) | +2.16 | **0.000** | 梯度 100% 死亡，naturalness 约束无效 |
| **V2 (sigmoid)** | -0.50 | **-0.786** | ✅ 梯度活的，naturalness 从 -1.47→-0.79 |
| NoNat (消融) | +2.36 | N/A | 纯 reward hacking baseline |

### 已解决的 8 个 DynamiCrafter 适配 Bug
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

---

## 三、下次要做的事（按顺序）

### Step 1：重新渲染 eval 视频（~30 min GPU）

V1/V2/NoNat 各只渲了 10/40，需要补齐剩下 30 个。

```bash
# 首先重建 seed_00 子目录（渲染器需要）
mkdir -p data/first_frames_eval/seed_00
cp data/first_frames_eval/*.png data/first_frames_eval/seed_00/

# 渲染剩余视频（每组 ~8 min）
for tag_lora in \
  "videos_v1_nat_clamp outputs/opsd_tier2_cvpr/lora_step00500" \
  "videos_v2_nat_sigmoid outputs/opsd_tier2_cvpr_v2/lora_step00500" \
  "videos_ablation_no_nat outputs/opsd_tier2_cvpr_no_nat/lora_step00500"; do
  tag=$(echo $tag_lora | cut -d' ' -f1)
  lora=$(echo $tag_lora | cut -d' ' -f2)
  python -m src.render_i2v \
    --frames_dir data/first_frames_eval \
    --prompts data/prompts/tier2_eval_40.json \
    --out_dir outputs/eval_cvpr/$tag \
    --seeds 0 --num_frames 16 --fps 8 --limit 40 \
    --i2v_lora_dir $lora
done
```

### Step 2：生成 GIF 对比（新的 40 eval 场景）

当前只有 6 个旧场景的 GIF，需要从 40 eval 视频生成 Base vs V2 对比。

```bash
# 需要写一个 mp4→GIF 并排对比脚本，大致逻辑：
# for each of 40 eval prompts:
#   left = outputs/eval_cvpr/videos_base/seed_00/{idx:03d}.mp4
#   right = outputs/eval_cvpr/videos_v2_nat_sigmoid/seed_00/{idx:03d}.mp4
#   → docs/figures/eval_duel/duel_{idx:03d}.gif (side-by-side, 256px wide each)
```

### Step 3：增加训练步数到 2000+

500 步 LoRA L2 norm 仅 ~27，eval metrics 无显著差异。需要至少 2000 步：

```bash
# 修改 configs/tier2_cvpr_v2.yaml: max_train_steps: 500 → 2000
# 从现有 checkpoint 续训（如果支持）或重新训练
python -m src.train_entry --method opsd_i2v --config configs/tier2_cvpr_v2.yaml
```

### Step 4：用 n=40 重新计算 eval metrics

```bash
# 当 40 个视频全部渲染完成后
python -c "
import json, numpy as np
from pathlib import Path
from src.eval.naturalness_v2 import naturalness_v2_video
import imageio.v3 as iio

configs = {
    'base': 'outputs/eval_cvpr/videos_base/seed_00',
    'v1_nat_clamp': 'outputs/eval_cvpr/videos_v1_nat_clamp/seed_00',
    'v2_nat_sigmoid': 'outputs/eval_cvpr/videos_v2_nat_sigmoid/seed_00',
    'ablation_no_nat': 'outputs/eval_cvpr/videos_ablation_no_nat/seed_00',
}
# ... (eval 代码同之前，改 n=40)
"
```

### Step 5：更新 README

- 用新的 40 场景 GIF 替换旧的 6 场景
- 更新 eval metrics 表格
- 如果 2000 步训练完成，更新训练曲线

### Step 6（可选）：调 reward 参数
- `dynamic_degree` 的 `cap_tau=0.6` 太低（50 步就饱和），改 0.85+
- `motion_smoothness` 在高噪声下 RAFT 不可靠，加 `sigma_threshold`
- 考虑加 FVD/FID 标准视频 metrics

---

## 四、环境依赖

```
# H20 GPU 环境
- NVIDIA H20 80GB
- Python 3.10 + .venv
- PyTorch 2.x + CUDA
- DynamiCrafter（在 third_party/，被 gitignore）
- RAFT（在 third_party/，被 gitignore）
- 基座模型权重在 weights/（被 gitignore，~4GB）

# 需要重新下载的：
pip install -r requirements.txt  # 如果 venv 丢失
# DynamiCrafter 和 RAFT 需要重新 clone
# 基座模型权重需要重新下载（见 README "Setup" 部分）
```

---

## 五、文件结构

```
08-video-opsd-i2v/
├── README.md                               ← GitHub 首页（GIF + 训练曲线 + metrics）
├── STATUS.md                               ← 本文件（进度 + 下一步指南）
├── requirements.txt
├── configs/                                10 套 YAML 配置
│   ├── tier2_cvpr.yaml                     V1 (nat_clamp)
│   ├── tier2_cvpr_v2.yaml                  V2 (nat_sigmoid) ← 主力
│   └── tier2_cvpr_ablation_no_nat.yaml     消融
├── src/
│   ├── opsd_trainer_i2v.py                 训练循环（OPSD RL）
│   ├── render_i2v.py                       推理渲染
│   ├── rewards/                            4 个 reward 函数
│   ├── postprocess/                        lightness 后处理
│   └── eval/                               naturalness_v2 评估
├── scripts/
│   ├── generate_prompts_200.py             生成 200 prompts
│   ├── generate_first_frames.py            SDXL-Turbo 生成首帧
│   ├── plot_training_curves.py             单组训练曲线
│   ├── plot_comparison.py                  三组对比曲线
│   └── render_eval_all.sh                  批量渲染脚本
├── data/
│   ├── prompts/                            5 个 JSON prompt 文件
│   ├── first_frames_200/                   200 张训练首帧 (35MB)
│   └── first_frames_eval/                  40 张 eval 首帧 (8.7MB)
├── outputs/
│   ├── opsd_tier2_cvpr/                    V1: metrics + figures + LoRA
│   ├── opsd_tier2_cvpr_v2/                 V2: metrics + figures + LoRA
│   ├── opsd_tier2_cvpr_no_nat/             NoNat: metrics + figures + LoRA
│   ├── figures_comparison/                 三组对比图 (3 张 PNG)
│   └── eval_cvpr/                          eval metrics + 视频(gitignored)
└── docs/figures/duel/                      32 个 GIF (161MB, 仅 6 场景)
```
