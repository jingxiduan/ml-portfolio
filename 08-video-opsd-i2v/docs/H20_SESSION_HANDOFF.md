# H20 Session Handoff — 2026-10-30 00:00–10:00 window

> **目标读者：H20 容器里新启动的 CodeBuddy / 任何 AI 助手。**
> 这个文档是从 10/29 这几天的开发会话压缩出来的 context —— 读完它，你就
> 完全接手了之前的进度，不需要再问用户历史问题。

---

## 1 · 身份与目标

- 用户：**jingxiduan**，M3 Pro Mac 本机开发者，GitHub `jingxiduan/ml-portfolio`
- 本次会话目标：**在 H20 10 小时窗口内，训出一个真实的 Tier-2 OPSD-I2V LoRA**，产出 3-way 对比视频，commit 到 portfolio
- 预约窗口：**2026-10-30 00:00 — 10:00 (UTC+8)**，H20-单机1卡(长时)

## 2 · 你是谁、不是谁

- **你是** `08-video-opsd-i2v/` 项目的技术执行者：读 README + playbook + 这个文档，然后按 `H20_PLAYBOOK.md` 的三步命令跑起来
- **你不是** 项目设计者 —— 整条管线（OPSD trainer、cond-stage 集成、BoN、lightness、naturalness_v2）已经在前几次会话里设计好并通过了单测，不要重新设计
- **不要碰的事**：
  - 不要改 `src/opsd_trainer_i2v.py` 的训练循环逻辑（已被 6 个单测 pin 死）
  - 不要改 `src/dynamicrafter_wrapper.py` 的 `build_cond` / `CondStage`（已和 DynamiCrafter 源码对齐）
  - 不要改 reward 配置权重（`configs/tier2_natural.yaml` 是精心设计的）
  - **绝对不要**在任何 commit 消息里出现 "paper / submission / CVPR / under review" 字样（用户明确标记为敏感）

## 3 · 前情提要（压缩版）

### 项目做什么
用户有一个外部训的 OPSD-LoRA (SD3.5-M + HPSv3 reward)，发现应用到图生视频时严重 reward hacking（过饱和×1.95，过曝×1.65，锐边×1.86）。
Tier 1 已证实了这个 hacking。Tier 2 的目标是**修复**，有 3 条路：
1. **BoN 测试时筛选**（已 ship）—— `src/eval/tier2_bon.py` + metrics in `docs/tier2/tier2_bon_r{1,2}_metrics.md`
2. **lightness 后处理**（已 ship）—— `src/postprocess/lightness_lift.py`，Wilcoxon p=0.016 vs base
3. **重训 LoRA**（本次 H20 任务）—— trainer 已通过 M3 Pro 1-step smoke test，缺一张 GPU 来真跑 50 步

### 之前为什么没训出来
Trainer 曾经有 8 个 bug，全部已在前几次会话修完并单测锁住（见 `docs/tier2/trainer_smoke_log.md`）：
1. `UNetModel.forward` 的 `cond` 被 `**kwargs` 吞 → 自写 `_call_unet` 解包
2. `vae.decode().sample` 不存在 → 自写 `_vae_decode_video`
3. `.latent_dist.sample()` 不存在 → `.sample()`
4. LoRA 正则不匹配真实模块树 → 运行时 `_collect_temporal_attn_module_names`
5. Dataset cond 格式错 → `CondStage` + `build_cond` 装配
6. CLIP 硬编码 CUDA → 覆盖 `.device`
7. 0-D timestep → `_call_unet` 内自动 broadcast
8. T 不一致 → `rollout_until` 从 `c_concat` 读 T

M3 Pro smoke 用 mini shape `(4 × 256²)` 跑通了；H20 的任务是用 production shape `(16 × 320×512)` 跑完 50 步。

## 4 · 环境关键差异

| 项 | 开发机 (M3 Pro) | H20 容器 |
|---|---|---|
| device | MPS | **CUDA** |
| dtype | fp32 (MPS fp16 有 bug) | **fp16** (`recommended_dtype` 已自动) |
| 显存 | 48GB 共享 | **96GB HBM3 专用** |
| CUDA 版本 | — | 查 `nvidia-smi`，`h20_setup.sh` 自动选 cu118/cu121 wheel |
| OpenCLIP 下载 | 本地已缓存 | 需要重新下 `ViT-H-14 (laion2b_s32b_b79k)` + `ViT-B-32 (openai)` |
| DynamiCrafter ckpt | `weights/dynamicrafter_512/model.ckpt` 已有 | 需要重新下，9.7GB |

## 5 · 必读文件（按顺序读）

1. **`AGENTS.md`** (仓库根) —— 全局规则
2. **本文件** —— H20 会话 context
3. **`08-video-opsd-i2v/docs/H20_PLAYBOOK.md`** —— 时间线 + 三条命令 + risk register
4. **`08-video-opsd-i2v/README.md` §4** —— 项目核心叙事，特别是 "trainer end-to-end alive" 这节
5. **`08-video-opsd-i2v/docs/tier2/trainer_smoke_log.md`** —— M3 Pro 侧 smoke 的真实输出，当 H20 跑 smoke 时对比用

## 6 · 第一步该干什么

严格按 `H20_PLAYBOOK.md` 的"The three commands"顺序：

```bash
# 1. 环境探查，确认 CUDA、显存、磁盘
nvidia-smi
df -h ~
free -h

# 2. clone 仓库
cd ~ && git clone git@github.com:jingxiduan/ml-portfolio.git
cd ml-portfolio/08-video-opsd-i2v

# 3. 一键 setup（含 smoke test on production shape）
bash scripts/h20_setup.sh
```

**只有当 setup.sh 最后一行打印 `✅ H20 SETUP DONE — trainer is alive on CUDA`
才能进入下一步训练。** 如果 smoke 失败，看 `logs/h20_setup.log`，不要盲目继续。

## 7 · 预期的里程碑输出

训练完成后 push 到 portfolio 的新文件（这是**最终交付**）：

```
08-video-opsd-i2v/
├── outputs/opsd_tier2_h20/
│   ├── lora_step00010/adapter_model.safetensors
│   ├── lora_step00020/adapter_model.safetensors
│   ├── lora_step00030/adapter_model.safetensors
│   ├── lora_step00040/adapter_model.safetensors
│   └── lora_step00050/adapter_model.safetensors   ← 最终 LoRA
├── docs/tier2/
│   ├── tier2_lora_metrics.md
│   └── tier2_lora_metrics.json
└── docs/figures/duel/
    ├── tier2_lora_sun_rays.gif         (base | scale=0.2 | tier2_lora, 3-way)
    ├── tier2_lora_houses_hillside.gif
    ├── tier2_lora_pagoda_snowy.gif
    ├── tier2_lora_candle_dark.gif
    ├── tier2_lora_fog_pine_dawn.gif
    └── tier2_lora_waterfall_zoom.gif
```

训练完**最后一步必做**：push 到 GitHub，否则 10:00 资源回收数据丢失。

## 8 · 如果训练卡住 / OOM / 超时

降级策略（按顺序尝试）：

| 症状 | 对策 |
|---|---|
| 单步 > 8 min | 把 `inner_fit_steps` 从 1 保持，减 `num_steps` 到 25 |
| OOM | 把 `resolution` 从 `[320,512]` 降到 `[256,448]`；或 `num_frames` 16 → 8 |
| CUDA kernel 报错 | 检查 `torch.__version__` 和 CUDA 版本是否对得上，必要时重新装 wheel |
| 实在训不完 | **至少保证 push 已有的 ckpt**（`outputs/opsd_tier2_h20/lora_step{10,20,30,40}` 任一个都行），用它跑 infer 对比 |

## 9 · 跟用户沟通的 tone

- 诚实：数据不好不要包装；n 太少直说
- 简洁：用户不要 fluff，直接说做了什么、结果怎样、下一步什么
- 用中文回复（用户是中文母语）
- 关键决策（比如要不要降 shape 挽救训练）**问用户**，不要自作主张

## 10 · 收尾清单

- [ ] `bash scripts/h20_setup.sh` 通过
- [ ] `bash scripts/h20_run_tier2.sh` 产出 `outputs/opsd_tier2_h20/lora_step00050/`
- [ ] `bash scripts/h20_infer_compare.sh` 产出 6 个 `tier2_lora_*.gif` + metrics
- [ ] `git add -A && git commit -m "08: Tier-2 LoRA trained on H20 (50 steps, 6-prompt eval)" && git push`
- [ ] 简要向用户汇报结果（nat 均值、Δvs base/scale=0.2、最终 commit hash）
