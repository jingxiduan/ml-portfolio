# ml-portfolio · Project Status & Roadmap

> **Purpose of this doc:** 整个 ml-portfolio 的**单一事实源**。既是对已完成工作的清单，
> 也是接下来要做什么的 roadmap。所有分散在各项目 README / docs 里的「当前进度 /
> 待办 / 规则 / H20 任务」都在这里汇总，用户后续迭代直接改这个文件。
>
> **Last updated:** 2026-10-29 evening · commit `0cffe4e`

---

## 1 · 项目索引

| # | 项目 | 领域 | 状态 | 核心叙事 |
|---|---|---|---|---|
| 01 | `01-multitask-ctr-cvr-rec` | 推荐 (MovieLens) | ✅ shipped | Multi-task CTR/CVR + RankMixer dense interaction, GAUC/AUC/LogLoss |
| 02 | `02-rtb-bid-shading-control` | 竞价 (Avazu) | ✅ shipped | Win-rate 模型 + PID/MPC bid shading 反馈控制 |
| 03 | `03-reward-model-dpo-pipeline` | LLM 对齐 (HH-RLHF) | ✅ shipped | 偏好数据构造 → LLM Judge → Reward Model → DPO |
| 04 | `04-diffusion-opsd-mps` | Diffusion RL (SD-Turbo) | ✅ shipped | OPSD 从零实现 on MPS，CLIPScore reward，LoRA 400 steps |
| 05 | `05-video-opsd-mps` | 视频生成 (AnimateDiff) | ✅ shipped | OPSD 扩展到 video latent，temporal consistency reward |
| 06 | — | — | — | *no project at this slot* |
| 07 | `07-talking-head-motion-opsd` | 数字人 (JoyVASA + LivePortrait) | 🟡 Tier-2 in flight | OPSD 用于 motion diffusion，Part 2 engine routing + LinUCB |
| 08 | `08-video-opsd-i2v` | I2V 诊断修复 (DynamiCrafter + 外部 LoRA) | 🟡 Tier-2 in flight | Reward hacking 诊断 + 3 条修复 track（BoN / lightness / 真训） |

**仓库根文件**：
- `README.md` · 项目总入口
- `AGENTS.md` · 面向 AI agent 的 house rules
- `PROJECT_STATUS.md` · **本文件**（state + roadmap）

---

## 2 · 当前最活跃的工作

### 2.1 · 项目 07 · Tier-2 BoN + blink thinning（新加）

**诊断问题**（README §Honest Caveats 里诚实承认过的）：
- OPSD-LoRA 让 blink_rate 从 0 → 55/min（人类 15-20），**过量 2.8×**
- Head jitter +110%（braked by penalty 但没压平）
- 本质：和项目 08 的 reward hacking 同类（reward 边界条件被 policy 刷爆）

**已完成**（commit `0cffe4e`）：
- `src/tier2_bon.py` — anti-hacking naturalness_v2 scoring + blink thinning post-process
- `src/make_tier2_gifs.py` — 3-way side-by-side GIF 生成
- `outputs/tier2/selected_coefs.pt` (2.1 MB) — 8 clips × 5 variants motion coefs（Mac 已 BoN 选好）
- 1 个 clip 的完整 video-space 验证（见下表）
- 预览 GIF: `outputs/tier2/tier2_compare_00.gif`

**验证结果（clip 0, MediaPipe 独立评测，不是训练 reward）：**

| variant | blink_rate | jaw_audio_corr (lip-sync) |
|---|---|---|
| base_naive | 0.00 | -0.26 |
| opsd_naive | 37.5 | -0.24 |
| opsd + BoN | 31.3 | -0.24 |
| **opsd + BoN + thin** | **18.75** | **-0.23** |

→ blink_rate 完全推到人类区间（10-20），lip-sync 保持。

**待做**（H20 窗口里 30 min 内收尾）：
- 渲染剩余 7 clips × 4 variants = 28 videos → `outputs/tier2/bon_c{01..07}_*.mp4`
- 生成 8 个对比 GIF → `outputs/tier2/tier2_compare_{00..07}.gif`
- 8-clip 平均 metrics → `outputs/tier2/bon_metrics.{json,md}`
- commit + push
- （可选）07 README 加一节 "Tier-2 fix" + 把 1 clip 的表换成 8-clip 平均

**playbook**: `07-talking-head-motion-opsd/docs/H20_RENDER_PLAYBOOK.md`

### 2.2 · 项目 08 · Tier-2 LoRA 真训（已就位）

**诊断问题**：
- 外部训了一个 OPSD-LoRA (SD3.5-M + HPSv3)，应用到 I2V 显示 reward hacking：
  饱和度 ×1.95，过曝 ×1.65，锐边 ×1.86

**已完成**（commits `fbbcc41`, `d4fd926`, `e33bbd8`）：
- 3 条 Tier-2 修复 track 已 ship：
  1. Best-of-N 推理时筛选 → `docs/tier2/tier2_bon_r{1,2}_metrics.md`
  2. lightness 后处理 → `docs/tier2/verdict_table.md` (Wilcoxon p=0.016 vs base, n=6)
  3. BoN × lightness 复合 → `docs/tier2/bon_plus_lightness_table.md` (mean Δnat +0.174, 6/6 pos)
- H20 setup/run/infer 三脚本就位（`scripts/h20_{setup,run_tier2,infer_compare}.sh`）
- H20 playbook + SSH 预配置文档就位

**待做**（H20 窗口里主任务，~7h）：
- `bash scripts/h20_setup.sh` — 装环境、下 DynamiCrafter 512 ckpt、production-shape smoke（45 min）
- `bash scripts/h20_run_tier2.sh` — 50 步 OPSD 训练，ckpt_every=10 存 5 份 LoRA（4-6h）
- `bash scripts/h20_infer_compare.sh` — 6 prompts × 3 variants 推理 + metrics（1-1.5h）
- commit + push 产出

**playbook**: `08-video-opsd-i2v/docs/H20_PLAYBOOK.md`

---

## 3 · H20 窗口任务板

### 预约信息

- **时间**：10/30（周五）21:00 — 次日 11:00（14h）
- **规格**：H20 **单机 2 卡**（长时）= **28 卡时**
  - ⚠ 用户 note：本周配额 168 卡时，28 卡时够用；但 2 卡对我们的任务**没加速**
    （07 用 onnxruntime 单实例、08 trainer 没写 DDP），第 2 张卡会空转。
    想省配额可考虑在页面改成单机 1 卡 14h（14 卡时）。决定权在用户。

### 执行顺序

```
21:00 ──┬── 0. SSH key 粘贴 (3 min)        ← Mac id_ed25519 已存好
        │
        ├── 1. clone portfolio             (1 min)
        │
        ├── 2. 【07】Tier-2 render         (30 min)
        │     bash scripts/h20_setup...? → 实际走 docs/H20_RENDER_PLAYBOOK.md
        │     产出: 8 clips × 4 variants, 8 GIFs, bon_metrics.md
        │     commit + push                ← 30 min 内已经有新 demo
        │
        └── 3. 【08】Tier-2 LoRA 真训      (~7h)
              bash scripts/h20_setup.sh    (45 min, 必须 smoke pass)
              bash scripts/h20_run_tier2.sh (4-6h, nohup 后台)
              bash scripts/h20_infer_compare.sh (1-1.5h)
              commit + push
              
≈ 04:00 ── 预计全部完成，剩余 ~6h buffer
```

### 进入 H20 侧 CodeBuddy 的提示语

```
请读以下 3 份文档，然后按它们的指示执行：
1. https://github.com/jingxiduan/ml-portfolio/blob/main/AGENTS.md
2. https://github.com/jingxiduan/ml-portfolio/blob/main/08-video-opsd-i2v/docs/H20_SESSION_HANDOFF.md
3. https://github.com/jingxiduan/ml-portfolio/blob/main/PROJECT_STATUS.md  ← 本文件
```

---

## 4 · 跨会话约定（House Rules）

这些规则来自开发过程中用户的反复要求，所有 agent / 所有新窗口都要遵守。
**同步源**：`AGENTS.md`（面向 agent）+ 本文件（面向用户 + agent）。

### 4.1 · 文字 / commit 禁用词

**绝对禁止**在任何 commit message、README、文档中出现这些词（除非是合法的 paper 引用）：
- `CVPR` / `NeurIPS` / `ICCV` 等会议名作为"自己要投"的语境
- `under review` / `submission` / `paper deadline`
- `submitted to` / `submitting to`

合法用法：引用他人工作，如 `"InstructVideo [CVPR 2024]"` ✅

**Push 前 grep 检查**：
```bash
grep -iE 'cvpr|neurips|iccv|under review|submission|paper deadline' <changed files>
```

### 4.2 · 数字必须可验证

- README / docs 里的每个数字都必须能从磁盘上的 artifact 回溯
- 禁止"大致"、"约"、"估算" —— 要么给精确数字 + 来源，要么不给
- 以前出现过虚构数字被用户发现并要求删除，之后不要再出现

### 4.3 · "did not deliver" 是 feature 不是 bug

- `08-video-opsd-i2v/README.md §4` 的 honest limitations 表是设计选择，不是没写完
- 当某项 limitation 被修复，**移到 delivery 表** —— 不要静默删除
- 保留"坦诚承认不足"的章节结构

### 4.4 · 路径约定

| 环境 | 路径 |
|---|---|
| Mac dev workspace (项目 08) | `/Users/jingxiduan/CodeBuddy/20261027111906/08-video-opsd-i2v/` |
| Mac dev workspace (项目 07) | `/Users/jingxiduan/CodeBuddy/20261014132041/algo-portfolio/07-talking-head-motion-opsd/` |
| Portfolio shipping copy | `/Users/jingxiduan/CodeBuddy/20261027111906/_portfolio_repo/ml-portfolio/` |
| GitHub remote | `git@github.com:jingxiduan/ml-portfolio.git` (branch `main`) |

- 数据流向：**dev → shipping → GitHub**（通过显式 `cp` / `rsync` + `git push`）
- 不要在 shipping copy 里直接改代码然后 reverse-sync
- shipping 是 "GitHub 看到的"，shipping ≡ remote state

### 4.5 · Commit 规范

- 格式：`<项目号>: <简述>`，例：`07: Tier-2 BoN + blink-thinning`、`08: ship Tier-2 lightness`
- 一个 commit 一个逻辑变更
- 测试结果（如验收指标）放 commit message 第二行起
- 分支：`main` 唯一长期分支

### 4.6 · 不要碰的东西

| 项目 | 不要动 | 原因 |
|---|---|---|
| 08 | `src/opsd_trainer_i2v.py` 的训练循环 | 6 个单测 pin 死 |
| 08 | `src/dynamicrafter_wrapper.py` 的 `build_cond` / `CondStage` | 和 DynamiCrafter 源码对齐 |
| 08 | `configs/tier2_natural.yaml` reward 权重 | 精心调过 |
| 07 | `src/motion_reward.py` 的 reward 结构 | OPSD 已基于它训好，改了要重训 |
| 07 | `src/eye_probe.py` 的 MLP arch | 已训练并对标 MediaPipe |

---

## 5 · 关键文档索引

### 面向人类（README 系）

- `README.md` — portfolio 总入口
- 各项目 `<N>-xxx/README.md` — 项目独立说明
- 项目 08 的 `docs/tier2/` 目录 — 诊断 + 3 条修复 track 的 metrics / verdict / method

### 面向 agent（AI 入口文档）

- `AGENTS.md` — house rules + 项目简介 + 导航表
- `08-video-opsd-i2v/docs/H20_SESSION_HANDOFF.md` — H20 侧 AI 的 context 压缩
- `08-video-opsd-i2v/docs/H20_PLAYBOOK.md` — 08 窗口的操作手册
- `07-talking-head-motion-opsd/docs/H20_RENDER_PLAYBOOK.md` — 07 窗口的操作手册

### 核心代码入口

| 任务 | 入口文件 |
|---|---|
| 项目 08 真训 Tier-2 | `08-video-opsd-i2v/scripts/h20_run_tier2.sh` |
| 项目 08 Tier-2 推理对比 | `08-video-opsd-i2v/scripts/h20_infer_compare.sh` |
| 项目 07 Tier-2 BoN | `07-talking-head-motion-opsd/src/tier2_bon.py` |
| 项目 07 GIF 生成 | `07-talking-head-motion-opsd/src/make_tier2_gifs.py` |

---

## 6 · 后续迭代方向（Backlog）

分短期（1-2 周内可做）和长期（视机会）。

### 6.1 · 短期（demo 面试友好 / 补课性质）

- [ ] **项目 07**：Tier-2 BoN 完整 8 clips 完工后，README §Results 补一节
  "Tier-2 fix" 对标 Honest Caveats 部分
- [ ] **项目 07 Track C**（可选）：改 reward（blink 高斯目标 + jitter 硬 cap），
  Mac 本机重训 600 rounds（11 min），得到 `+OPSD_v2`
- [ ] **项目 07 Track D**（锦上添花）：把 BoN / thin / OPSD_v2 作为新 engine
  config 加进 `engine_profiles.py`，看 LinUCB 会不会偏好它们
- [ ] **项目 08**：H20 跑完后，Tier-2 LoRA + BoN + lightness 可以组合跑一次
  "四合一"，看能不能把 Δnat 推到 +0.3 以上
- [ ] **portfolio README**：加一个 "Methodology thread" 章节，把 04/05/07/08
  四个 OPSD 项目串成"同一套方法学的扩展"故事
- [ ] **portfolio README**：在项目 08 位置标注 "diagnose-and-fix track"，
  对标 07 的 "measure-and-route track"，突出两个项目的方法学互补

### 6.2 · 长期（有新机会 / 新数据再做）

- [ ] 项目 07 / 08 的 "trained Tier-2 LoRA" 做人工评价（小样本 n=10-20），
  看 MediaPipe / naturalness_v2 指标是否与人类感知对齐
- [ ] 项目 07 的 engine routing 用真实用户数据（而不是 simulated segments）
  重新跑一次 —— 需要先有数据
- [ ] 把 03 的 reward model + DPO pipeline 和 04/05/07/08 的 OPSD 对比写一个
  survey-style 文档，说明"RL 后训练在 generative media 的两条主路线"
- [ ] 新增项目 06（slot 预留）—— 待定

### 6.3 · 文档维护

- [ ] 每次大 commit 后更新本文件的 §1 状态、§2 当前工作、以及 last commit hash
- [ ] 新加项目时更新 §1 索引 + AGENTS.md 的导航表
- [ ] house rules 变化时同步更新 §4 和 `AGENTS.md`

---

## 7 · 历史演进（简要）

| 日期 | 里程碑 |
|---|---|
| 2026-07/08 | 项目 01/02 完工（推荐 + RTB） |
| 2026-09 | 项目 03/04 完工（LLM 对齐 + Diffusion OPSD） |
| 2026-10 初 | 项目 05 完工（Video OPSD） |
| 2026-10/14-20 | 项目 07 Part 1（JoyVASA OPSD）+ Part 2（engine routing）完工 |
| 2026-10/21-28 | 项目 08 Tier-1 诊断 + 3 条 Tier-2 track（BoN / lightness / 复合） |
| 2026-10/29 | 项目 07 加 Tier-2 BoN + blink thinning（本次会话）；H20 playbook 就位 |
| 2026-10/30 | 【待发生】H20 窗口：07 render 补齐 + 08 真训 LoRA |

---

## 8 · 迭代方式

想更新这个文档时：

1. 改 §1 的 "状态" 列（🟡 → ✅）
2. 改 §2 的 "已完成/待做"（已完成项移到 §6.3 的 "历史演进"）
3. 改 §7 的 last commit hash + §1 顶部的 Last updated
4. push

如果需要新增一类规则 / 任务板 / 项目，用相同的编号格式续写，不要拆成多个文件。
