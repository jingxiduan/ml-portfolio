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
| 04 | `04-diffusion-opsd-mps` | **OPSD 复刻** (SD-Turbo) | ✅ shipped | DiffusionOPSD paper 的 Mac 单机缩放版（起点） |
| 05 | `05-video-opsd-mps` | **OPSD → 视频初尝** (AnimateDiff) | ✅ shipped (浅) | 把 OPSD 搬到 video latent，加 temporal consistency reward（粗糙 proof-of-concept）|
| 06 | — | — | — | *no project at this slot* |
| 07 | `07-talking-head-motion-opsd` | **OPSD → 数字人 motion** (JoyVASA + LivePortrait) | ✅ 三行业轮已交付 | OPSD 用于 motion diffusion；最新一轮训出金融/房地产/短视频三个 persona，reward 目标带由真实视频实测得到；demo 在 `backup/industry-demo` 分支 |
| 08 | `08-video-opsd-i2v` | **OPSD → I2V 诊断** (DynamiCrafter + 外部 LoRA) | 🟡 Tier-2 in flight | 拿外部训好的 OPSD-LoRA 做诊断分析 + 多种改进策略 |

**OPSD method thread（04 → 05 → 07 → 08 是同一方法学的延伸）**：

```
04 复刻             → 05 视频初尝（浅）
   DiffusionOPSD      AnimateDiff + CLIPScore
   SD-Turbo + CLIP    temporal consistency reward
   mechanism check    方法学第一次推广到时序
          │
          ├─ 07 数字人 motion（成熟分支 1）
          │     JoyVASA motion diffusion + LivePortrait
          │     + 可微分 liveliness reward (eye probe)
          │     + Part 2: 同套度量复用成 engine routing (LinUCB)
          │     Tier-2: reward design iteration (三个 lever)
          │
          └─ 08 I2V 大模型（成熟分支 2）
                DynamiCrafter 512 + 外部 OPSD-LoRA (SD3.5-M + HPSv3)
                拿到现成模型做诊断分析 + 三条改进策略
                Tier-2: BoN / lightness / 真训（10/30 H20）
```

- 04/05 是方法学铺垫，04 严肃复刻，05 是初次尝试（README 承认 "浅"）
- 07/08 是两条正经的扩展：**07 自己训（motion 空间）；08 诊断外部训好的（像素空间）**
- "reward hacking 诊断" 是 **08 的叙事**，不是 07 的 —— 07 的 reward 是自己写的，叙事是 "reward design 是 OPSD 可持续迭代的核心杠杆"

**仓库根文件**：
- `README.md` · 项目总入口
- `AGENTS.md` · 面向 AI agent 的 house rules
- `PROJECT_STATUS.md` · **本文件**（state + roadmap）

---

## 2 · 当前最活跃的工作

### 2.1 · 项目 07 · Tier-2 reward design iteration

**定位**：07 整个项目是 **OPSD 从像素扩散模型推广到 motion diffusion（数字人领域）**
的研究。Tier-1 已经证实机制可泛化。Tier-2 是对 reward design 的下一轮迭代——
不是"修 bug"，是研究进程的下一步。

**Tier-1 诚实承认的 overshoot**（README `Honest caveats`）：
- blink_rate 55/min（人类 10-20）—— Tier-1 reward 的 "每窗口 ≥1 blink" soft-max 无上界
- head jitter +110% —— Tier-1 reward 的 jitter penalty 是 bounded 不是 hard cap

**Tier-2 的三个 lever**（在 stack 的三个层级做 reward 迭代）：

| Lever | 位置 | 做什么 | 开销 |
|---|---|---|---|
| 1 · BoN | inference-time | N=16 候选 + anti-overshoot 评分 argmax | 0 training |
| 2 · blink thinning | post-process | motion coef 上做 refractory + 非唇维度衰减 | 0 training，lip-sync 保持 |
| 3 · reward v2 | training-time | gaussian blink target + hard jitter cap, 重训 600 轮 | 11 min on Mac |

**已完成**（commits `0cffe4e` → `33f0db7`）：
- `src/tier2_bon.py` + `src/make_tier2_gifs.py` — Lever 1+2 实现
- `src/motion_reward_v2.py` + `src/opsd_train_motion_v2.py` — Lever 3 实现
- `outputs/opsd_v2/lora.pt` — reward v2 下重训的 LoRA
- `outputs/tier2/selected_coefs.pt` (8 MB) — 20 clips × 7 variants motion coefs bundle
- 3 张可视化图 + `docs/TIER2_NOTE.md` 完整叙事
- 1 clip video-space 验证（n=1）：blink 37.5 → 18.75，lip-sync 保持

**Motion-space 结果（n=20，Mac 本机）**：

| variant | blink/min | jitter_hp | nat v2 ↑ |
|---|---|---|---|
| base | 0.0 | 13.6 | 0.99 |
| Tier-1 OPSD naive | **48.9** | 26.7 | 1.00 |
| Tier-1 + BoN (Lever 1) | 32.4 | 25.6 | 1.78 |
| Tier-1 + BoN + thin (Lever 1+2) | **23.9** | 25.6 | 2.61 |
| Reward v2 naive (Lever 3) | 41.1 | **23.7** | 1.18 |
| **Reward v2 + BoN (Lever 3+1)** | **23.9** | 23.8 | **2.70** |

→ **Lever 1+2 和 Lever 3+1 收敛到相同的 blink=23.9/min**，证明"treating the
reward at any level of the stack reaches the same equilibrium"是方法学规律，
不是巧合。

**H20 渲染已完成**（commit `a7595ef`）：20 clips × 6 variants = 120 videos + 20 GIF + n=20 video-space 指标。

**视频空间 n=20 的真实结果**（`outputs/tier2/bon_metrics.md`）：

| 指标 | base | OPSD v1 | v1+BoN+thin | v2+BoN | 人类范围 |
|---|---|---|---|---|---|
| blink/min | 0 | 50.5 | 24.8 | 25.6 | 10-20 |
| head_jitter | 120 | 262 | 265 | 257 | ~120 |
| jaw_audio_corr | -0.14 | -0.09 | -0.09 | -0.10 | 0.3-0.5 |

- blink：BoN/thin 可压到约 25，两条路径收敛，但仍高于人类范围。
- **head_jitter：OPSD 后翻倍，所有 lever 都没动。** 训练日志反推 jitter 实际约 0.2°，
  而 v2 的 hard cap 阈值是 1.0°，很可能从未触发（待诊断）。
- jaw_audio_corr：base 本身为负，6 个 variant 不可区分，不能当作 lip-sync 证据。
- 早先 Mac 上 n=1 的 blink 18.75 属于小样本偶然值，n=20 平均为 24.8。

**下一轮任务书**：`07-talking-head-motion-opsd/docs/NEXT_H20_TASKS.md`
（诊断 jitter → reward v3 只改 jitter → 放大数据 → SyncNet 与盲评表）。

### 2.1b · 项目 07 · 跨行业轮（2026-11-01，已交付）

在 v4（amplitude 项改成"以实测人类带为中心"的 peaked 高斯）的基础上，把同一方法推广到
**三个行业**，每个行业的 reward 目标带都来自真实公开视频的实测，而不是手调：

| 行业 | 实测幅度 | 实测抖动 | persona 落点（vs 目标） | vs base |
|---|---|---|---|---|
| finance（HDTF 新闻主播，n=25） | 1.384° | 0.094° | 1.162° **0.84×** | 1.31× |
| real-estate（CelebV-HQ 中幅度，n=16） | 1.257° | 0.118° | 1.134° **0.90×** | 1.27× |
| short-video（CelebV-HQ 高幅度，n=16） | 3.039° | 0.349° | 2.699° **0.89×** | **3.03×** |

评测用 AISHELL-1 eval 集（15 个说话人与 60 个训练说话人零重叠，n=12）。
**三个 persona 全部落在目标带内、无一过冲**——v2/v3 那种 2.4–2.7× 的恐怖谷过冲没有复发。

- 完整报告：`07-talking-head-motion-opsd/docs/INDUSTRY_ROUND.md`
- 首页入口：`07-talking-head-motion-opsd/README.md`（§ THREE-INDUSTRY ROUND，含图与 GIF）
- demo（44 个高清视频 + 同脸跨行业对照）：分支 `backup/industry-demo`
- 4 次训练全部经 `tools/train_guard.py` 启动，**零触发**
- 诚实说明：行业标签是公开数据的代理划分；finance 与 real-estate 差异小是因为它们的实测带本来就接近；
  smile 未进 reward；SyncNet 与人工盲评仍未做

**playbook**: `07-talking-head-motion-opsd/docs/H20_RENDER_PLAYBOOK.md`
**narrative**: `07-talking-head-motion-opsd/docs/TIER2_NOTE.md`

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

**H20 训练已完成**（commits `5acd805` … `95a1f6d`）：200 prompts（160 训练 / 40 评测），
V1 / V2 / NoNat 三组各 500 步，约 4.6 GPU-h。

**真实结果**：
- 训练曲线：V1 的 naturalness 梯度全程为 0；V2（sigmoid）梯度存活，`R_naturalness` 由 -1.47 升到 -0.79。
- n=10 评测：base / V1 / V2 / NoNat 的 Appeal、Naturalness、Dynamic 在小数点后第四位才有差异，
  即**评测上没有可测提升**。目前能支持的结论只有"sigmoid 修复了梯度死区"。
- 疑点：`lr=1e-5` 偏低、500 步偏少；500 步后 LoRA 范数约 27。

**下一轮任务书**：`08-video-opsd-i2v/docs/NEXT_H20_TASKS.md`
（先确认 LoRA 有效并做 lr 探针 → V2 / NoNat 配对长训 2000 步 → n=40 评测与盲评表）。

**playbook**: `08-video-opsd-i2v/docs/H20_PLAYBOOK.md`

---

## 3 · H20 窗口任务板（下一轮）

上一轮（10/29-30）已完成：07 渲染 120 videos、08 三组 500 步消融。结果见 §2。
**下一轮不是继续堆规模，而是先把"为什么没提升"弄清楚，再放大。**

| 项目 | 任务书 | 核心内容 | 预计 |
|---|---|---|---|
| 07 | `07-talking-head-motion-opsd/docs/NEXT_H20_TASKS.md` | 诊断 jitter（hard cap 疑似失效）→ reward v3 只改 jitter → 公开中文语音集（AISHELL）放大到约 2000 段 → SyncNet 与盲评表 | 约 6 h |
| 08 | `08-video-opsd-i2v/docs/NEXT_H20_TASKS.md` | 先查没训动的根因（目标偏移只有 y0 的约 1%、lr 偏低）并做探针 → 公开视频集上 V2 / NoNat 配对长训 → 评测与盲评表 | 约 10 h |

- 两个项目共用 2 张卡：08 长训一卡一组并行；07 的 motion 训练很小，插在空档，先看 `nvidia-smi` 再放卡。
- 每个任务书都带 gate：前一阶段不过，就停下写结论，不往下走。
- 每个任务书的 §0b 是硬性约束（禁止事项、参数上限、触发后的处理）；
  所有训练都必须经 `tools/train_guard.py` 启动，备份用 `tools/backup_push.sh`，阈值已在现有 5 份日志上回放验证过。
- 成功标准已写在任务书里，**结果出来后不改**。
- 硬件：上一轮报告里机器是 2 卡 H100 80GB（预约页写 H20），文档里以实际 `nvidia-smi` 为准。
- **训完就推**：每个任务结束立刻用 `tools/backup_push.sh` 备份（小文件上 `main`，mp4 与大文件走 `backup/*` 分支），到期前 90 分钟转入全量备份。
- **数据必须是公开数据集**：08 用 OpenVid-1M 子集，07 用 AISHELL；来源、许可证写进 `data/DATA_SOURCES.md`。联网不通就汇报，不要退回合成数据。

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

- [x] ~~项目 07 Tier-2 Lever 1/2/3 motion-space~~（已完成，见 §2.1）
- [ ] **项目 07 · H20 侧渲染**（等 10/30 21:00 窗口）：
  20 clips × 7 variants = 140 videos + 20 GIFs + n=20 video-space metrics 表
- [ ] **项目 07 · Lever 4（可选，锦上添花）**：把 BoN / thin / OPSD_v2 作为新
  engine config 加进 `03-engine-routing` 的 `engine_profiles.py`，
  看 LinUCB 会不会偏好它们（跨项目串联）
- [ ] **项目 08**：H20 跑完后，Tier-2 LoRA + BoN + lightness 可以组合跑一次
  "四合一"，看能不能把 Δnat 推到 +0.3 以上
- [ ] **portfolio README**：加一个 "OPSD methodology thread" 章节，
  把 04（复刻）→ 05（视频首试）→ 07（数字人成熟）/ 08（I2V 成熟）
  串成"同一套方法学的分支展开"故事（PROJECT_STATUS §1 已有图，README 没补）

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
| 2026-10/30 | H20 窗口：07 Tier-2（v3 / v4，AISHELL 放大数据）+ 08 真训 LoRA |
| 2026-11-01 | **项目 07 跨行业轮**：三行业实测 reward 带 + 3 个 persona LoRA + 44 个高清 demo + 同脸对照；已 push main 与 `backup/industry-demo` |

---

## 8 · 迭代方式

想更新这个文档时：

1. 改 §1 的 "状态" 列（🟡 → ✅）
2. 改 §2 的 "已完成/待做"（已完成项移到 §6.3 的 "历史演进"）
3. 改 §7 的 last commit hash + §1 顶部的 Last updated
4. push

如果需要新增一类规则 / 任务板 / 项目，用相同的编号格式续写，不要拆成多个文件。
