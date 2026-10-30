# 07 · 下一轮 H20 任务书（jitter 单点优化 + 放大数据 + 独立评测）

> 写给下一个 H20 session 的 CodeBuddy。先读 `/AGENTS.md` 和 `/PROJECT_STATUS.md`，再读本文件。
> 本文件只定义"做什么、怎么判定成败、什么时候停"。不要在没过 gate 的情况下往下走。

---

## 0 · 背景（只保留与任务有关的事实）

n=20 视频空间结果（`outputs/tier2/bon_metrics.md`）：

| 指标 | base | OPSD v1 | v1+BoN+thin | v2+BoN | 人类范围 |
|---|---|---|---|---|---|
| blink/min | 0 | 50.5 | 24.8 | 25.6 | 10-20 |
| head_jitter | 120 | 262 | 265 | 257 | ~120 |
| jaw_audio_corr | -0.14 | -0.09 | -0.09 | -0.10 | 0.3-0.5 |

- blink：BoN/thin 能压到约 25，还没进人类范围，但方向对。
- **head_jitter：OPSD 后翻倍，所有手段都没动它。这是本轮要解决的唯一一个点。**
- jaw_audio_corr：base 本身就是负的，6 个 variant 不可区分，**不能再当作 lip-sync 证据**。
- 训练数据只有 25 条 macOS TTS 音频（`data/audio/train`），eval 20 条。

### 对 jitter 为什么没降的假设（待 T1 验证，不是结论）

`outputs/opsd/metrics.jsonl` 与 `outputs/opsd_v2/metrics.jsonl` 里的 `jitter` 项，训练过程中是**变差**的：

| run | step 1 | step 300 | step 600 |
|---|---|---|---|
| v1 | -0.312 | -0.389 | -0.458 |
| v2（带 hard cap） | -0.312 | -0.394 | -0.445 |

按 `motion_reward_v2.py` 里 `jitter_bounded = -(std_high 加权均值)/0.5` 反推，
std_high 从约 0.16° 涨到约 0.22°。而 hard cap 的阈值 `jitter_cap_deg = 1.0°`，
**实际值远低于阈值，hard cap 从头到尾没被触发**（relu 恒为 0，没有梯度）。
v2 的 jitter 改动因此等于没改，v1 与 v2 曲线几乎重合也符合这一点。

这与 08 里 V1 `ReLU+clamp` 梯度死区是同一类问题：惩罚项写了，但落在它不起作用的区间。

第二个可能：奖励里的 jitter 是 pose 三个维度 6-12.5Hz 频带的 std（度），
而视频空间的 `head_jitter`（MediaPipe）量的是另一个量，两者未必同向。

---

## 1 · 任务总览与 gate

| 阶段 | 内容 | 预计时间 | 通过条件（否则停下写结论） |
|---|---|---|---|
| T0 | 环境 | 20 min | `h20_render_setup.sh` smoke 通过 |
| T1 | 诊断 jitter | 40 min | 得到明确结论：cap 是否失效 / 奖励指标与视频指标是否相关 |
| T2 | reward v3（只改 jitter）在 25 条上重训 | 20 min | motion-space jitter 不高于 base 的 1.3 倍 |
| T3 | 放大数据到 ~2000 条，v2 / v3 各训一次 | 约 2 h | 同数据量下 v3 的 jitter 优于 v2 |
| T4 | 独立评测：渲染 + SyncNet + 盲评表 | 约 2 h | 见 §6 |
| T5 | 写结果、push | 30 min | |

**T2 没过 gate 就不要做 T3。** 没过就把诊断和失败原因写进结果文档，这本身是有效产出。

---

## 2 · T0 · 环境

```bash
git clone git@github.com:jingxiduan/ml-portfolio.git && cd ml-portfolio/07-talking-head-motion-opsd
bash scripts/h20_render_setup.sh
source .venv/bin/activate
nvidia-smi   # 记录两张卡当前占用
```

同机还会跑 08 的长训练（见 `08-video-opsd-i2v/docs/NEXT_H20_TASKS.md`）。
07 的 motion 训练模型很小，可以和 08 共用一张卡；用 `CUDA_VISIBLE_DEVICES` 指定，先看 `nvidia-smi` 的剩余显存再定。
渲染（LivePortrait）放到 T4 再做。

---

## 3 · T1 · 诊断（新增 `src/diag_jitter.py`）

输入：`outputs/opsd/lora.pt`、`outputs/opsd_v2/lora.pt`、base，在 `data/audio/eval` 的 20 条上各生成 motion coefs
（直接复用 `outputs/tier2/selected_coefs.pt` 里已有的 20 clip × variant，不必重采样）。

要回答三个问题，结果写到 `outputs/diag/jitter_diag.json` 和 `docs/RESULTS_NEXT.md`：

1. **cap 是否失效**：对 base / v1 / v2 各自算 reward 里的 `std_high`（pose 三轴 6-12.5Hz），
   报 mean、p90、max。若 p90 远小于 1.0°，则假设成立。
2. **各奖励项的梯度占比**：对一批 coef 反传，分别记录 `amp / jitter / couple / blink / gaze / ood`
   对 coef 的梯度 L2 范数。判断 jitter 项相对 amp、blink 是否被淹没。
3. **奖励指标与视频指标是否同向**：在 20 clip × 6 variant 共 120 个点上，
   算奖励侧 `std_high` 与 `outputs/tier2/bon_metrics.json` 里视频侧 `head_jitter` 的 Spearman 相关。
   同时在 `src/video_metrics.py` 里确认 `head_jitter` 的定义和单位，写进结果文档。

判定：
- 问题 1 成立 → T2 用"按 base 分布标定的阈值"。
- 问题 3 相关性低（|rho| < 0.5）→ T2 的 jitter 项要改成视频侧同口径的量（例如姿态的二阶差分），并在文档里说明为什么。
- 两个都不成立 → 停下，把现象写清楚，不要硬往下做。

---

## 4 · T2 · reward v3（只改 jitter 一处）

新增 `src/motion_reward_v3.py`、`src/opsd_train_motion_v3.py`，写法照 v2（薄封装，复用
`opsd_train_motion.py` 的 `add_lora / build_windows / guided_x0 / rollout_to / set_mode`）。

**只改 jitter 项，其余 amp / couple / blink / gaze / ood 和 v2 保持完全一致**，这样差异可归因。

- 把 `jitter_cap_deg` 从 1.0° 改成 T1 得到的 base 分布 p90 附近（预计约 0.2°，以 T1 实测为准）。
- 保证 cap 在训练初期就有梯度（记录 cap 触发比例，写进 metrics）。
- 其余超参与 v2 相同：600 rounds、lora-r 8、lr 3e-4、seed 0。

产出：`outputs/opsd_v3/lora.pt`、`outputs/opsd_v3/metrics.jsonl`（额外记录 `cap_active_frac`）。

Gate：用 `src/tier2_bon.py --motion-only` 在 eval 20 clip 上评测，
motion-space `jitter_hp` 不高于 base 的 1.3 倍（base 约 13.6，v1/v2 约 24-27），
且 blink 不比 v2 更差。没过就回到 T1 重新看，不进 T3。

---

## 5 · T3 · 放大数据（只在 T2 过 gate 后）

原因：现在训练只有 25 条 TTS 音频，结论很难外推。OPSD 训练只需要音频，motion 空间训练很便宜，放大成本低。

数据：
- 目标约 2000 条训练、100 条评测，8-15 秒，16kHz 单声道，**训练与评测说话人互不重叠**。
- 优先用公开中文语音集（如 AISHELL 系列，先确认 H20 能联网下载、许可允许）。
- 下载不通则退到多音色开源 TTS 生成，但要在文档里写明数据来源，且不要和现有 25 条混为一谈。
- 数据清单写到 `data/audio_manifest_large.json`，**音频文件本身不要进 git**。

对照实验（保证"提升来自改动，而不是来自数据量"）：

| 组 | 奖励 | 训练数据 | rounds |
|---|---|---|---|
| A | v2 | 2000 条 | 3000 |
| B | v3 | 2000 条 | 3000 |

已有的 25 条数据上 v2 / v3 结果作为小规模参照。三组都固定 seed 0，B 若有余量再加 seed 1、2。

---

## 6 · T4 · 独立评测（不能再用奖励自己的项当结论）

当前 blink、jitter 既是训练奖励又是评测指标，这是循环论证。本轮必须补上独立的量。

1. **SyncNet LSE-C / LSE-D**：对渲染视频算口型-音频同步。需要 SyncNet 预训练权重，先确认能下载；
   下载不到则在文档里写明未做，不要用 `jaw_audio_corr` 顶替。
2. **盲评表**：从 50 条 eval 中取 30 条，`base / v2 / v3(+BoN)` 随机配对并随机左右，
   生成 `outputs/human_eval/sheet.csv`（只有编号与评分列）和 `key.csv`（对应关系，单独保存）。
   暂时不需要有人打完，只需要生成好，后面由人来评。
3. **视频侧 MediaPipe 指标**（沿用 `video_metrics.py`）：blink_rate、head_jitter、gaze_std 等。
4. 渲染规模：50 条 eval × 4 个 variant（base / v2 / v3 / v3+BoN）= 200 个视频，约 1.2 h。
   **mp4 不要 commit**（上一轮 120 个 mp4 已让 `outputs/tier2` 占 751MB），只 commit GIF、json、md。

### 成功标准（先写下来，结果出来后不改）

- 视频侧 head_jitter：v3 ≤ base 的 1.4 倍（base 约 120，即 ≤ 约 170），v2 目前约 260。
- blink：v3+BoN 不比 v2+BoN（25.6）更差。
- SyncNet：v3 不比 v2 差于噪声水平（报数值和标准差）。
- 达不到：如实写"jitter 单点优化未奏效，原因是……"，附 T1 的诊断。

---

## 7 · 时间预算与停止线

| 阶段 | 时间 |
|---|---|
| T0 | 0.3 h |
| T1 | 0.7 h |
| T2 | 0.4 h |
| T3 | 2.0 h |
| T4 | 2.0 h |
| T5 | 0.5 h |
| 合计 | 约 6 h |

和 08 共用 GPU 时，07 排在 08 长训练的空档里。超过窗口剩余时间的 80% 时，
砍 T3 的第二、第三个 seed，不砍 T1 与 T4。

---

## 8 · 提交规则

- 每个阶段结束就 commit + push，不要攒到最后（上一台机器的 mp4 在到期时丢过）。
- 用 SSH key 推送，**不要把任何 token 写进文件、remote URL 或命令历史**。
- push 前按 `/AGENTS.md` 的 house rules 做禁用词检查。
- 结果统一写 `docs/RESULTS_NEXT.md`，数字照实写，包括失败的。
  README 与 `TIER2_NOTE.md` 里的措辞要和结果一致（例如 jitter 没降就不能写成"已缓解"）。
- 完成后更新 `STATUS.md` 与 `/PROJECT_STATUS.md` 的 07 一节。
