# 07 · 下一轮 H20 任务书 **v2**（让"训完"肉眼可见地优于 base）

> 版本 v2 · 2026-11-01。**本文件取代 `docs/NEXT_H20_TASKS.md` 的任务范围**
> （旧任务书的 T1–T3 已于 10-30 跑完，结果在 `docs/RESULTS_NEXT.md`；
> 它的 §0b "只改 jitter cap" 这一条在本轮取消，原因见 §0.3）。
>
> 写给下一个 H20 session 的 CodeBuddy。先读 `AGENTS.md`、`PROJECT_STATUS.md`、
> `docs/INDUSTRY_ROUND.md`，再读本文件。

---

## 0 · 这一轮要解决的唯一问题

不是指标，是观感：

> **base 与训完的 persona 两个视频并排放，必须第一眼看出"训完的更像真人"，
> 而且 base 身上那个毛病被明确地修掉了。**

现状（2026-11-01 已交付的三行业轮）：finance 只有 base 的 **1.31×**、realestate **1.27×**，
看起来就是"同一张脸晃得稍微多一点"；而且训完的版本**反而更假**。

### 0.1 诊断：为什么训不出可见差别（已确认，不是假设）

v4 reward 六个项里，四项在调 pitch/yaw/roll 这 **3 个数**：

| 项 | 权重 | 实际优化对象 | 维数 |
|---|---|---|---|
| `amp` | 1.0 | 头部低频幅度的高斯峰 | **3（pose）** |
| `jitter` | 0.5 | 头部高频抖动 → 0 | **3（pose）** |
| `couple` | 0.5 | 4 s 窗口内「响度包络 vs 头部速度」的全局 Pearson | **1 个标量** |
| `blink` | 0.5 | 每 2 s 眨眼信号 smoothmax 峰值 − 超标惩罚 | 经 eye_probe |
| `gaze` | 0.3 | `tanh(gaze_std/0.1)` —— 越飘越好 | 经 eye_probe |
| `ood` | 1.0 | 只裁剪 `|exp|>3` | 无方向性 |

63 维表情里只有眨眼/注视被间接碰到，**眉 / 颊 / 嘴角 / 下颌随语音的微动没有任何一项奖励**，
因此表情基本冻结在源肖像上（`INDUSTRY_ROUND.md` §5 自己写了：微笑来自源肖像，不在 reward 里）。

→ **可调空间 = "头晃多大"这一个旋钮。** 这就是 finance = 1.31× 且看不出差别的全部原因。
v4 把幅度调进目标带（0.84–0.90×）这一步做对了，但做完之后**剩下的可调空间本来就是零**。

### 0.2 三个已经量出来的"假"（不是感觉）

1. **眨眼率超标 2–5 倍。** 实测人类 15 次/min（带 7.5–37.5）；base 只有 5.5；训完在渲染视频上
   用 EAR 谷值数出 **35–77 次/min**（`INDUSTRY_ROUND.md` §7.6 已承认但未修）。
   base = 不眨眼（僵），训完 = 神经质地眨。**中间 15–18 次/min 的位置，两边都没站上。**
   根因：blink 项奖励的是窗口内**峰值**（smoothmax）并只压 `mean`，控制不了**频率**。
2. **静音段完全没被约束。** `couple` 是 4 s 全局 Pearson，静音时 `env≈0` 梯度消失，
   于是模型在"听"的时候仍按同一节奏晃。真人是**几乎静止 + 偶发小幅点头**。这是"假"的第一大来源。
3. **表情是常量 + 只有头动。** 头在动、脸是死的；肩/上半身是静止图，头与身体无耦合 → "贴上去的头"。

**base 显得更自然，是因为它动得少，把这三个缺陷藏起来了。** 单纯加幅度会放大假感
（这正是 v2/v3 恐怖谷的机制），v4 靠压幅度绕开了它，但**绕开 ≠ 解决**。

### 0.3 这一轮的打法

把优化对象从"3 个幅度标量"换成"一整套肉眼可见的行为事件"：

| 新增 / 重写 | 解决什么 | 阶段 |
|---|---|---|
| **`expr` 表情动态** | 让脸活起来（眉/颊/下颌随语音微动） | **P1（先只做这一个）** |
| `listen` 静音段行为 | 会"听"，不再匀速抖 | P2 |
| `phase` 相位锁定 | 跟重音点头，而不是匀速晃 | P2 |
| `blink` 事件化重写 | 眨眼回到 15–18 次/min | P3 |
| `gaze` 注视-扫视重写 | 眼睛不再一直游移 | P3 |
| 渲染侧：半身 + 画质 | 去掉"贴上去的头"，提升清晰度 | P4 |
| 大量 demo | ≥10 组/行业 + 慢放 + 眼部裁切 | P5 |

**顺序原则：一次只加一组项。** 这不是硬规则，是为了还能归因——
先只加 `expr` 训一版，确认"脸活了"，再叠 listen/phase，再动 blink/gaze。

---

## 0b · 硬性约束（本轮只有这五条）

1. **窗口结束前 60 分钟必须完成全量备份并推送**（§0c）。
2. **不删除任何已有文件**，不 `git push --force`，不 `git reset --hard`。只新增文件/目录。
3. 所有训练一律经 `tools/train_guard.py` 启动，先跑 `--selftest` 必须输出 `selftest OK`。
4. 数字必须可验证；不达标就写 FAIL，**不许改成功标准，不许把失败改写成"已缓解"**。
5. 每个任务一结束（正常跑完 / 被守护脚本停掉 / 手动停掉），**立刻备份，再开始下一个**。

> 旧任务书 §0b 的"只改 jitter cap、其余与 v2 完全一致"在本轮**取消**。
> 可归因性改由"一次只加一组项"的分批策略保证（§0.3），而不是靠锁死 reward 公式。

## 0c · 备份与推送

| 内容 | 命令 | 去向 |
|---|---|---|
| 日志 jsonl / metrics / 诊断 json / 图 / GIF / LoRA / 清单 / md | `tools/backup_push.sh small "<msg>" <paths...>` | 当前分支 |
| mp4、音频子集、任何 >90 MB 的文件 | `tools/backup_push.sh big r3-07 <paths...>` | 分支 `backup/r3-07` |
| 看还有多少没推出去 | `tools/backup_push.sh status` | |

- **停止线：窗口结束前 60 分钟不再启动新任务**，转全量备份：
  先 `big` 打包所有 mp4 与渲染输出，再 `small` 兜底推 `main`。
- `backup/*` 分支总量 ≤ 1.5 GB（与 08 共用）。超了先压 480p / crf 28；仍超就只留评测集与最终产出，
  并在结果文档里写明舍弃了什么。
- 用机器上已配好的 SSH 推送，**不要**把任何凭据写进文件或命令。

---

## 1 · 阶段总览与 gate

| 阶段 | 内容 | 预计 | gate（不过就停下写结论，不往下走） |
|---|---|---|---|
| T0 | 环境 + 复现现状 | 0.5 h | 三行业 LoRA 渲出的 base vs persona 与已交付 demo 一致 |
| **C1** | **标定**：表情动态 / 静音段 / 相位 / 眨眼事件 / 注视结构 | 1.5 h | 五个量都测出 median + IQR，且量纲与 reward 内部一致 |
| **P1** | **只加 `expr` 项**，三行业各训一次 | 1.0 h | **"脸活了"**：非唇表情低频 std ≥ base 的 1.5×，且 ood 不恶化 |
| **P1e** | P1 的 demo 渲染 + 人眼确认 | 1.0 h | 并排看得出差别，且没有变假 |
| P2 | 叠 `listen` + `phase` | 1.0 h | 静音段头部幅度 ≤ 语音段的 0.6×；onset 命中率高于 base |
| P3 | `blink` / `gaze` 重写 | 1.0 h | 渲染视频眨眼 12–22 次/min |
| P4 | 渲染侧：半身 + 画质 | 1.5 h | 输出带肩，且看不出头身接缝 |
| P5 | 大量 demo + 图 + 表 | 1.5 h | 每行业 ≥10 组 |
| P6 | 写结果、更新 README/STATUS、push | 0.7 h | |

合计约 **9.5 h**。**C1 在 CPU 上做，与 T0 / P1 并行，不占 GPU。**

---

## 2 · T0 · 环境

```bash
git clone git@github.com:jingxiduan/ml-portfolio.git && cd ml-portfolio/07-talking-head-motion-opsd
bash scripts/h20_render_setup.sh
source .venv/bin/activate
nvidia-smi                              # 记录两张卡当前占用
python ../tools/train_guard.py --selftest   # 必须输出 selftest OK
```

回显基准（确认环境没坏）：

```bash
python scripts/eval_industries.py --n-clips 12 --out outputs/industry_eval_r3_baseline.json
```

应与 `docs/INDUSTRY_ROUND.md` §3 的表一致（base 0.890°、finance 1.162°、realestate 1.134°、
shortvideo 2.699°）。差得多就先修环境，不要往下走。

---

## 3 · C1 · 标定（本轮所有目标带的来源，全部实测）

沿用行业轮成功的做法：**目标带从真实公开视频实测得到，不手调。**
新增 `scripts/calibrate_behaviour_bands.py` → 输出 `outputs/behaviour_bands.json`。

输入：`data/videos_public/hdtf/*`（finance）、`data/videos_public/celebv/*`
（realestate / shortvideo，沿用 `outputs/industry_bands.json` 的分组），音频用视频自带人声。

### 3.1 量纲对齐（关键，别踩 v4 §7.7 那个坑）

- **pose**：MediaPipe + solvePnP，与 v4 同口径（已验证）。
- **表情：必须在 LivePortrait 系数空间里测。** 用 `src/engine.py` 里已有的
  `LivePortraitWrapper` 的 motion extractor 逐帧提 63 维 exp，再套**与 reward 内部完全相同**
  的 Butterworth 带通（`src/motion_reward.py:_bandpass`），这样量纲才能直接对齐。
  MediaPipe 给不出 LivePortrait 的 keypoint，**不能**用来定表情目标。

### 3.2 要测的量

| 量 | 定义 | 用在哪 |
|---|---|---|
| `expr_low_std` | 非唇 exp 维（排除 `LIP_DIMS`）0.3–3 Hz std 的 median | `expr` 项 target |
| `expr_env_corr` | `Pearson(env, |d expr_low/dt|)` | `expr` 项耦合权重 |
| `silence_ratio` | 静音段 low_std(pose) ÷ 语音段 low_std(pose) | `listen` 项 target |
| `silence_nod_hz` | 静音段 1–3 Hz 功率占比 | `listen` 项 target |
| `onset_hit` | 语音 onset 后 80–240 ms 内出现速度峰的比例 | `phase` 项 target |
| `onset_lag_ms` | 该速度峰的延迟 median | `phase` 项窗口 |
| `blink_rate` / `blink_dur_s` / `ibi_cv` | EAR 谷值事件检测（率 / 时长 / IBI 变异系数） | `blink` 重写 target |
| `gaze_saccade_sparsity` / `fixation_dwell_s` | 注视速度 L1/L2 比、低速帧占比 | `gaze` 重写 target |

VAD：能量法即可（`env` 相对该 clip 的分位数 + 迟滞），不引入新依赖；
若 `silero-vad` 可下载则优先用，并在 `data/DATA_SOURCES.md` 记明。

产出：`outputs/behaviour_bands.json`（每行业 × 每量 的 median / IQR / n）。
**测不出来就如实写"未测"，不要用手调值顶替**——v4 §7.7 的教训正是"量纲对不上还硬直译"。

---

## 4 · P1 · 只加 `expr`（让脸活起来）

### 4.1 新增 `src/motion_reward_v5.py`

继承 `LivelinessRewardV4`，**只新增一个 `expr` 项；`amp / jitter / couple / blink / gaze / ood`
的权重与公式逐字照抄 v4**。起步权重 `expr = 1.0`，三个行业先共用同一个值。

```python
NON_LIP = [i for i in range(63) if i not in LIP_DIMS]
exp     = coef[..., NON_LIP]
low     = _bandpass(exp, 0.3, 3.0)                 # 与 pose 用同一个滤波器
std_low = (low.pow(2).mean(1) + 1e-6).sqrt()       # (B, |NON_LIP|)
dev     = (std_low.mean(-1) - self.expr_target) / self.expr_tol
expr    = torch.exp(-0.5 * dev * dev)              # 与 amp 同形状的 peaked 高斯
```

`expr_target` / `expr_tol` 从 `outputs/behaviour_bands.json` 读（各行业的实测 median；
`tol` 起步用 v4 已验证的 1.0，若该行业 IQR 更宽则取 `max(1.0, 0.5*(p75-p25))`）。

**必须记录 `expr_active_frac`**（std_low 落在 `target ± tol` 内的样本比例）并写进 `metrics.jsonl`。
这一项若全程为 0 或 1，说明 target 定错了——与 v2 的 jitter cap 是同一类失效。

### 4.2 新增 `src/opsd_train_motion_v5.py`

照 `src/opsd_train_motion_v4.py` 薄封装，只换 reward 类并加
`--expr-target / --expr-tol / --expr-weight`。**保留 `lip_mask` 逻辑**（不动唇，保 lip-sync）。

### 4.3 训练（三行业各一次，各约 14 min / H20）

沿用 `outputs/industry_bands.json` 的已验证幅度参数，以及行业轮修眼半闭后的 blink 参数：

| 行业 | `--amp-target-deg` | `--jitter-cap-deg` |
|---|---|---|
| finance | 1.3838 | 0.1883 |
| realestate | 1.2570 | 0.2358 |
| shortvideo | 2.9299 | 0.5000 |

（另固定：`--amp-tol-deg 1.0 --blink-target-mean 0.12 --blink-overshoot-penalty 8.0`）

```bash
OUT=outputs/opsd_v5_finance
mkdir -p $OUT
nohup env CUDA_VISIBLE_DEVICES=0 python src/opsd_train_motion_v5.py \
    --out $OUT --data-dir data/audio_public/train --rounds 3000 --seed 0 \
    --amp-target-deg 1.3838 --amp-tol-deg 1.0 --jitter-cap-deg 0.1883 \
    --blink-target-mean 0.12 --blink-overshoot-penalty 8.0 \
    --expr-target <C1 实测> --expr-tol <C1 实测> > $OUT/train.log 2>&1 &
TRAIN_PID=$!
nohup python ../tools/train_guard.py --kind motion --log $OUT/metrics.jsonl \
    --pid $TRAIN_PID > $OUT/guard.log 2>&1 &
```

### 4.4 Gate（P1）

用 `src/tier2_bon.py --motion-only` 在同一组 eval 上测：

- **非唇表情低频 std ≥ base 的 1.5×**（base 记 1.00×）——这是"脸活了"的定量判据；
- `ood` 不比 v4 差（不得 < −0.05）；
- `amp` 仍在目标带 0.8–1.1×（不许为了涨表情把头压死或放大）；
- `expr_active_frac` 落在 0.2–0.8（说明 target 有梯度）。

**任一不过就停**：把数值写进 `docs/RESULTS_R3.md` 并汇报，不要进 P2。

---

## 5 · P1e · 人眼确认（本轮最重要的一步）

```bash
python scripts/render_industry_demo.py --manifest data/industry_demo_manifest_v3.json \
    --lora-dir outputs --out outputs/v5_demo --gpu 0 --tag-prefix v5_
python scripts/make_demo_gifs.py --video-dir outputs/v5_demo/finance \
    --out outputs/v5_gifs/finance --variants base finance
```

每行业渲 3 组 base vs v5（1280 px + pasteback）。看三件事，都写进 `docs/RESULTS_R3.md`：

1. **脸有没有活**：眉 / 颊 / 下颌是否出现随语音的微动（base 是常量）；
2. **有没有变假**：眨眼是否更夸张、脸有没有被拉扯变形；
3. **行业感还在不在**：三个 persona 是否还分得开。

**如果"活了但变假了"，先停在 P1**：把变假的那个维度（多半是眨眼频率或幅度）单独调，再进 P2。

---

## 6 · P2 · 叠 `listen` + `phase`

新增 `src/motion_reward_v6.py`（新增文件，不改 v5），在 v5 之上再加两项：

- **`listen`**：用 VAD 把窗口切成语音/静音两段，两段各一个 peaked 目标
  （语音段 = `amp_target`，静音段 = `amp_target × silence_ratio`）；
  另加一项静音段 1–3 Hz 功率占比的 peaked 项（目标 `silence_nod_hz`）。
  **原来的全局 `amp` 保留不动**，用于对照。
- **`phase`**：从 `env` 取 onset（上升沿 + 迟滞），奖励"onset 后 `onset_lag ± 80ms` 窗口内
  出现 `|速度|` 局部峰"的比例，peaked 到实测 `onset_hit`。**原来的 `couple` 保留不动**，用于对照。

Gate：静音段头部幅度 ≤ 语音段的 0.6×；onset 命中率显著高于 base（报数值 + base 对照）。

---

## 7 · P3 · `blink` / `gaze` 重写

- **`blink`**：删掉 `smoothmax` 峰值写法，改成**事件检测**（`b` 上穿 0.35 + 0.2 s refractory）
  → 得到 rate / duration / IBI，三项各一个 peaked 高斯（target 来自 C1）；
  **必须保留** `mean(b) ≤ 0.12` 的硬约束（`INDUSTRY_ROUND.md` §4 的教训：目标一高眼睛就半闭）。
- **`gaze`**：`tanh(g_std/0.1)` 换成「注视速度 L1/L2 稀疏度」+「fixation 停留占比」两个 peaked 项；
  保留原来的 mean-offset 惩罚。

Gate：渲染视频上 EAR 事件数 **12–22 次/min**（人类 10–20，实测 median 15）。
检测沿用 `scripts/measure_eye_openness.py`，但**先在小样本上验证计数法可信**——
`INDUSTRY_ROUND.md` §7.6 已说明旧计数法对渲染噪声敏感、不可信。
**不可信就换方法，或如实写未达标，不要拿不可信的数当结论。**

---

## 8 · P4 · 渲染侧（不训练，观感立刻提升）

新增 `scripts/render_torso.py`（或给 `render_industry_demo.py` 加 flag）：

1. **半身**：pasteback 时对身体区域做由 head pose 驱动的轻微仿射 warp
   （肩随 roll/pitch 的 0.3–0.5 倍幅度 + 0.2 Hz 呼吸起伏），并优先选用**带肩的半身肖像**。
   目标：头与身体不再脱节。
2. **画质**：pasteback **前**对动画脸做 CodeFormer / GFPGAN 修复；pasteback **后** Real-ESRGAN 2×；
   编码改 crf 18。眼部区域单独再超分一次。
3. 每一步都留一份"关掉该步"的基线视频，便于判断到底是哪一步在起作用。

Gate：输出带肩，且并排看不出头身接缝。

---

## 9 · P5 · 大量 demo

- 每行业 **≥10 组**（不同脸 × 不同音色）：
  `python scripts/pick_portraits.py --scan 400 --per-industry 10 --min-ear 0.30 --min-px 800`
- 保留同脸跨行业四联对照（最有说服力的一组）
- 新增两种对照视图：**0.25× 慢放**（幅度/时序差异慢放才看得清）、**眼部裁切放大**
- 总览 contact sheet + 每行业一张并排图
- mp4 走 `backup/r3-07`；`main` 只放 GIF / json / md / 图

---

## 10 · P6 · 结果

统一写 `docs/RESULTS_R3.md`：每个 gate 的关键数值、PASS / FAIL、base vs 训完的对照表。
同步更新 `README.md`、`STATUS.md`、`/PROJECT_STATUS.md` 的 07 一节。
措辞必须与结果一致（没改善就写没改善）。push 前按 `AGENTS.md` 做禁用词 grep。

---

## 11 · 时间预算与停止线

| 阶段 | 时间 |
|---|---|
| T0 | 0.5 h |
| C1（CPU，并行） | 1.5 h |
| P1 | 1.0 h |
| P1e | 1.0 h |
| P2 | 1.0 h |
| P3 | 1.0 h |
| P4 | 1.5 h |
| P5 | 1.5 h |
| P6 | 0.7 h |
| **合计** | **约 9.5 h** |

时间不够时的砍顺序（从后往前）：P5 的组数 → P4 的超分 → P3 → P2。
**不砍 C1 与 P1 / P1e** —— C1 是后面所有目标带的来源，P1 / P1e 是本轮的核心判据。

**窗口结束前 60 分钟停止一切新任务，转入全量备份（§0c）。**
