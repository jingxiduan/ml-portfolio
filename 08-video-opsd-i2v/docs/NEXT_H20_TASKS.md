# 08 · 下一轮 H20 任务书（先确认 LoRA 真的在动，再长训，再做对照评测）

> 写给下一个 H20 session 的 CodeBuddy。先读 `/AGENTS.md`、`/PROJECT_STATUS.md`、`STATUS.md`，再读本文件。
> 本文件只定义"做什么、怎么判定成败、什么时候停"。

---

## 0 · 背景（只保留与任务有关的事实）

上一轮 3 组消融（各 500 步，160 条训练 prompt，约 1.43 GPU-h / 组）：

| 配置 | 训练末期 reward | R_naturalness | 说明 |
|---|---|---|---|
| V1（ReLU+clamp） | +2.16 | 0.000 | naturalness 梯度全程为 0 |
| V2（sigmoid） | -0.50 | -1.47 → -0.79 | 梯度存活，惩罚项确实在下降 |
| NoNat | +2.36 | - | 无 naturalness 约束 |

n=10 留出 prompt 上的评测（`outputs/eval/metrics_table.md`）：

| 配置 | Appeal | Naturalness | Dynamic |
|---|---|---|---|
| Base | 0.7295 | 0.9515 | 0.5075 |
| V1 | 0.7294 | 0.9515 | 0.5074 |
| V2 | 0.7291 | 0.9513 | 0.5069 |
| NoNat | 0.7296 | 0.9512 | 0.5079 |

**四个配置在小数点后第四位才有差异。** 目前能支持的结论只有一条：
"sigmoid 修复了 naturalness 的梯度死区"（训练曲线可证）。
**不能说 OPSD 在这个设置下提升了视频质量。**

### 为什么评测上几乎没动（两个待验证的假设）

1. **学习率偏低**：`configs/tier2_v2.yaml` 里 `lr: 1.0e-5`，LoRA rank 16。LoRA 常见学习率高一个量级，
   500 步后 LoRA 权重 L2 范数约 27，输出几乎没变。
2. **步数偏少**：上一轮报告建议 2000 步以上。

在没确认"LoRA 真的改变了输出"之前，直接把步数拉到 3000 可能只是白烧 17 GPU-h。
所以先做一个便宜的探针（T1），再决定长训的学习率。

另外，当前评测指标（Appeal / Naturalness / Dynamic）和训练奖励属于同一族，
本轮要补一个不在奖励里的独立量。

---

## 1 · 任务总览与 gate

| 阶段 | 内容 | 预计时间 | 通过条件（否则停下写结论） |
|---|---|---|---|
| T0 | 环境与权重 | 0.75 h | 能用现有 `lora_step00500` 渲染 1 条视频 |
| T1 | LoRA 是否有效 + lr 探针 | 1 h | 找到一个 lr，使 200 步后 LoRA 输出相对 base 有可测差异且不崩 |
| T2 | 长训：V2 与 NoNat 各 2000 步 | 约 5.7 h（两卡并行） | 1000 步处的中途检查通过（§4） |
| T3 | 评测：n=40，多个 checkpoint | 约 1 h | |
| T4 | 独立评测 + A/B 盲评表 | 0.5 h | |
| T5 | 结果文档、push | 0.5 h | |

2000 步是默认值。如果 T1 之后窗口剩余时间足够（> 11 h），可延长到 3000 步；不够就保持 2000。

---

## 2 · T0 · 环境

`docs/H20_PLAYBOOK.md` 里有 SSH 设置与 `scripts/h20_setup.sh`（幂等，下载 DynamiCrafter-512，约 30-45 min）。

```bash
git clone git@github.com:jingxiduan/ml-portfolio.git && cd ml-portfolio/08-video-opsd-i2v
bash scripts/h20_setup.sh
source .venv/bin/activate && export PYTHONPATH="$PWD"
nvidia-smi
```

mp4 被 gitignore，上一台机器到期时已丢失，需要重渲。
评测首帧要有 `seed_00` 子目录的软链（renderer 要求）：

```bash
mkdir -p data/first_frames_eval/seed_00
for f in data/first_frames_eval/*.png; do ln -sf "$(realpath $f)" "data/first_frames_eval/seed_00/$(basename $f)"; done
```

冒烟：用 `outputs/opsd_tier2_v2/lora_step00500` 渲染 1 条评测视频，确认链路通。

---

## 3 · T1 · LoRA 是否有效 + 学习率探针

### 3.1 现有 checkpoint 的差异量化（10 分钟，不训练）

写 `scripts/lora_effect_probe.py`：对同一条 prompt、同一 seed、同一首帧，
分别用 base 和 `lora_step00500`（V2）渲染，计算：
- 逐帧像素 L1 / LPIPS 差异的均值；
- LoRA 的 `B@A` 增量范数相对原权重范数的比值。

如果像素差异接近 0，说明 500 步 + 1e-5 基本没有改变模型，后面的所有评测差异都是噪声。

### 3.2 lr 探针（约 0.6 h，两卡并行）

用 V2 配置，只改学习率，训 200 步：

| 卡 | lr | 输出目录 |
|---|---|---|
| GPU0 | 5e-5 | `outputs/probe_lr5e-5` |
| GPU1 | 1e-4 | `outputs/probe_lr1e-4` |

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.train_entry --method opsd_i2v --config configs/probe_lr5e-5.yaml &
CUDA_VISIBLE_DEVICES=1 python -m src.train_entry --method opsd_i2v --config configs/probe_lr1e-4.yaml &
wait
```

（`configs/probe_*.yaml` 从 `tier2_v2.yaml` 复制，仅改 `lr`、`num_steps: 200`、`out_dir`、`ckpt_every: 50`。）

每个探针要看：
- `train_metrics.jsonl` 中 `reward`、`R_naturalness_penalty`、`grad_norm` 是否正常，没有发散；
- 对 3.1 同一条 prompt 重新做差异量化；
- 8 条 eval prompt 的视频是否出现崩坏（全黑、闪烁、过饱和）。

### 3.3 选 lr 的规则（先写下来）

- 选"像素差异明显大于 1e-5 基线，且没有崩坏、`grad_norm` 稳定"的最大 lr。
- 两个都崩 → 选更小的那个，并在文档里写明；
- 两个都不崩且差异接近 → 选较小的。
- 如果全部都没有可测差异（即使 1e-4），停下，把现象写清楚，不进 T2。

---

## 4 · T2 · 长训（V2 与 NoNat 配对）

两组**只在 naturalness 项上不同**，其余完全一致：

| 组 | 配置来源 | 改动 |
|---|---|---|
| V2 | `configs/tier2_v2.yaml` | `lr` 用 T1 选定值，`num_steps: 2000`，`out_dir: outputs/opsd_long_v2` |
| NoNat | `configs/tier2_ablation_no_nat.yaml` | 同样的 `lr`、`num_steps`，`out_dir: outputs/opsd_long_no_nat` |

- 一卡一组并行，`nohup` 后台跑，日志落盘。500 步约 1.43 GPU-h，2000 步约 5.7 h。
- `ckpt_every: 50` 保持不变，评测只用 500 / 1000 / 2000 三个点。
- **不要同时改 `cap_tau`、reward 权重等其他东西**，否则无法归因。
  `dynamic_degree` 在 50 步内就饱和（`cap_tau: 0.6` 偏低）是已知问题，列为后续项，本轮不动。

### 中途检查（1000 步处，写进日志）

- `reward` 与 `R_naturalness_penalty` 是否仍有趋势（V2）；
- `grad_norm` 没有塌缩到接近 0；
- 用 1000 步 checkpoint 渲染 4 条 eval 视频，目视没有崩坏。

任一项不过 → 停掉该组，保留已有 checkpoint 和日志，写结论，不要硬跑到 2000 步。

---

## 5 · T3 · 评测（n=40）

`scripts/eval_full.py` 的 `CONFIGS` 里补上新配置（base，V2@500/1000/2000，NoNat@500/1000/2000），
然后：

```bash
python scripts/eval_full.py --render-only    # 约 11 min / 配置，7 个配置，两卡拆开跑约 40 min
python scripts/eval_full.py --metrics-only
```

产出：
- `outputs/eval/metrics_long.json`、`metrics_long.md`：n=40，每个配置的 Appeal / Naturalness / Dynamic，
  同时给出**配对差值的 bootstrap 95% 置信区间**（V2 − NoNat、V2 − Base），不要只给均值。
- 指标随步数的曲线：`outputs/figures_comparison/metrics_vs_steps.png`。

---

## 6 · T4 · 独立评测

训练奖励里已经有 subject_consistency、motion_smoothness、dynamic_degree、naturalness，
评测不能只用这些。

1. **一个不在奖励里的自动指标**：优先看 `configs/vbench_rewards.yaml`，
   选 VBench 里没被奖励用到的维度（例如 aesthetic / imaging quality 一类）。
   如果安装与跑通超过 30 分钟，直接放弃，只保留第 2 项，并在文档里写明。
2. **A/B 盲评表**：从 40 条 eval 中取 20 条，`base / V2@终点 / NoNat@终点` 随机两两配对并随机左右，
   生成 `outputs/human_eval/sheet.csv`（只有编号与评分列）和单独保存的 `key.csv`。
   不需要在 H20 上打分，生成好即可，后面由人来评。

### 成功标准（先写下来，结果出来后不改）

- V2 相对 NoNat：Naturalness 的配对差值 95% CI 不包含 0 且为正，同时 Dynamic 的降幅不超过 0.05。
- V2 相对 Base：Appeal 不低于 Base（差值 CI 下界不小于 -0.01）。
- 达不到就如实写。可以接受的结论有两种：
  "在该设置下 OPSD 仍无可测提升，原因是……"，或"提升集中在 naturalness，未扩展到 Appeal"。
  不可以写没有数据支持的"显著提升"。

---

## 7 · 时间预算与停止线

| 阶段 | 时间 |
|---|---|
| T0 | 0.75 h |
| T1 | 1.0 h |
| T2 | 5.7 h |
| T3 | 1.0 h |
| T4 | 0.5 h |
| T5 | 0.5 h |
| 合计 | 约 9.5 h |

07 的任务（`07-talking-head-motion-opsd/docs/NEXT_H20_TASKS.md`）会和 T2 共用 GPU，
07 的 motion 训练很小，先看 `nvidia-smi` 剩余显存再决定放哪张卡。
窗口剩余 < 11 h 时不要延长到 3000 步。窗口剩余时间不足以完成 T3 时，宁可缩短 T2，也要留出评测与 push 的时间。

---

## 8 · 提交规则

- 每个阶段结束就 commit + push（T1 结束、T2 的 1000 步与终点、T3 结束），不要攒到最后。
- 用 SSH key 推送，**不要把任何 token 写进文件、remote URL 或命令历史**。
- mp4 保持 gitignore；commit 日志、json、md、曲线图、GIF。LoRA checkpoint（约 13MB / 个）只提交 500 / 1000 / 2000 三个点。
- push 前按 `/AGENTS.md` 的 house rules 做禁用词检查；文案用英文、无 emoji 的现有风格保持不变。
- 结果统一写 `docs/RESULTS_LONG.md`，数字照实写，包括失败的；
  README 的措辞要和结果一致。
- 完成后更新 `STATUS.md` 与 `/PROJECT_STATUS.md` 的 08 一节。
