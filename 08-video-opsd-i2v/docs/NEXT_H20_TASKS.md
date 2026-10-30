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

## 0b · 硬性约束（先读这里，优先级高于后面所有章节）

### A. 已核对的日志事实（T1 之前先自己复核一遍）

对 `outputs/opsd_tier2*/train_metrics.jsonl` 的 1500 步统计：

| 现象 | 数据 | 含义 |
|---|---|---|
| `grad_norm` 恒为 1.0 | 三组 500 步全部 | 这是归一化后的**奖励梯度**，不是参数梯度，**不能用来监控训练** |
| 奖励几乎没动 | NoNat：前 50 步均值 2.21，后 50 步 2.18；V2：-0.28 到 -0.18 | 模型基本没在学 |
| 对比损失约等于 0 | `loss_plus` 与 `loss_minus` 都约 3.8e-4，差值约 0 | 正负目标拟合得一样好，没有梯度信号 |
| `R_motion_smoothness` 恒为 0.36788 | 500 步全部 | 代码里是 `exp(-jerk / jerk.detach().mean())`，batch=1 时分母就是自己，数值恒为 e^-1。**数值恒定是设计使然，梯度不一定为 0**，但这一项不能当指标看，也不能据此判断它有没有起作用 |
| dynamic_degree 没饱和 | 0.86 到 0.83（还略降） | 上一份报告说"50 步内饱和到 1.0"，与日志不符，以日志为准 |
| 学习率曲线 | 起点 2e-7，终点 0（warmup 50 步 + 余弦衰减） | 平均有效学习率只有标称值的约一半 |

### A2. 没训动的根因线索（代码核对，T1 要逐条实测，不是已证结论）

1. **目标偏移量太小。** `y0_plus = y0 + c_adv * sigma_t * g`，其中 `g` 被 L2 归一化到整个张量范数为 1。
   日志里 `sigma_q` 平均 480，对应 `sigma_t ≈ 0.80`，所以偏移范数约 `5 × 0.80 = 4`。
   潜变量有 4×16×40×64 = 163,840 个元素，`‖y0‖` 约 405 × rms，所以**偏移只有 y0 的约 1%**。
   上游 DiffusionOPSD 的目标半径 ρ 是 0.10（相对信任域），按这个口径我们小了约一个数量级。
   偏移这么小，正负目标几乎重合，所以 `loss_plus ≈ loss_minus`，这与日志吻合。
2. **学习率偏低**：`lr=1e-5`，且余弦衰减到 0。07 同样的方法用 3e-4 就能训动，差 30 倍。
3. **batch=1、inner_fit_steps=1**：每步只有一个 prompt 的奖励梯度，方向噪声大；
   AdamW 会把很小的梯度归一化放大，随机方向的更新累积起来更像随机游走。上游每次更新是 48 个 prompt 组。
4. **奖励里真正有信号的项很少**：`subject_consistency` 在 0.984 附近、`motion_smoothness` 值恒定，
   `dynamic_degree` 在下降，V2 里剩下 `naturalness_penalty` 在起作用。

以上四条里，1 和 2 是本轮探针的直接对象；3 用 batch_size 做小范围试验；4 本轮只记录，不改奖励。

所以这一轮的风险有两种：**训崩**，和**没训动**。上一轮实际发生的是后者。两种都算失败，都要停下写结论。

### B. 禁止事项

1. **不修改已有文件**：`src/rewards/*`、`configs/tier2*.yaml`、`outputs/opsd_tier2*`、`data/prompts/*`、评测首帧。只新增文件。
   - 唯一允许的对已有文件的改动：在 `src/opsd_trainer_i2v.py` 的 metrics 里**追加**三个日志字段
     `param_grad_norm`（`clip_grad_norm_` 的返回值，目前被丢弃了）、`shift_ratio`、`lora_B_norm`，不改变任何训练行为。
2. **不改成功标准、不改守护脚本阈值**，不为了让 gate 通过而放宽。
3. **不加新奖励项**，不改奖励权重、`cap_tau`、`sat_threshold` 等。本轮只允许动 `lr`、`c_adv`、`batch_size`、`num_steps`（上限见 §C）。
4. **gate 没过不进入下一阶段。**
5. 不 `git push --force`，不 `git reset --hard`，不删除 `outputs/` 下任何已有内容。
6. 推送与备份规则见 §0c，**每个训练跑完（或被守护脚本停掉）立刻备份**，不要攒到最后。
7. 不重新生成评测首帧与 prompts，不更换 seed（固定 0）。

### C. 参数硬上限

| 项 | 上限 / 固定值 |
|---|---|
| `lr` | **≤ 1e-4**（探针最大值，不得更大）；长训的 lr 必须取自已通过探针的值 |
| `num_steps` | ≤ 2000 |
| `grad_clip` | 1.0 不变 |
| LoRA | rank 16 / alpha 32 不变 |
| `c_adv` | **≤ 40**（目标偏移约为 y0 的 8%，接近上游 ρ=0.10，再大会越过上游的信任域）；默认探针值见 §3.2 |
| `batch_size` | ≤ 4（受显存限制，先看 `nvidia-smi`） |
| `inner_fit_steps` | 1 不变 |
| seed | 0 |
| 重试次数 | 每组最多 1 次（`lr` 和 `c_adv` 都改为原来的 1/3），仍触发则停 |

### D. 训练必须通过守护脚本启动

先自检：`python ../tools/train_guard.py --selftest`，必须输出 `selftest OK`。

```bash
OUT=outputs/opsd_long_v2
mkdir -p $OUT
nohup python -m src.train_entry --method opsd_i2v --config configs/long_v2.yaml > $OUT/train.log 2>&1 &
TRAIN_PID=$!
nohup python ../tools/train_guard.py --kind i2v --log $OUT/train_metrics.jsonl --pid $TRAIN_PID > $OUT/guard.log 2>&1 &
```

守护脚本每 20 秒读一次日志，触发时写 `$OUT/GUARD_STOP.json` 并给训练进程发 SIGTERM。
这些阈值已在现有 3 份日志上回放过，**没有误报**；对 NaN、主体漂移、奖励塌陷等合成场景能正确触发。

| 规则 | 触发条件（50 步窗口均值） | 依据 |
|---|---|---|
| `nonfinite` | 任一指标出现 NaN / inf | |
| `loss_blowup` | `loss_plus` > 前 50 步均值的 10 倍 | 基线约 3.8e-4，正常波动在 2 倍以内 |
| `subject_drift` | `R_subject_consistency` < 0.95 | 实测 0.984，单步标准差 0.014 |
| `reward_collapse` | `reward` 比前 50 步均值低 1.0 以上 | 窗口均值噪声约 0.07，现有三组前后变化都小于 0.1 |
| `naturalness_worse` | `R_naturalness_penalty` 比前 50 步低 0.5 以上 | V2 实测是上升的 |
| `stalled` | 1200 秒没有新日志 | |

### E. 触发后怎么处理

1. **不要手动继续，也不要原样重启。**
2. 保留 `outputs/<run>/` 与 `GUARD_STOP.json`，**立刻**按 §0c 备份（日志、曲线、已有 checkpoint），再做后续处理。
3. 用最近一个 checkpoint 渲染 4 条 eval 视频，目视确认是否崩坏。
4. 允许重试 1 次：`lr` 和 `c_adv` 都改为原来的 1/3，`out_dir` 加后缀 `_retry`。再次触发就停，不要第三次。
5. V2 与 NoNat 是配对实验：一组被停，配对对比只用**两组都跑到的最大共同步数**。
6. 在 `docs/RESULTS_LONG.md` 里写明：哪条规则触发、触发步数、日志数值。

### F. "没训动"也是失败

判据见 §3.3。T1 的探针都不满足"输出变了"和"奖励在动"，就视为没训动，**停下，不进入 T2**。

此时的产出是诊断（为什么 `loss_plus - loss_minus` 约等于 0，对照 A2 的四条线索逐条给出实测数），
并向用户汇报，而不是把步数加大。

### G. Gate 记录

每个 gate 在 `docs/GATE_LOG.md` 追加一行：时间 | gate | 关键数值 | PASS 或 FAIL。人工汇报时直接引用这个文件。

---

## 0c · 推送、备份与数据来源

### 推送与备份：到期会回收，没推出去的就没了

**规则：每个任务一结束就立刻备份，再开始下一步。** 这里的"结束"包括：正常跑完、被守护脚本停掉、手动停掉。

用 `tools/backup_push.sh`（已在临时仓库里测过：小文件上 main，大文件与媒体打包到独立分支，还原可用）：

| 内容 | 命令 | 去向 |
|---|---|---|
| 日志 jsonl、metrics、曲线图、GIF、配置、LoRA（单个 ≤ 90MB，当前约 13MB） | `tools/backup_push.sh small "<msg>" outputs/<run> ...` | `main` |
| 评测 mp4、抽取出来的数据子集、任何 > 90MB 的文件 | `tools/backup_push.sh big r2-08 outputs/eval data/public_subset` | 独立分支 `backup/r2-08` |
| 看现在有多少东西没推出去 | `tools/backup_push.sh status` | |

- **到期时间 = 窗口开始时间 + 窗口时长**（用户给了就用用户的，没给按 14h）。到期前 90 分钟：不再启动新训练，做一次全量备份。
- T0 先用一个很小的目录跑一次 `big`，测出上传速度，据此估算全量备份要多久。
- `backup/*` 分支总量不超过 1.5GB。超了先压缩视频（480p，crf 28），仍超就只保留评测集与最终 checkpoint，并在结果文档里写明舍弃了什么。
- 每次备份后，把提交 hash 或分支名记进 `docs/GATE_LOG.md`。
- 推送使用这台机器上已配置好的方式（SSH），不要把任何凭据写进文件或命令。不要 `--force`。

### 数据来源：必须是公开数据集

上一轮的训练数据是 GPT 写的 160 条 prompt 加 SDXL-Turbo 合成首帧，规模小而且是合成的。本轮改用公开视频集：

- **首选 OpenVid-1M**（Hugging Face `nkp37/OpenVid-1M`，带 caption，许可证以数据集页面为准）。
  **只下 CSV 和一个视频分片，不要全量下载**；先 `df -h`，分片很大，磁盘不够就换备选。
  取每个视频的首帧作为条件图、caption 作为 prompt。
- 规模：训练 ≥ 1000 条，评测 100 条，按视频 id 划分，**训练与评测不重叠**。
- 备选：任何有明确许可证、带文字描述的公开视频集。
- 旧的 40 条合成评测集保留为 `legacy eval`，新旧两套都评，方便和上一轮对照。
- 必须产出 `data/DATA_SOURCES.md`：名称、URL、许可证、下载日期、抽取方法、条数、id 列表文件的路径。
  id 列表与下载脚本进 `main`（公开集可以重下，但抽样必须可复现）；抽出来的子集走 `big` 备份。
- T0 先验证联网：`curl -sI https://huggingface.co | head -1`。**不通就汇报用户，不要退回合成数据并称其为公开数据。**
- 数据准备在 CPU 上做，与 T1 探针并行，不占 GPU。T1 探针沿用现有数据，因为它测的是训练机制本身。

---

## 1 · 任务总览与 gate

| 阶段 | 内容 | 预计时间 | 通过条件（否则停下写结论） |
|---|---|---|---|
| T0 | 环境、权重、联网检查、公开数据下载启动（CPU 后台） | 0.75 h | 能用现有 `lora_step00500` 渲染 1 条视频；`big` 备份测通 |
| T1 | 探针：lr 与目标偏移（§3） | 1.5 h | 至少一个探针满足 §3.3 四条判据，否则停 |
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

### 3.2 探针（每个 200 步，两卡并行，分两轮，共约 1.2 h）

用 V2 配置（`tier2_v2.yaml`）复制出 `configs/probe_*.yaml`，**只改下表的量**，另改 `num_steps: 200`、`out_dir`、`ckpt_every: 50`。
先在 `src/opsd_trainer_i2v.py` 的 metrics 里追加三个纯日志字段（不改训练行为）：
`param_grad_norm`（`clip_grad_norm_` 的返回值）、`shift_ratio`（`‖c_adv·σ_t·g‖ / ‖y0‖`）、`lora_B_norm`。

| 轮 | 探针 | lr | c_adv | 要回答的问题 |
|---|---|---|---|---|
| 1 | P1 | 1e-4 | 5（原值） | 只提高学习率能不能训动 |
| 1 | P2 | 1e-5（原值） | 40 | 只放大目标偏移能不能训动 |
| 2 | P3 | 1e-4 | 40 | 两者叠加 |
| 2 | P4（可选） | P3 的值 | P3 的值 | `batch_size: 4`，100 步，看噪声是否是瓶颈 |

第 2 轮在第 1 轮结果出来后再启动，每轮都经守护脚本启动。

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.train_entry --method opsd_i2v --config configs/probe_p1.yaml &
CUDA_VISIBLE_DEVICES=1 python -m src.train_entry --method opsd_i2v --config configs/probe_p2.yaml &
wait
```

### 3.3 "动了"的判据（先写下来，结果出来后不改）

一个探针要同时满足四条，才算"训动且没崩"：

1. **输出变了**：同一 prompt、同一 seed、同一首帧，8 条 eval prompt 上与 base 的逐帧像素 L1 均值，
   **至少是旧 `lora_step00500`（lr=1e-5）的 3 倍**。
2. **奖励在动**：`reward` 的最后 50 步窗口均值比前 50 步**高 0.3 以上**（窗口噪声约 0.07）。
3. **没崩**：守护脚本未触发，8 条视频目视无全黑、闪烁、过饱和，`R_subject_consistency` ≥ 0.95。
4. **偏移量符合预期**：日志里 `shift_ratio` 的均值在 `c_adv=40` 时应落在 5% 到 12%，`c_adv=5` 时约 1%。
   不符合说明上面 A2-1 的估算有误，要先查原因再继续。

### 3.4 选参规则

- 在满足四条的探针里，选 `lr × c_adv` **最小**的那个（最保守）。
- 第 1 轮有探针满足，第 2 轮可以省略。
- 第 2 轮仍然没有探针满足 → **停下，不进 T2**。产出是：四个探针的 `shift_ratio`、`param_grad_norm`、
  `lora_B_norm`、像素差异的完整表，以及对根因的判断，然后向用户汇报。
- 全部都崩 → 同样停下，附上守护脚本的触发记录。

---

## 4 · T2 · 长训（V2 与 NoNat 配对）

两组**只在 naturalness 项上不同**，其余完全一致：

| 组 | 配置来源 | 改动 |
|---|---|---|
| V2 | `configs/tier2_v2.yaml` | `lr`、`c_adv` 用 T1 选定值，`num_steps: 2000`，`out_dir: outputs/opsd_long_v2`；训练集换成公开数据子集（§0c） |
| NoNat | `configs/tier2_ablation_no_nat.yaml` | 同样的 `lr`、`c_adv`、`num_steps`、数据，`out_dir: outputs/opsd_long_no_nat` |

- 一卡一组并行，`nohup` 后台跑，日志落盘。500 步约 1.43 GPU-h，2000 步约 5.7 h。
- 配置文件是**新增**的 `configs/long_v2.yaml` 与 `configs/long_no_nat.yaml`（分别从 `tier2_v2.yaml`、`tier2_ablation_no_nat.yaml` 复制，只改上表的几项），原文件不动。
- 必须按 §0b-D 通过守护脚本启动。
- `ckpt_every: 50` 保持不变，评测只用 500 / 1000 / 2000 三个点。
- **不要同时改 `cap_tau`、reward 权重等其他东西**，否则无法归因。
  `dynamic_degree` 在 50 步内就饱和（`cap_tau: 0.6` 偏低）是已知问题，列为后续项，本轮不动。

### 中途检查（1000 步处，写进日志）

- `reward` 与 `R_naturalness_penalty` 是否仍有趋势（V2）；
- 守护脚本全程未触发；
- 50 步窗口均值的 `reward` 相对前 50 步有变化（不是持平）；
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
| T1 | 1.5 h |
| T2 | 5.7 h |
| T3 | 1.0 h |
| T4 | 0.5 h |
| T5 | 0.5 h |
| 合计 | 约 10 h |

07 的任务（`07-talking-head-motion-opsd/docs/NEXT_H20_TASKS.md`）会和 T2 共用 GPU，
07 的 motion 训练很小，先看 `nvidia-smi` 剩余显存再决定放哪张卡。
窗口剩余 < 11 h 时不要延长到 3000 步。窗口剩余时间不足以完成 T3 时，宁可缩短 T2，也要留出评测与备份的时间。
**到期前 90 分钟不再启动新训练，转入全量备份。**

---

## 8 · 提交规则

- **训完就推，不要等到阶段末。** 备份方式与分支见 §0c，每个训练结束后第一件事就是 `tools/backup_push.sh`，并在 `docs/GATE_LOG.md` 里记下提交 hash。
- 推送使用这台机器上已配置的方式（SSH），不要把凭据写进任何文件或命令。
- mp4 不进 `main`，走 `backup/*` 分支。LoRA checkpoint（约 13MB / 个）进 `main`，只提交 500 / 1000 / 2000 三个点。
- push 到 `main` 前按 `/AGENTS.md` 的 house rules 做禁用词检查；文案用英文、无 emoji 的现有风格保持不变。
- 结果统一写 `docs/RESULTS_LONG.md`，数字照实写，包括失败的；
  README 的措辞要和结果一致。
- 完成后更新 `STATUS.md` 与 `/PROJECT_STATUS.md` 的 08 一节。
