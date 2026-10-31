# 07 · 跨行业数字人：金融 / 房地产 / 短视频

> 本轮目标：把 OPSD 数字人做成**三个行业**的可用形象，且每个行业都必须自然——
> 点头、倾听、眨眼、微笑是真人行为，不是夸张的"AI 感"。
> 方法仍然是 [DiffusionOPSD](https://github.com/worldbench/DiffusionOPSD) 的
> On-Policy Self-Distillation，只是在 **motion space**（73 维系数）上，
> 基线模型是 JoyVASA `motion_generator_hubert_chinese`（冻结），
> 渲染用 LivePortrait（1280 px + pasteback）。

---

## 1. 一句话结论

**三个行业的 persona 都落在了"真人实测运动带"内（幅度 0.84–0.90× 目标，无一过冲），
而 v2/v3 那种 2.4–2.7× 的恐怖谷过冲没有再出现。** 同一张脸在三种行业 LoRA 下
会呈现三种不同的行为（稳重 / 温和 / 活泼），这就是"行业感"的来源。

**中途发现并修掉了一个真问题**：第一版渲出来的人物**眼睛半闭**（base 反而最好看）。
根因有两条，都用测量确认（见 §5）：blink 目标被我抬到 0.18–0.21（v4 验证值是 0.15），
以及部分源肖像本身眼睛就小。修完后 persona 的眼睛开合度 **EAR 0.102 → 0.348**，
与 base 的 0.343 持平——**行业动作保留，眼睛睁开了**。

## 2. 三个行业的 reward model 是怎么定的（不是拍脑袋）

v4 的核心经验：**amplitude 项必须是 peaked 的（高斯），中心值必须是实测出来的**。
本轮把这个原则推广到行业维度——**每个行业一个实测运动带**：

| 行业 | 视频代理（公开、真实人声） | 语义 | n |
|---|---|---|---|
| finance | HDTF 新闻主播片段 | 正式、克制、可信 | 25 |
| realestate | CelebV-HQ 中等幅度片段 | 讲解、亲和 | 16 |
| shortvideo | CelebV-HQ 高幅度片段 | 活泼、抓眼球 | 16 |

说明：公开数据集没有"行业"标签，这是**显式声明的代理划分**——CelebV-HQ 是真实
YouTube 说话视频，按**实测运动幅度**切出温和/活泼两组。所有数字来自
`scripts/calibrate_industry_bands.py` 的实测（MediaPipe FaceLandmarker + solvePnP，
Butterworth 滤波与 reward 内部用的过滤器**完全一致**，所以量纲可直接对齐）。

### 实测结果（`outputs/industry_bands.json`）

| 行业 | 低频幅度 (0.3–3 Hz, deg) | 高频抖动 (6–12.5 Hz, deg) | 眨眼 (次/min) |
|---|---|---|---|
| finance | **1.384** [IQR 1.03–1.50] | **0.094** | 18.1 |
| realestate | **1.257** [1.03–1.50] | **0.118** | 20.3 |
| shortvideo | **3.039** [2.70–3.49] | **0.349** | 22.6 |

行业差异是真实存在的：短视频口播的头部幅度约为金融播报的 **2.2×**，抖动 **3.7×**。

### 质量门控（重要）

原始测量里 CelebV-HQ 有大量 solvePnP 噪声（一个 clip 报出 37° 的"头部运动"）。
不加门控会把噪声当成"行业特性"。加了四条硬门控后剔除了 71% 的 CelebV-HQ clip：

- 近正面（mean |yaw|、|pitch| ≤ 25°）
- 低频幅度 ∈ [0.2, 8.0] deg（真人说话的合理范围）
- 高频抖动 ≤ 1.0 deg
- 眨眼 ≤ 60 次/min

HDTF 保留 25/40，CelebV-HQ 保留 18/90（扩采到 390 个后复测为 16/…，见 `industry_bands_full.json`）。

### 从实测带到 reward 参数（唯一入口）

| 参数 | 映射 | finance | realestate | shortvideo |
|---|---|---|---|---|
| `--amp-target-deg` | 实测低频幅度 median | 1.3838 | 1.2570 | 2.9299 |
| `--amp-tol-deg` | 固定 1.0（v4 已验证值） | 1.0 | 1.0 | 1.0 |
| `--jitter-cap-deg` | 2 × 实测高频 median | 0.1883 | 0.2358 | 0.5000 |
| `--blink-target-mean` | 实测眨眼 / 100，clamp 到 [0.10, 0.22] | 0.1815 | 0.2030 | 0.2091 |

**除这四个数之外，三个行业的 reward 权重、公式、LoRA 配置、lr、rounds、seed 与 v4 完全一致。**
这样任何差异都可归因到"目标带"本身，而不是超参差异（沿用任务书 §0b 的可归因原则）。

## 3. 基线与改进（回答"基线咋样、加了方法有什么改进"）

### 基线 = 未做 OPSD 的冻结 JoyVASA

10 条公开语音上的复测（`outputs/naturalness_eval_r3.json`，与人类带对比）：

| 变体 | 低频幅度 (deg) | vs 真人 1.878° | 高频抖动 (deg) |
|---|---|---|---|
| 真人（专业口播） | 1.878 | 1.00× | 0.107 |
| **base（未训练）** | 0.879 | **0.47×（太僵）** | 0.119 |
| OPSD v2 | 5.166 | 2.75×（过冲→恐怖谷） | 0.193 |
| OPSD v3 | 4.460 | 2.38×（过冲） | 0.173 |
| **OPSD v4（peaked reward）** | **1.659** | **0.88×** | **0.116** |

**结论：base 不是"自然但平淡"，而是"僵"（0.47× 真人）；naive 的 reward hacking 会
把它推到 2.4–2.7×（诡异）；把 amplitude 项改成实测带中心的 peaked 高斯后，
落到 0.88×——这就是把 DiffusionOPSD 迁到 motion space 后真正学到的东西。**

### 三个行业 persona 在**独立说话人**上的表现（`outputs/industry_eval.json`）

12 条 AISHELL-1 eval 语音（15 个评测说话人与 60 个训练说话人**完全不重叠**）：

| 变体 | 幅度 (deg) | vs 各自目标 | vs base | 抖动 (deg) | vs 目标 |
|---|---|---|---|---|---|
| base | 0.890 | — | 1.00× | 0.1187 | — |
| finance | 1.162 | **0.84×** | **1.31×** | 0.1196 | 1.27× |
| realestate | 1.134 | **0.90×** | **1.27×** | 0.1124 | 0.95× |
| shortvideo | 2.699 | **0.89×** | **3.03×** | 0.1561 | 0.45× |

读法：

- **没有 overshoot**：三个都在目标的 0.84–0.90×，方向是"略保守"，不是"夸张"。
  v2/v3 的 2.4–2.7× 过冲没有复发。
- **三个行业真正分开了**：shortvideo 的幅度是 finance/realestate 的 2.3×，
  与实测带一致（3.04 vs 1.38/1.26）。
- **抖动全部贴近人类带**（0.11–0.16 deg，人类 0.09–0.35）。
- finance 与 realestate 的幅度接近（1.16 vs 1.13）——**这是诚实的结论**：
  真实视频里"新闻播报"和"温和讲解"的头部运动本来就几乎一样（1.38 vs 1.26 deg），
  不应该为了"看起来有差异"而编造一个更大的差距。

### 训练曲线

三行业 reward 曲线见 `outputs/figures/fig_industry_training.png`；
行业带 vs persona 落点见 `outputs/figures/fig_industry_bands.png`；
base/v2/v3/v4 对比见 `outputs/figures/fig_main_comparison.png`；
方法示意见 `outputs/figures/fig_method_motion.png`。

## 4. 中途发现的问题：人物眼睛半闭（已修）

第一版三行业 demo 渲出来，三个 persona 都**眯着眼**，而 base 反而最自然。
这不是"感觉"，用 MediaPipe 逐帧眼睛纵横比（EAR）量出来：

| | EAR（中位数） | 相对 base |
|---|---|---|
| base（未训练） | 0.343 | — |
| 第一版 persona（blink 目标 0.18–0.21） | **0.102** | 眼睛开合只剩约 30 % |
| **修好后 persona（blink 目标 0.12，惩罚 ×2）** | **0.348** | **1.01× base** |

两条根因：

1. **blink 目标定高了。** v4 的验证值是 0.15，我按"实测眨眼率 / 100"直译成
   0.18–0.21。但 blink 项奖励的是 blink 信号的**峰值**（`smoothmax`），
   目标一高，OPSD 就把表情系数停在接近闭合的位置。
   → 修：`--blink-target-mean 0.12`，`--blink-overshoot-penalty` 4 → 8。
2. **部分源肖像本身就眯眼。** 源图的 EAR 低，再怎么训也睁不开。
   → 修：新增 `scripts/pick_portraits.py`，把 **EAR ≥ 0.30** 作为硬门控，
   与"≥800 px、正面、微笑"一起筛，220 个候选里只留 9 张。

验证用 `scripts/measure_eye_openness.py`（逐帧 EAR + 闭眼帧占比），
修复后闭眼帧占比降到 **0.05–0.17**。

## 5. Demo（跨行业、高清、自然）

### 肖像与音频（无 TTS，**英语真人语音**）

- 肖像 = 真实公开视频的**首帧**，门控：EAR ≥ 0.30、≥800 px、|yaw|/|pitch| ≤ 15°、微笑 > 0
- 驱动音频 = 同一段视频的**真实人声**（CelebV-HQ / HDTF 源片的原始语音，**英语**，16 kHz mono）
- 每行业 3 组（肖像 + 对应语音）× base/persona = 6 个 mp4，另加同脸跨行业对照 6 个
- mp4 已直接放进 `main`：`outputs/industry_demo_v3/{finance,realestate,shortvideo}/`，
  首页 README 里每个行业都有可点的 mp4 链接与并排 GIF

### 渲染设置（高清）

`source_max_dim=1280` + `flag_pasteback=True`：动画脸合成回原分辨率肖像，
不是裁成 256 的小脸。输出 H.264 + AAC 音频，25 fps。

| 行业 | 肖像分辨率 | mp4 数 |
|---|---|---|
| finance | 844–1030 px | 6 |
| realestate | 808–1082 px | 6 |
| shortvideo | 886–1066 px | 6 |
| 同脸对照 | 1030 px（同一张脸） | 6 |

### 同脸对照（最有说服力的一组）

同一张高清肖像（1030 px）+ 同一段英语语音，分别用三个行业 LoRA 渲染
（`outputs/industry_demo_v3/*_control/`）。因为人脸和语音完全相同，
画面上的任何行为差异都只能来自那个行业的 persona——这是"行业感存在"的直接证据。
并排四联 GIF：`outputs/industry_gifs_v3/control/`。

### 眨眼 / 点头 / 倾听 / 微笑 的来源

| 行为 | 从哪来 |
|---|---|
| 眨眼 | reward 的 blink 项（peaked），目标 **0.12**（保守值：实测眨眼率/100 的直译会让眼睛半闭，见 §4），eye_probe 提供梯度 |
| 点头 | amplitude 项（peaked 高斯，中心 = 实测低频幅度） |
| 倾听（非语音段仍有微动作） | couple 项把语音包络和头部速度耦合，没有语音时 amp 仍保持底噪 |
| 微笑 | **不在 reward 里**。微笑来自源肖像本身（候选按嘴角上扬排序），
  LivePortrait 保留源表情。我们没有把 smile 加进 reward，避免这一轮引入不可归因的变化 |

## 6. 复现

```bash
# 0. 环境（JoyVASA + LivePortrait + HuBERT + MediaPipe，~2.5 GB）
bash scripts/h20_render_setup.sh
#    若 cv2 报 libGL/libGLES 缺失：dnf install -y mesa-libGL mesa-libEGL libglvnd-gles
#    Linux 上 HuBERT 目录名：ln -s chinese-hubert-base \
#        third_party/JoyVASA/pretrained_weights/'TencentGameMate:chinese-hubert-base'
#    ffmpeg：本项目用 imageio_ffmpeg 自带的，ln -sf $(python -c
#        'import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())') /usr/local/bin/ffmpeg

# 1. 数据（全部公开）
python scripts/fetch_public_data.py --aishell-speakers 75 --hdtf 40 --celebv 300
python scripts/build_audio_segments.py --n-train 2000 --n-eval 100   # 说话人不重叠

# 2. 三个行业的实测运动带（reward model 的来源）
python scripts/calibrate_industry_bands.py            # -> outputs/industry_bands.json

# 3. 训练（每个 ~14 min on H20；必须过 train_guard.py 守护）
ROUNDS=3000 bash scripts/train_industries.sh          # -> outputs/opsd_ind_*/lora.pt

# 4. 独立评测（eval 说话人与训练不重叠）
python scripts/eval_industries.py --n-clips 12        # -> outputs/industry_eval.json

# 5. demo 肖像 + 渲染 + GIF + 图
python scripts/pick_portraits.py --scan 220 --per-industry 3   # EAR>=0.30, >=800px, frontal, smiling
python scripts/render_industry_demo.py --manifest data/industry_demo_manifest_v3.json --gpu 0
python scripts/measure_eye_openness.py --pattern 'outputs/industry_demo_v3/finance/*.mp4'
# 同脸对照（任选 1-2 张高清肖像，三行业同一张脸）
python scripts/render_industry_demo.py --portrait-override <png...> --gpu 0
python scripts/make_demo_gifs.py --video-dir outputs/industry_demo/finance \
    --out outputs/industry_gifs/finance --variants base finance
python scripts/make_industry_figures.py
```

守护：每次训练都通过 `tools/train_guard.py --kind motion` 启动（selftest OK），
本轮 3 次训练 + 1 次重训**零触发**（无 GUARD_STOP.json）。

## 7. 诚实说明（没做到的）

1. **行业标签是代理**：公开数据没有按行业标注的说话人视频。finance 用新闻主播、
   另两个用 CelebV-HQ 的幅度分位，这是能落地的最诚实做法，但不是"真实房产中介视频"。
2. **finance/realestate 差异小**：因为它们的实测带本来就接近（1.38 vs 1.26 deg）。
   我们不把这个说成"三个行业截然不同"。
3. **jitter 项在 finance 上是 1.27× 目标**（0.120 vs 0.094）：绝对值仍在人类带内
   （0.09–0.35），但没有精确压到目标；这是 peaked reward 在目标较严时的正常松弛。
4. **微笑不在 reward 里**（见 §4），如果要求"由模型控制的微笑"，需要新增表情项并重训，
   本轮为保持归因清晰没有做。
5. **没有 SyncNet / 人工盲评**：本轮 GPU 时间给了三行业训练与渲染与一次返工；
   口型同步仍未测量（与 README 的 limitation 一致）。
6. **眨眼率没有精确校准**：用 EAR 谷值在渲染视频上数出的 blink/min 是 35–77，
   明显高于人类 10–20，说明这个计数法对渲染噪声太敏感、不可信。
   本轮因此**只用 EAR（眼睛开合度）和闭眼帧占比**作为眨眼相关的证据，
   没有把 blink/min 写进结论表。要让眨眼率也进人类区间，需要更可靠的检测与单独一轮调参。
7. **实测带不能线性直译成 reward 参数**（本轮最大教训）：幅度可以直接用
   （因为单位与 reward 内部一致），眨眼却不行——reward 内部是 eye_probe 的归一化量，
   与"次/分钟"不是 1:100。第一版按直译设 0.18–0.21，直接把眼睛训成半闭（§4）。
   教训写在这里：能对齐量纲的才直译，对不上的先做小规模扫描再定。

## 8. 产出清单

```
outputs/
├── industry_bands.json / industry_bands_full.json / industry_bands_small.json
├── industry_eval.json                  # 12 条独立说话人评测
├── naturalness_eval_r3.json            # base/v2/v3/v4 vs 人类带复测
├── opsd_ind_finance|realestate|shortvideo/lora.pt + metrics.jsonl
├── industry_demo_v3/<industry>/*.mp4   # 三行业 demo（base vs persona，高清睁眼）
├── industry_demo_v3/<industry>_control/# 同脸跨行业对照
├── industry_gifs_v3/<industry>/*.gif   # 并排对比 GIF（每行业 base vs persona）
├── industry_gifs_v3/control/*.gif      # 同脸四联对照 GIF
├── eye_openness.json                   # 逐帧 EAR：修眼半闭的前后证据
└── figures/fig_*.png                   # 论文图（训练曲线 / 行业带 / 主结果 / 方法）
scripts/
├── fetch_public_data.py                # AISHELL-1 / HDTF / CelebV-HQ 下载
├── calibrate_industry_bands.py         # 三行业实测运动带（带质量门控）
├── train_industries.sh                 # 三行业训练（参数自动从实测 json 读取）
├── eval_industries.py                  # 独立说话人评测
├── prep_industry_demo.py               # 肖像/音频准备（正面+微笑排序）
├── pick_portraits.py                   # 肖像门控：EAR ≥ 0.30 + ≥800px + 正面 + 微笑
├── render_industry_demo.py             # 高清渲染（支持同脸对照）
├── measure_eye_openness.py             # 逐帧眼睛开合度（EAR / 闭眼帧占比）
└── make_industry_figures.py            # 论文图
```
