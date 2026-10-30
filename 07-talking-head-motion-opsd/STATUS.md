# 07 — Talking-Head Motion OPSD · 项目状态

> 最后更新：2026-10-30 H20 session 到期前
>
> **下一轮任务见 [`docs/NEXT_H20_TASKS.md`](docs/NEXT_H20_TASKS.md)**（jitter 单点优化 + 放大数据 + 独立评测）。
> 下面"尚未完成"一节是旧清单，以任务书为准。

## 已完成

### 1. OPSD LoRA 训练（motion-space）
- **V1**: 600步，reward 从 0.35 → 1.16（win rate 100% vs base）
- **V2**: 600步，reward 改为 gaussian blink target + jitter hard cap（阈值 1.0°；从训练日志反推 jitter 实际约 0.2°，cap 很可能从未触发，待诊断）
- LoRA checkpoints: `outputs/opsd/lora.pt`, `outputs/opsd_v2/lora.pt`
- 训练 metrics: `outputs/opsd/metrics.jsonl`, `outputs/opsd_v2/metrics.jsonl`（各 600 步）

### 2. Tier-2: 视频空间 Eval（H20 GPU，n=20 clips × 6 variants = 120 videos）
- 6 个配置对比：base / opsd_v1_naive / opsd_v1_BoN / opsd_v1_BoN+thin / opsd_v2_naive / opsd_v2_BoN
- MediaPipe 视频空间 metrics（blink_rate, head_jitter, gaze_std, etc.）
- 关键结果：

| metric | base | opsd v1 (naive) | + BoN | + BoN + thin | opsd v2 (naive) | opsd v2 + BoN | human target |
|---|---|---|---|---|---|---|---|
| blink_rate | 0.00 | 50.49 | 34.06 | **24.78** | 42.90 | **25.62** | 10-20 |
| head_jitter | 119.79 | 262.09 | 264.69 | 265.22 | 261.06 | **256.91** | ~120 |
| head_pitch_std | 1.27 | 3.68 | 3.67 | 3.66 | **3.35** | 3.35 | 1-4 |
| speech_head_corr | 0.13 | **0.27** | 0.27 | 0.27 | 0.25 | 0.26 | 0.2-0.4 |

- 20 组 3-way GIF 对比: `outputs/tier2/tier2_compare_00..19.gif`
- 训练曲线图: `fig_tier2_blink.png`, `fig_tier2_summary.png`, `fig_v1_vs_v2_curve.png`

### 3. Engine Routing Prototype
- 6 个 engine profile: `outputs/profiles/engine_profiles.json`
- Contextual bandit routing: `outputs/routing/routing_results.json` + `routing_curve.png`

### 4. 文档
- 完整 README.md（271行）
- `docs/H20_RENDER_PLAYBOOK.md`
- `docs/TIER2_NOTE.md`

## 产出文件清单

```
outputs/
├── opsd/lora.pt + metrics.jsonl          # V1 LoRA + 600步 metrics
├── opsd_v2/lora.pt + metrics.jsonl       # V2 LoRA + 600步 metrics
├── tier2/                                # 120个 mp4 + 20个 GIF + metrics + figures (751MB)
├── eval/eval_results.json                # motion-space eval
├── profiles/engine_profiles.json         # engine profiling
├── routing/routing_results.json + .png   # routing experiment
├── samples/                              # early samples (11MB)
└── eye_probe/                            # eye debugging (352K)
```

## 尚未完成 / 下次要做

### 优先级高
1. **降低 blink_rate**：当前 v2+BoN=25.62，人类 target 10-20。方案：
   - 加强 blink_thinning 后处理（缩小最小间距从 2.5s → 3.5s）
   - 或在 V3 reward 中降低 blink 权重、提高 jitter penalty
2. **降低 head_jitter**：当前 ~260，人类 ~120。OPSD 让头动了但太抖。方案：
   - 提高 jitter penalty 的权重（当前 -0.5 → -1.5）
   - 或加 temporal smoothing 后处理
3. **jaw_audio_corr 是负数**：唇动和音频不同步。需要重新审视 lip-sync loss

### 优先级中
4. **增加 eval clips 数量**：当前 n=20，需要 50+ 才统计显著
5. **Human evaluation**：当前全是自动 metrics，需要 A/B 人工打分
6. **FVD / LPIPS 指标**：标准视频生成 metrics 未计算

### 优先级低
7. **Routing bandit 在线化**：当前是离线 simulation，可以接真实用户偏好
8. **不同 portrait / audio pair**：当前只用了限定几个人脸+音频
