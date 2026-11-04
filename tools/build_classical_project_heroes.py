#!/usr/bin/env python3
"""Generate compact, consistent hero panels for portfolio projects 01-03."""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
BG = "#081321"
PANEL = "#101e31"
PANEL_2 = "#142641"
TEXT = "#f3f6ff"
MUTED = "#9fb2d1"
BLUE = "#72a8ff"
GREEN = "#53d6a2"
AMBER = "#f5bb61"
RED = "#f07f87"
GRID = "#263b5b"
W, H = 2200, 1220


def font(size: int, bold: bool = False):
    name = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    return ImageFont.truetype(name, size) if Path(name).is_file() else ImageFont.load_default()


def text(draw, xy, value, size=24, fill=TEXT, bold=False, anchor=None):
    draw.text(xy, value, font=font(size, bold), fill=fill, anchor=anchor)


def rounded(draw, box, fill=PANEL, outline=GRID, radius=26, width=2):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def base(title, kicker, subtitle):
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, W, 12), fill=BLUE)
    text(d, (90, 72), kicker.upper(), 30, BLUE, True)
    text(d, (90, 115), title, 64, TEXT, True)
    text(d, (90, 190), subtitle, 25, MUTED)
    return im, d


def metric_card(d, x, y, w, label, value, note, accent=BLUE):
    rounded(d, (x, y, x + w, y + 170), PANEL_2, GRID, 22)
    d.rectangle((x, y, x + 8, y + 170), fill=accent)
    text(d, (x + 30, y + 27), label.upper(), 19, MUTED, True)
    text(d, (x + 30, y + 70), value, 42, TEXT, True)
    text(d, (x + 30, y + 130), note, 19, MUTED)


def arrow(d, x1, y, x2, color=BLUE):
    d.line((x1, y, x2 - 18, y), fill=color, width=5)
    d.polygon([(x2 - 18, y - 10), (x2, y), (x2 - 18, y + 10)], fill=color)


def project01():
    metrics = json.loads((ROOT / "01-multitask-ctr-cvr-rec/outputs/metrics.json").read_text())["test"]
    im, d = base("Multi-task CTR / CVR ranking", "01 · recommendation", "RankMixer-style dense interaction · shared representation with masked CVR supervision")
    metric_card(d, 90, 265, 470, "CTR AUC", f"{metrics['ctr']['auc']:.4f}", f"GAUC {metrics['ctr']['gauc']:.4f} · LogLoss {metrics['ctr']['logloss']:.3f}", BLUE)
    metric_card(d, 590, 265, 470, "CVR AUC", f"{metrics['cvr']['auc']:.4f}", f"clicked-sample task · LogLoss {metrics['cvr']['logloss']:.3f}", GREEN)
    metric_card(d, 1090, 265, 470, "CTR calibration", f"{metrics['ctr']['abs_bias']:.4f}", "absolute bucket bias · lower is better", AMBER)
    metric_card(d, 1590, 265, 520, "Runtime", "~3 min", "CPU · 4 epochs · MovieLens-1M proxy labels", BLUE)

    rounded(d, (90, 485, 2110, 860), PANEL, GRID, 28)
    text(d, (130, 530), "MODEL PATH", 22, BLUE, True)
    boxes = [(130, "Sparse + dense\nfeatures"), (570, "Shared bottom\nembeddings"), (1010, "RankMixer\ntoken/channel mix"), (1450, "Task towers\nCTR + CVR")]
    for i, (x, label) in enumerate(boxes):
        rounded(d, (x, 635, x + 320, 755), PANEL_2, GRID, 18)
        lines = label.split("\n")
        text(d, (x + 160, 675), lines[0], 25, TEXT, True, "mm")
        text(d, (x + 160, 715), lines[1], 22, MUTED, False, "mm")
        if i < len(boxes) - 1:
            arrow(d, x + 335, 695, boxes[i + 1][0] - 18)
    text(d, (130, 815), "CVR loss is applied only when click = 1  ·  avoids treating every non-click as a conversion negative", 23, MUTED)

    rounded(d, (90, 900, 2110, 1130), PANEL_2, GRID, 22)
    text(d, (130, 940), "WHY IT MATTERS", 21, BLUE, True)
    text(d, (130, 990), "One shared representation learns ranking signals while task-specific towers respect the click → conversion dependency.", 25, TEXT)
    text(d, (130, 1040), "Dataset caveat: MovieLens ratings and negative sampling are documented proxy labels, not production ad logs.", 22, MUTED)
    im.save(ROOT / "01-multitask-ctr-cvr-rec/outputs/hero.png", optimize=True)


def project02():
    comp = json.loads((ROOT / "02-rtb-bid-shading-control/outputs/strategy_comparison.json").read_text())
    im, d = base("RTB bid shading as feedback control", "02 · ad auction control", "pCTR + win-rate model + candidate-bid search · PID / MPC adapt the bid coefficient online")
    metric_card(d, 90, 265, 470, "Best profit", f"{comp['pid']['total_profit']:,.0f}", "PID feedback · same simulated market", GREEN)
    metric_card(d, 590, 265, 470, "Win rate", f"{comp['pid']['overall_win_rate']:.1%}", f"vs fixed {comp['fixed']['overall_win_rate']:.1%}", BLUE)
    metric_card(d, 1090, 265, 470, "ROI", f"{comp['pid']['overall_roi']:.2f}", "target ROI = 1.4", AMBER)
    metric_card(d, 1590, 265, 520, "pCTR AUC", "0.7796", "400K real Avazu click records", BLUE)

    rounded(d, (90, 485, 2110, 860), PANEL, GRID, 28)
    text(d, (130, 530), "CONTROL LOOP", 22, BLUE, True)
    boxes = [(130, "pCTR\nmodel"), (510, "Bid / market\nfeatures"), (890, "Win-rate\nmodel"), (1270, "PID / MPC\ncontroller"), (1650, "Next bid\ncoefficient")]
    for i, (x, label) in enumerate(boxes):
        rounded(d, (x, 635, x + 280, 755), PANEL_2, GRID, 18)
        lines = label.split("\n")
        text(d, (x + 140, 675), lines[0], 25, TEXT, True, "mm")
        text(d, (x + 140, 715), lines[1], 22, MUTED, False, "mm")
        if i < len(boxes) - 1:
            arrow(d, x + 295, 695, boxes[i + 1][0] - 18)
    text(d, (130, 815), "The controller closes the loop on ROI / spend deviation instead of using one static bid multiplier.", 23, MUTED)

    rounded(d, (90, 900, 2110, 1130), PANEL_2, GRID, 22)
    text(d, (130, 940), "SAME MARKET · THREE POLICIES", 21, BLUE, True)
    names = [("Fixed", comp["fixed"]["total_profit"], RED), ("PID", comp["pid"]["total_profit"], GREEN), ("MPC", comp["mpc"]["total_profit"], BLUE)]
    max_profit = max(value for _, value, _ in names)
    for i, (name, value, color) in enumerate(names):
        y = 1000 + i * 40
        text(d, (130, y), name, 21, TEXT, True)
        d.rounded_rectangle((300, y - 11, 300 + int(1100 * value / max_profit), y + 12), 10, fill=color)
        text(d, (1450, y), f"profit {value:,.0f}", 20, MUTED)
        text(d, (1800, y), f"ROI {comp[name.lower()]['overall_roi']:.2f}", 20, MUTED)
    im.save(ROOT / "02-rtb-bid-shading-control/outputs/hero.png", optimize=True)


def project03():
    funnel = json.loads((ROOT / "03-reward-model-dpo-pipeline/outputs/dpo_data_funnel.json").read_text())[0]
    metrics = json.loads((ROOT / "03-reward-model-dpo-pipeline/outputs/reward_model_metrics.json").read_text())
    im, d = base("Preference data → Reward Model → DPO", "03 · alignment pipeline", "confidence filtering turns noisy preference pairs into a lightweight, deployable reward signal")
    metric_card(d, 90, 265, 470, "Gold DPO pairs", f"{funnel['n_gold_for_dpo']:,}", f"from {funnel['n_raw_pairs']:,} raw train pairs", GREEN)
    metric_card(d, 590, 265, 470, "Held-out RM AUC", f"{metrics['held_out_test_auc']:.3f}", f"pairwise accuracy {metrics['held_out_test_pairwise_acc']:.3f}", BLUE)
    metric_card(d, 1090, 265, 470, "Judge → RM", f"{metrics['latency_benchmark']['speedup_x']:,.0f}×", "inference speedup on 300 samples", AMBER)
    metric_card(d, 1590, 265, 520, "RM latency", f"{metrics['latency_benchmark']['reward_model_ms_per_sample']:.4f} ms", "linear forward on precomputed features", BLUE)

    rounded(d, (90, 485, 2110, 860), PANEL, GRID, 28)
    text(d, (130, 530), "DATA FUNNEL", 22, BLUE, True)
    stages = [("Raw pairs", funnel["n_raw_pairs"], BLUE), ("Ambiguous", funnel["n_ambiguous_sent_to_judge"], AMBER), ("Noisy removed", funnel["n_noisy_discarded_by_judge"], RED), ("Gold for DPO", funnel["n_gold_for_dpo"], GREEN)]
    max_value = max(v for _, v, _ in stages)
    for i, (label, value, color) in enumerate(stages):
        x = 130 + i * 500
        h = int(220 * value / max_value)
        d.rounded_rectangle((x, 785 - h, x + 280, 785), 18, fill=color)
        text(d, (x + 140, 815), label, 22, TEXT, True, "ma")
        text(d, (x + 140, 850), f"{value:,}", 25, MUTED, False, "ma")
        if i < len(stages) - 1:
            arrow(d, x + 300, 675, x + 470, MUTED)
    text(d, (130, 555), "human preference labels  →  proxy judge on ambiguous cases  →  confidence filter  →  standard DPO format", 22, MUTED)

    rounded(d, (90, 900, 2110, 1130), PANEL_2, GRID, 22)
    text(d, (130, 940), "DEPLOYMENT PAYOFF", 21, BLUE, True)
    text(d, (130, 990), "Distill an expensive judge call into a cheap scalar reward that can score the full candidate pool.", 25, TEXT)
    text(d, (130, 1040), "Boundary: the included DPO trainer is a 20-step CPU smoke test; the measured result is the data funnel and reward-signal latency.", 22, MUTED)
    im.save(ROOT / "03-reward-model-dpo-pipeline/outputs/hero.png", optimize=True)


def main():
    project01()
    project02()
    project03()
    print("generated hero.png for projects 01, 02, and 03")


if __name__ == "__main__":
    main()
