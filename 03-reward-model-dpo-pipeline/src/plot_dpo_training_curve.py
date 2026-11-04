"""Plot the recorded DPO smoke-training history without inventing intermediate values."""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "outputs" / "dpo_smoke_training.json"
OUT = ROOT / "outputs" / "dpo_smoke_training_curve.png"
W, H = 1800, 1050
BG, PANEL, GRID = "#081321", "#101e31", "#263b5b"
TEXT, MUTED, BLUE, GREEN, AMBER = "#f3f6ff", "#9fb2d1", "#72a8ff", "#53d6a2", "#f5bb61"


def font(size: int, bold: bool = False):
    name = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    return ImageFont.truetype(name, size) if Path(name).is_file() else ImageFont.load_default()


def plot_line(draw, values, box, color, lo=None, hi=None):
    x0, y0, x1, y1 = box
    lo = min(values) if lo is None else lo
    hi = max(values) if hi is None else hi
    span = max(hi - lo, 1e-9)
    points = []
    for i, value in enumerate(values):
        x = x0 + (x1 - x0) * i / max(len(values) - 1, 1)
        y = y1 - (y1 - y0) * (value - lo) / span
        points.append((round(x), round(y)))
    draw.line(points, fill=color, width=5)
    for x, y in points:
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=color)
    return lo, hi


def main() -> None:
    history = json.loads(LOG.read_text())
    rows = [row for row in history if "step" in row and "loss" in row and "rewards/accuracies" in row]
    if not rows:
        raise SystemExit(f"no step-level records found in {LOG}")
    steps = [int(row["step"]) for row in rows]
    losses = [float(row["loss"]) for row in rows]
    accuracies = [float(row["rewards/accuracies"]) for row in rows]
    margins = [float(row["rewards/margins"]) for row in rows]

    image = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, W, 12), fill=BLUE)
    draw.text((80, 55), "DPO SMOKE TRAINING", font=font(28, True), fill=BLUE)
    draw.text((80, 100), "Recorded step-level training curves", font=font(54, True), fill=TEXT)
    draw.text((80, 174), "Actual Trainer log: 20 steps on 300 examples sampled from the 2,624 gold DPO pairs.", font=font(23), fill=MUTED)

    draw.rounded_rectangle((70, 235, 1130, 940), radius=24, fill=PANEL, outline=GRID, width=2)
    left, right = 160, 1050
    top, middle, bottom = 330, 590, 850
    for y in (top, middle, bottom):
        draw.line((left, y, right, y), fill=GRID, width=2)
    draw.text((90, top - 20), "loss", font=font(20, True), fill=BLUE)
    draw.text((90, middle - 20), "reward margin", font=font(20, True), fill=GREEN)
    draw.text((90, bottom - 20), "accuracy", font=font(20, True), fill=AMBER)

    loss_lo, loss_hi = min(losses) - 0.0001, max(losses) + 0.0001
    margin_abs = max(abs(min(margins)), abs(max(margins)), 1e-5)
    plot_line(draw, losses, (left, top + 35, right, middle - 30), BLUE, loss_lo, loss_hi)
    plot_line(draw, margins, (left, middle + 35, right, bottom - 30), GREEN, -margin_abs, margin_abs)
    points = []
    for i, value in enumerate(accuracies):
        x = left + (right - left) * i / max(len(accuracies) - 1, 1)
        y = bottom - 30 - (bottom - middle - 65) * value
        points.append((round(x), round(y)))
    draw.line(points, fill=AMBER, width=5)
    for x, y in points:
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=AMBER)
    for i in (0, len(steps) // 2, len(steps) - 1):
        x = left + (right - left) * i / max(len(steps) - 1, 1)
        draw.text((x, bottom + 24), f"step {steps[i]}", font=font(18), fill=MUTED, anchor="ma")

    draw.rounded_rectangle((1210, 235, 1710, 940), radius=24, fill=PANEL, outline=GRID, width=2)
    draw.text((1260, 285), "RUN RECORD", font=font(22, True), fill=BLUE)
    facts = [
        ("steps", f"{len(rows)}"),
        ("examples", "300"),
        ("source", "gold DPO"),
        ("final loss", f"{losses[-1]:.4f}"),
        ("final accuracy", f"{accuracies[-1]:.1%}"),
        ("final margin", f"{margins[-1]:+.2e}"),
    ]
    y = 355
    for label, value in facts:
        draw.text((1260, y), label.upper(), font=font(18, True), fill=MUTED)
        draw.text((1260, y + 32), value, font=font(31, True), fill=TEXT)
        y += 88
    draw.line((1260, 875, 1660, 875), fill=GRID, width=2)
    draw.text((1260, 895), "Smoke test only; not a\nquality or preference claim.", font=font(18), fill=MUTED, spacing=8)
    image.save(OUT, optimize=True)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
