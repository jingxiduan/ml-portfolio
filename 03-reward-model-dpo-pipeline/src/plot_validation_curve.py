"""Plot the recorded Reward Model cross-validation results."""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "outputs" / "reward_model_metrics.json"
OUT = ROOT / "outputs" / "reward_model_validation.png"
W, H = 1600, 900
BG, PANEL, GRID = "#081321", "#101e31", "#263b5b"
TEXT, MUTED, BLUE, GREEN, AMBER = "#f3f6ff", "#9fb2d1", "#72a8ff", "#53d6a2", "#f5bb61"


def f(size: int, bold: bool = False):
    name = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    return ImageFont.truetype(name, size) if Path(name).is_file() else ImageFont.load_default()


def main() -> None:
    metrics = json.loads(METRICS.read_text())
    folds = metrics["cv_auc_per_fold"]
    mean = metrics["cv_auc_mean"]
    held_out = metrics["held_out_test_auc"]
    image = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, W, 12), fill=BLUE)
    draw.text((80, 58), "REWARD MODEL VALIDATION", font=f(28, True), fill=BLUE)
    draw.text((80, 105), "Recorded 5-fold Bradley–Terry evaluation", font=f(52, True), fill=TEXT)
    draw.text((80, 175), "Values are read directly from outputs/reward_model_metrics.json; no synthetic training history.", font=f(22), fill=MUTED)

    left, top, right, bottom = 150, 290, 1030, 730
    draw.rounded_rectangle((80, 240, 1120, 800), radius=24, fill=PANEL, outline=GRID, width=2)
    lo, hi = 0.59, 0.63
    for tick in [0.59, 0.60, 0.61, 0.62, 0.63]:
        y = bottom - int((tick - lo) / (hi - lo) * (bottom - top))
        draw.line((left, y, right, y), fill=GRID, width=2)
        draw.text((95, y - 13), f"{tick:.2f}", font=f(18), fill=MUTED)
    points = []
    for i, value in enumerate(folds):
        x = left + int(i * (right - left) / (len(folds) - 1))
        y = bottom - int((value - lo) / (hi - lo) * (bottom - top))
        points.append((x, y))
    draw.line(points, fill=BLUE, width=6)
    for i, (x, y) in enumerate(points):
        draw.ellipse((x - 11, y - 11, x + 11, y + 11), fill=BLUE, outline=TEXT, width=2)
        draw.text((x, bottom + 28), f"fold {i + 1}", font=f(18), fill=MUTED, anchor="mm")
        draw.text((x, y - 32), f"{folds[i]:.4f}", font=f(18, True), fill=TEXT, anchor="mm")
    mean_y = bottom - int((mean - lo) / (hi - lo) * (bottom - top))
    draw.line((left, mean_y, right, mean_y), fill=GREEN, width=4)
    draw.text((right - 8, mean_y - 28), f"CV mean {mean:.4f}", font=f(19, True), fill=GREEN, anchor="ra")
    draw.text((left, top - 45), "AUC", font=f(20, True), fill=MUTED)

    draw.rounded_rectangle((1170, 240, 1520, 800), radius=24, fill=PANEL, outline=GRID, width=2)
    draw.text((1210, 290), "HELD-OUT", font=f(20, True), fill=AMBER)
    draw.text((1210, 345), f"{held_out:.3f}", font=f(58, True), fill=TEXT)
    draw.text((1210, 425), "test AUC", font=f(23), fill=MUTED)
    draw.line((1210, 500, 1480, 500), fill=GRID, width=2)
    draw.text((1210, 555), "Pairwise accuracy", font=f(20, True), fill=BLUE)
    draw.text((1210, 605), f"{metrics['held_out_test_pairwise_acc']:.3f}", font=f(42, True), fill=TEXT)
    draw.text((1210, 690), "Linear reward model\n5-fold CV + held-out test", font=f(20), fill=MUTED, spacing=10)
    image.save(OUT, optimize=True)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
