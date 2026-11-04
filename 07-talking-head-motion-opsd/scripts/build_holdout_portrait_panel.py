#!/usr/bin/env python3
"""Build the polished male/female held-out portrait panel used by the README."""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageFilter

ROOT = Path(__file__).resolve().parents[1]


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def fit_square(image: Image.Image, size: int) -> Image.Image:
    image = image.convert("RGB")
    scale = max(size / image.width, size / image.height)
    resized = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
    left = (resized.width - size) // 2
    top = (resized.height - size) // 2
    return resized.crop((left, top, left + size, top + size))


def add_card(canvas: Image.Image, source: Path, x: int, label: str, subtitle: str) -> None:
    draw = ImageDraw.Draw(canvas)
    card_w, card_h = 1020, 1140
    image_size = 940
    y = 210
    shadow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow)
    shadow_draw.rounded_rectangle((x + 12, y + 18, x + card_w + 12, y + card_h + 18), radius=34, fill=(0, 0, 0, 110))
    canvas.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(16)))
    draw.rounded_rectangle((x, y, x + card_w, y + card_h), radius=34, fill="#111c30", outline="#33496d", width=3)

    portrait = fit_square(Image.open(source), image_size)
    mask = Image.new("L", (image_size, image_size), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, image_size, image_size), radius=24, fill=255)
    canvas.paste(portrait, (x + 40, y + 36), mask)

    draw = ImageDraw.Draw(canvas)
    draw.text((x + 42, y + 1000), label, font=font(34, True), fill="#f3f6ff")
    draw.text((x + 42, y + 1052), subtitle, font=font(21), fill="#9fb1d0")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--female", type=Path, default=ROOT / "data/portraits_new_licensed/pexels_5336771_new_female.png")
    parser.add_argument("--male", type=Path, default=ROOT / "data/portraits_new_licensed/pexels_8379249_new_male.png")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/new_face_demo_r3/holdout_portraits_male_female.png")
    args = parser.parse_args()
    for path in (args.female, args.male):
        if not path.is_file():
            raise SystemExit(f"missing portrait: {path}")

    canvas = Image.new("RGBA", (2200, 1500), "#091321")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, canvas.width, 10), fill="#73a7ff")
    draw.text((90, 68), "TALKING-HEAD MOTION", font=font(34, True), fill="#73a7ff")
    draw.text((90, 112), "Held-out portrait set", font=font(64, True), fill="#f3f6ff")
    draw.text((90, 178), "Same audio, seed, and frozen renderer · male + female generalization check", font=font(24), fill="#9fb1d0")

    add_card(canvas, args.female, 90, "FEMALE · HELD-OUT", "Pexels license · 1024 × 1024 source")
    add_card(canvas, args.male, 1090, "MALE · HELD-OUT", "Pexels license · 1024 × 1024 source")

    draw = ImageDraw.Draw(canvas)
    draw.line((90, 1390, 2110, 1390), fill="#263955", width=2)
    draw.text((90, 1420), "Base vs calibrated motion is shown in the adjacent GIF/MP4 artifacts.", font=font(22), fill="#9fb1d0")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(args.out, format="PNG", optimize=True)
    print(f"wrote {args.out} ({canvas.width}x{canvas.height})")


if __name__ == "__main__":
    main()
