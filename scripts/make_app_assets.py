"""The app's logo files (assets/): the limit line with the building's load under it.

    python scripts/make_app_assets.py
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parents[1] / "assets"
TEAL, INK, BG = (94, 234, 212, 255), (232, 236, 244, 255), (10, 15, 28, 255)


def mark(draw: ImageDraw.ImageDraw, x: float, y: float, s: float) -> None:
    """The mark on a 24-unit grid, scaled by s, at (x, y)."""
    u = lambda v: v * s  # noqa: E731
    draw.rounded_rectangle((x + u(2.5), y + u(3.5), x + u(21.5), y + u(5.5)), radius=u(1), fill=TEAL)
    for bx, by, bh in ((3.5, 13, 8), (10, 9.5, 11.5), (16.5, 15, 6)):
        draw.rounded_rectangle((x + u(bx), y + u(by), x + u(bx + 4), y + u(by + bh)), radius=u(1), fill=INK)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    icon = Image.new("RGBA", (128, 128), (0, 0, 0, 0))
    d = ImageDraw.Draw(icon)
    d.rounded_rectangle((0, 0, 127, 127), radius=26, fill=BG)
    mark(d, 0, 0, 128 / 24)
    icon.save(OUT / "icon.png")

    word = Image.new("RGBA", (440, 96), (0, 0, 0, 0))
    d = ImageDraw.Draw(word)
    mark(d, 4, 16, 64 / 24)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/seguisb.ttf", 46)
    except OSError:
        font = ImageFont.load_default()
    d.text((84, 20), "Headroom", font=font, fill=INK)
    word.save(OUT / "logo.png")
    print("wrote", OUT / "icon.png", OUT / "logo.png")


if __name__ == "__main__":
    main()
