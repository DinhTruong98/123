# /// script
# requires-python = ">=3.11"
# dependencies = ["pillow==11.3.0"]
# ///
"""Draw the SoundWatch app icon (light / dark / tinted, 1024x1024) and the launch logo.

    uv run ios/scripts/make_app_icon.py

Motif: a watch-face ring around a sound waveform (5 rounded bars) with an alert dot, on a warm
red-orange gradient (light). Dark: same glyph in the gradient on a transparent background.
Tinted: white glyph on black (iOS applies the user's tint).
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "SoundWatch" / "Assets.xcassets"
S = 1024
SS = 4  # supersampling
TOP = (255, 122, 69)     # #FF7A45
BOTTOM = (214, 40, 57)   # #D62839
ALERT = (255, 214, 10)   # systemYellow


def gradient(size: int) -> Image.Image:
    """Diagonal top-left -> bottom-right gradient."""
    small = Image.new("RGB", (256, 256))
    px = small.load()
    for y in range(256):
        for x in range(256):
            t = (x + y) / 510
            px[x, y] = tuple(round(a + (b - a) * t) for a, b in zip(TOP, BOTTOM, strict=True))
    return small.resize((size, size), Image.BICUBIC)


def glyph_mask(size: int, with_dot: bool = True) -> tuple[Image.Image, Image.Image]:
    """(glyph mask, alert-dot mask) at `size`, drawn supersampled."""
    n = size * SS
    k = n / S
    g = Image.new("L", (n, n), 0)
    d = ImageDraw.Draw(g)
    c = n / 2
    r, w = 330 * k, 54 * k
    d.ellipse((c - r, c - r, c + r, c + r), outline=255, width=round(w))
    # gap in the ring at the top right, where the alert dot sits
    dot = Image.new("L", (n, n), 0)
    if with_dot:
        dx, dy, dr = c + 233 * k, c - 233 * k, 92 * k
        d.ellipse((dx - dr - 36 * k, dy - dr - 36 * k, dx + dr + 36 * k, dy + dr + 36 * k), fill=0)
        ImageDraw.Draw(dot).ellipse((dx - dr, dy - dr, dx + dr, dy + dr), fill=255)
    bw, gap = 52 * k, 34 * k
    heights = [150, 300, 430, 300, 150]
    x0 = c - (len(heights) * bw + (len(heights) - 1) * gap) / 2
    for i, h in enumerate(heights):
        x = x0 + i * (bw + gap)
        hh = h * k / 2
        d.rounded_rectangle((x, c - hh, x + bw, c + hh), radius=bw / 2, fill=255)
    return g.resize((size, size), Image.LANCZOS), dot.resize((size, size), Image.LANCZOS)


def light() -> Image.Image:
    img = gradient(S)
    g, dot = glyph_mask(S)
    img.paste((255, 255, 255), mask=g)
    img.paste(ALERT, mask=dot)
    return img


def dark() -> Image.Image:
    g, dot = glyph_mask(S)
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    grad = gradient(S).convert("RGBA")
    img.paste(grad, mask=g)
    img.paste(ALERT + (255,), mask=dot)
    return img


def tinted() -> Image.Image:
    g, dot = glyph_mask(S)
    img = Image.new("RGB", (S, S), (0, 0, 0))
    img.paste((255, 255, 255), mask=ImageChops.lighter(g, dot))
    return img


def launch_logo() -> Image.Image:
    """Glyph in white on transparent, 3x of a 160 pt logo."""
    size = 480
    g, dot = glyph_mask(size)
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    img.paste((255, 255, 255, 255), mask=g)
    img.paste(ALERT + (255,), mask=dot)
    return img


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")


def main() -> None:
    info = {"author": "xcode", "version": 1}
    write_json(ASSETS / "Contents.json", {"info": info})

    icon = ASSETS / "AppIcon.appiconset"
    icon.mkdir(parents=True, exist_ok=True)
    light().save(icon / "icon-light.png")
    dark().save(icon / "icon-dark.png")
    tinted().save(icon / "icon-tinted.png")
    entry = {"idiom": "universal", "platform": "ios", "size": "1024x1024"}
    write_json(icon / "Contents.json", {"images": [
        {**entry, "filename": "icon-light.png"},
        {**entry, "filename": "icon-dark.png", "appearances": [{"appearance": "luminosity", "value": "dark"}]},
        {**entry, "filename": "icon-tinted.png", "appearances": [{"appearance": "luminosity", "value": "tinted"}]},
    ], "info": info})

    logo = ASSETS / "LaunchLogo.imageset"
    logo.mkdir(parents=True, exist_ok=True)
    launch_logo().save(logo / "launch-logo.png")
    write_json(logo / "Contents.json", {"images": [
        {"idiom": "universal", "filename": "launch-logo.png", "scale": "3x"},
    ], "info": info})

    def color(r, g, b):
        return {"color": {"color-space": "srgb", "components": {
            "red": f"{r / 255:.3f}", "green": f"{g / 255:.3f}", "blue": f"{b / 255:.3f}", "alpha": "1.000"}}}

    write_json(ASSETS / "LaunchBackground.colorset" / "Contents.json", {"colors": [
        {"idiom": "universal", **color(*BOTTOM)},
    ], "info": info})
    write_json(ASSETS / "AccentColor.colorset" / "Contents.json", {"colors": [
        {"idiom": "universal", **color(*BOTTOM)},
        {"idiom": "universal", "appearances": [{"appearance": "luminosity", "value": "dark"}], **color(*TOP)},
    ], "info": info})
    print(f"wrote icons and launch assets to {ASSETS}")


if __name__ == "__main__":
    main()
