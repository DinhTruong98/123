"""Fonts, icons and colors for the watch simulator (WP6).

Everything is cached: fonts per (bold, size), icons per (file, size, color).
pygame.font must be initialised before a Theme is used (pygame.init() or pygame.font.init()).
"""

from __future__ import annotations

import unicodedata
import warnings
from pathlib import Path

import pygame

from soundwatch import paths
from soundwatch.config import Config
from soundwatch.types import Level, SoundClass

Color = tuple[int, int, int]

# watch / window palette
BLACK: Color = (0, 0, 0)
WHITE: Color = (255, 255, 255)
FACE_BG: Color = (8, 10, 14)
WINDOW_BG: Color = (24, 26, 32)
PANEL_BG: Color = (32, 35, 43)
PANEL_LINE: Color = (58, 62, 74)
TEXT_DIM: Color = (150, 156, 170)
TEXT_FAINT: Color = (95, 100, 112)
TRACK: Color = (40, 44, 54)
ACCENT: Color = (60, 200, 140)  # idle RMS arc / "on" toggles
BEZEL: Color = (70, 74, 84)

LEVEL_COLORS: dict[Level, Color] = {
    Level.INFO: (40, 140, 255),
    Level.WARN: (255, 140, 0),
    Level.DANGER: (230, 40, 40),
}
LEVEL_NAMES_VI: dict[Level, str] = {Level.INFO: "Thông tin", Level.WARN: "Chú ý", Level.DANGER: "Nguy hiểm"}


def darken(c: Color, amount: float = 0.30) -> Color:
    """Darken a color by `amount` (0.3 = 30 % darker)."""
    f = max(0.0, 1.0 - amount)
    return (int(c[0] * f), int(c[1] * f), int(c[2] * f))


def lighten(c: Color, amount: float) -> Color:
    return tuple(int(v + (255 - v) * amount) for v in c)  # type: ignore[return-value]


def mix(a: Color, b: Color, t: float) -> Color:
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))  # type: ignore[return-value]


def nfc(text: str) -> str:
    """Vietnamese text must be NFC (precomposed) so every letter maps to one font glyph."""
    return unicodedata.normalize("NFC", text)


class Theme:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.bold_path: Path = cfg.ui.font
        self.regular_path: Path = cfg.ui.font_regular
        self._fonts: dict[tuple[bool, int], pygame.font.Font] = {}
        self._icons: dict[tuple[str, int, Color], pygame.Surface] = {}
        self._text: dict[tuple[str, bool, int, Color], pygame.Surface] = {}
        for p in (self.bold_path, self.regular_path):
            if not p.exists():
                warnings.warn(f"font {p} missing, falling back to pygame default (no Vietnamese diacritics)",
                              stacklevel=2)

    # ---------------------------------------------------------------- fonts
    def font(self, size: int, bold: bool = True) -> pygame.font.Font:
        key = (bold, size)
        f = self._fonts.get(key)
        if f is None:
            path = self.bold_path if bold else self.regular_path
            f = pygame.font.Font(str(path) if path.exists() else None, size)
            self._fonts[key] = f
        return f

    def text(self, text: str, size: int, color: Color = WHITE, bold: bool = True) -> pygame.Surface:
        """Render (cached) antialiased text."""
        text = nfc(text)
        key = (text, bold, size, tuple(color))  # type: ignore[arg-type]
        s = self._text.get(key)
        if s is None:
            s = self.font(size, bold).render(text, True, color)
            if len(self._text) > 2000:
                self._text.clear()
            self._text[key] = s
        return s

    def text_width(self, text: str, size: int, bold: bool = True) -> int:
        return self.font(size, bold).size(nfc(text))[0]

    # ---------------------------------------------------------------- icons
    def icon(self, name: str, size: int, color: Color = WHITE) -> pygame.Surface:
        """Icon from assets/icons, recolored to `color` (alpha kept) and smooth-scaled to size×size."""
        key = (name, size, tuple(color))  # type: ignore[arg-type]
        s = self._icons.get(key)
        if s is None:
            s = self._load_icon(name, size, color)
            self._icons[key] = s
        return s

    def class_icon(self, sc: SoundClass, size: int, color: Color = WHITE) -> pygame.Surface:
        return self.icon(sc.icon, size, color)

    def _load_icon(self, name: str, size: int, color: Color) -> pygame.Surface:
        path = paths.ICONS / name
        if path.exists():
            # PNG with alpha loads as a 32-bit per-pixel-alpha surface; no convert() so this
            # also works before a display mode is set (tests).
            raw = pygame.image.load(str(path))
            if raw.get_bitsize() != 32 or not raw.get_flags() & pygame.SRCALPHA:
                tmp = pygame.Surface(raw.get_size(), pygame.SRCALPHA)
                tmp.blit(raw, (0, 0))
                raw = tmp
        else:  # placeholder: a ring with a question mark
            raw = pygame.Surface((96, 96), pygame.SRCALPHA)
            pygame.draw.circle(raw, WHITE, (48, 48), 44, 8)
            q = self.font(56).render("?", True, WHITE)
            raw.blit(q, q.get_rect(center=(48, 50)))
        if raw.get_size() != (size, size):
            raw = pygame.transform.smoothscale(raw, (size, size))
        out = pygame.Surface((size, size), pygame.SRCALPHA)
        out.fill((*color, 255))
        # tint: solid `color`, alpha taken from the source icon
        a = pygame.surfarray.pixels_alpha(out)
        a[:] = pygame.surfarray.array_alpha(raw)
        del a
        return out

    # --------------------------------------------------------------- colors
    def level_color(self, sc: SoundClass) -> Color:
        return tuple(sc.color) if sc.color else LEVEL_COLORS[sc.level]  # type: ignore[return-value]

    def alert_bg(self, sc: SoundClass) -> Color:
        """Alert background = class/level color darkened ~30 %."""
        return darken(self.level_color(sc), 0.30)
