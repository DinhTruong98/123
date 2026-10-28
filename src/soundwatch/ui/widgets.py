"""Drawing primitives for the watch simulator: arcs, circular mask, fitted labels, toggles,
vibration timeline, spectrogram colormap. Pure functions on pygame Surfaces (no main loop)."""

from __future__ import annotations

import math

import numpy as np
import pygame
import pygame.gfxdraw

from soundwatch.ui.theme import TEXT_DIM, TEXT_FAINT, TRACK, WHITE, Color, Theme, nfc
from soundwatch.ui.vibration import VibrationState


# ------------------------------------------------------------------ geometry
def polar(center: tuple[float, float], r: float, deg: float) -> tuple[float, float]:
    """Point at angle `deg` (0 = 12 o'clock, clockwise) and radius r."""
    a = math.radians(deg)
    return center[0] + r * math.sin(a), center[1] - r * math.cos(a)


def draw_arc(surf: pygame.Surface, color: Color, center: tuple[float, float], radius: float, width: float,
             start_deg: float, end_deg: float, caps: bool = True) -> None:
    """Thick antialiased arc; `radius` is the outer radius. Angles: 0 = top, clockwise."""
    if end_deg - start_deg <= 0.05:
        return
    r_in = radius - width
    n = max(2, int(abs(end_deg - start_deg) / 2.0) + 1)
    outer = [polar(center, radius, start_deg + (end_deg - start_deg) * i / (n - 1)) for i in range(n)]
    inner = [polar(center, r_in, end_deg - (end_deg - start_deg) * i / (n - 1)) for i in range(n)]
    pts = [(round(x), round(y)) for x, y in outer + inner]
    pygame.gfxdraw.filled_polygon(surf, pts, color)
    pygame.gfxdraw.aapolygon(surf, pts, color)
    if caps:
        rc = width / 2.0
        for deg in (start_deg, end_deg):
            x, y = polar(center, radius - rc, deg)
            pygame.gfxdraw.filled_circle(surf, round(x), round(y), max(1, round(rc - 0.5)), color)
            pygame.gfxdraw.aacircle(surf, round(x), round(y), max(1, round(rc - 0.5)), color)


def draw_gauge(surf: pygame.Surface, center: tuple[float, float], radius: float, width: float, value: float,
               color: Color, track: Color = TRACK, sweep: float = 270.0) -> None:
    """Gauge arc centred at the top, opening at the bottom; value in [0, 1]."""
    a0 = -sweep / 2.0
    draw_arc(surf, track, center, radius, width, a0, a0 + sweep)
    v = min(1.0, max(0.0, value))
    if v > 0.005:
        draw_arc(surf, color, center, radius, width, a0, a0 + sweep * v)


def aa_circle(surf: pygame.Surface, color: Color, center: tuple[int, int], r: int) -> None:
    pygame.gfxdraw.filled_circle(surf, center[0], center[1], r, color)
    pygame.gfxdraw.aacircle(surf, center[0], center[1], r, color)


def circle_mask(size: int) -> pygame.Surface:
    """White surface, alpha 255 inside the round display and 0 outside (AA edge). Blit it onto
    an SRCALPHA face with BLEND_RGBA_MULT to cut the face round."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) + 0.5
    c = size / 2.0
    d = np.sqrt((xx - c) ** 2 + (yy - c) ** 2)
    alpha = (1.0 - np.clip(d - (c - 1.0), 0.0, 1.0)) * 255.0
    s = pygame.Surface((size, size), pygame.SRCALPHA)
    s.fill((255, 255, 255, 255))
    a = pygame.surfarray.pixels_alpha(s)
    a[:] = alpha.T.astype(np.uint8)
    del a
    return s


def chord_half_width(size: int, y: float, margin: float = 0.0) -> float:
    """Half the usable width of the round display at row y."""
    r = size / 2.0 - margin
    dy = y - size / 2.0
    return math.sqrt(max(0.0, r * r - dy * dy))


# ---------------------------------------------------------------------- text
def _split_two(text: str) -> list[tuple[str, str]]:
    words = text.split(" ")
    return [(" ".join(words[:i]), " ".join(words[i:])) for i in range(1, len(words))]


def fit_lines(theme: Theme, text: str, max_w: float, sizes: tuple[int, ...], bold: bool = True,
              max_lines: int = 2, two_line_max: int | None = None) -> tuple[int, list[str]]:
    """Largest size (from `sizes`, descending) at which `text` fits in max_w on 1 or 2 lines
    (two lines only at sizes <= two_line_max)."""
    text = nfc(text)
    for size in sizes:
        if theme.text_width(text, size, bold) <= max_w:
            return size, [text]
        if max_lines >= 2 and (two_line_max is None or size <= two_line_max):
            best = None
            for a, b in _split_two(text):
                w = max(theme.text_width(a, size, bold), theme.text_width(b, size, bold))
                if w <= max_w and (best is None or w < best[0]):
                    best = (w, [a, b])
            if best:
                return size, best[1]
    return sizes[-1], [text]


def blit_center(surf: pygame.Surface, img: pygame.Surface, cx: float, cy: float) -> pygame.Rect:
    r = img.get_rect(center=(round(cx), round(cy)))
    surf.blit(img, r)
    return r


def draw_lines_centered(surf: pygame.Surface, theme: Theme, lines: list[str], size: int, cx: float,
                        top: float, color: Color = WHITE, bold: bool = True,
                        line_step: float = 1.12) -> float:
    """Draw centred lines starting at `top` (line pitch = line_step × size); returns the bottom y."""
    lh = size * line_step
    y = top
    for ln in lines:
        img = theme.text(ln, size, color, bold)
        surf.blit(img, img.get_rect(midtop=(round(cx), round(y))))
        y += lh
    return top + lh * (len(lines) - 1) + theme.font(size, bold).get_linesize()


# ------------------------------------------------------------------- widgets
def draw_ring(surf: pygame.Surface, color: Color, center: tuple[int, int], r_outer: int,
              r_inner: int) -> None:
    """Antialiased filled ring."""
    pygame.draw.circle(surf, color, center, r_outer, r_outer - r_inner + 1)
    pygame.gfxdraw.aacircle(surf, center[0], center[1], r_outer, color)
    pygame.gfxdraw.aacircle(surf, center[0], center[1], r_inner, color)


def draw_toggle(surf: pygame.Surface, rect: pygame.Rect, on: bool, on_color: Color,
                off_color: Color = (70, 74, 86)) -> None:
    r = rect.height // 2
    pygame.draw.rect(surf, on_color if on else off_color, rect, border_radius=r)
    knob_x = rect.right - r if on else rect.left + r
    aa_circle(surf, WHITE, (knob_x, rect.centery), r - 2)


def draw_vibration_glyph(surf: pygame.Surface, center: tuple[int, int], color: Color = WHITE,
                         scale: float = 1.0) -> None:
    """Phone-with-waves 'vibrate' glyph."""
    cx, cy = center
    w, h = int(10 * scale), int(16 * scale)
    pygame.draw.rect(surf, color, (cx - w // 2, cy - h // 2, w, h), max(1, int(2 * scale)),
                     border_radius=max(1, int(2 * scale)))
    lw = max(1, int(2 * scale))
    for side in (-1, 1):
        for k in (1, 2):
            x = cx + side * (w // 2 + int(3 * scale * k))
            pygame.draw.line(surf, color, (x, cy - int(4 * scale) + (k - 1)),
                             (x, cy + int(4 * scale) - (k - 1)), lw)


def draw_pattern_timeline(surf: pygame.Surface, theme: Theme, rect: pygame.Rect, pattern: tuple[int, ...],
                          state: VibrationState, color: Color, title: str) -> None:
    """Pattern on/off segments as a bar with a playhead; dim when not playing."""
    title_img = theme.text(title, 15, TEXT_DIM if not state.active else WHITE, bold=True)
    surf.blit(title_img, (rect.x, rect.y))
    bar = pygame.Rect(rect.x, rect.y + 24, rect.width, max(8, rect.height - 44))
    pygame.draw.rect(surf, (36, 39, 47), bar, border_radius=4)
    if not pattern:
        img = theme.text("(không rung)", 13, TEXT_FAINT, bold=False)
        surf.blit(img, img.get_rect(center=bar.center))
        return
    total = float(sum(pattern))
    x = float(bar.x)
    on_col = color if state.active else tuple(int(v * 0.45) for v in color)
    for i, d in enumerate(pattern):
        w = bar.width * d / total
        seg = pygame.Rect(round(x), bar.y, max(1, round(x + w) - round(x)), bar.height)
        if i % 2 == 0:
            c = on_col
            if state.active and i == state.segment:
                c = tuple(min(255, int(v * 1.25 + 30)) for v in color)
            pygame.draw.rect(surf, c, seg.inflate(-2, -2 if i != state.segment else 0), border_radius=3)
        x += w
    if state.active:
        px = bar.x + bar.width * state.progress
        pygame.draw.line(surf, WHITE, (px, bar.y - 4), (px, bar.bottom + 4), 2)
    ms = " · ".join(str(v) for v in pattern) + " ms"
    lab = theme.text(ms, 12, TEXT_FAINT, bold=False)
    surf.blit(lab, (rect.x, bar.bottom + 4))


# --------------------------------------------------------------- colormaps
_MAGMA = [(0, 0, 4), (28, 16, 68), (79, 18, 123), (129, 37, 129), (181, 54, 122), (229, 80, 100),
          (251, 135, 97), (254, 194, 135), (252, 253, 191)]


def magma_lut(n: int = 256) -> np.ndarray:
    anchors = np.asarray(_MAGMA, dtype=np.float32)
    xs = np.linspace(0, 1, len(anchors))
    t = np.linspace(0, 1, n)
    return np.stack([np.interp(t, xs, anchors[:, k]) for k in range(3)], axis=1).astype(np.uint8)
