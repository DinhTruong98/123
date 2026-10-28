"""Debug panel (right side of the window): per-class probabilities with threshold ticks,
scrolling log-mel spectrogram, RMS / gate, inference time, FPS and the last events."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pygame

from soundwatch.config import Config
from soundwatch.types import StatusUpdate
from soundwatch.ui import widgets as W
from soundwatch.ui.theme import PANEL_BG, PANEL_LINE, TEXT_DIM, TEXT_FAINT, WHITE, Theme, darken

if TYPE_CHECKING:
    from soundwatch.ui.screens import WatchUI

SPEC_FRAMES = 185  # columns kept in the scrolling spectrogram (~3.7 s at 20 ms/frame)


class DebugPanel:
    def __init__(self, cfg: Config, theme: Theme) -> None:
        self.cfg = cfg
        self.theme = theme
        self.n_mels = cfg.audio.n_mels
        # new frames per StatusUpdate: windows overlap, only the last hop's frames are new
        self.new_frames = max(1, min(cfg.audio.n_frames, round(cfg.audio.hop_samples / cfg.audio.frame_step)))
        self.spec = np.full((SPEC_FRAMES, self.n_mels), np.nan, dtype=np.float32)
        self._have_spec = False
        self._spec_surf: pygame.Surface | None = None
        self._lut = W.magma_lut()

    def on_status(self, msg: StatusUpdate) -> None:
        if msg.logmel is None:
            return  # keep the last spectrogram
        lm = np.asarray(msg.logmel, dtype=np.float32)
        if lm.ndim != 2 or lm.shape[1] != self.n_mels:
            return
        new = lm if not self._have_spec else lm[-self.new_frames:]
        new = new[-SPEC_FRAMES:]
        k = len(new)
        self.spec = np.roll(self.spec, -k, axis=0)
        self.spec[-k:] = new
        self._have_spec = True
        self._spec_surf = None

    def spectrogram_surface(self, size: tuple[int, int]) -> pygame.Surface:
        if self._spec_surf is not None and self._spec_surf.get_size() == size:
            return self._spec_surf
        data = self.spec
        valid = ~np.isnan(data)
        img = np.zeros(data.shape, dtype=np.uint8)
        if valid.any():
            vals = data[valid]
            lo = float(np.percentile(vals, 5))
            hi = max(float(np.percentile(vals, 99.5)), lo + 1.0)
            norm = np.clip((np.nan_to_num(data, nan=lo) - lo) / (hi - lo), 0.0, 1.0)
            img = (norm * 255).astype(np.uint8)
            img[~valid] = 0
        rgb = self._lut[img][:, ::-1, :]  # (frames, mels, 3); low frequencies at the bottom
        surf = pygame.transform.scale(pygame.surfarray.make_surface(np.ascontiguousarray(rgb)), size)
        self._spec_surf = surf
        return surf

    # ------------------------------------------------------------------ draw
    def draw(self, surf: pygame.Surface, rect: pygame.Rect, ui: WatchUI) -> None:
        th = self.theme
        pygame.draw.rect(surf, PANEL_BG, rect)
        pygame.draw.line(surf, PANEL_LINE, rect.topleft, rect.bottomleft)
        x, y, w = rect.x + 16, rect.y + 12, rect.width - 32
        st = ui.status

        surf.blit(th.text("DEBUG", 13, TEXT_DIM), (x, y))
        eng = "engine: " + (f"đã dừng ({ui.engine_stopped})" if ui.engine_stopped else "đang chạy")
        img = th.text(eng, 12, TEXT_FAINT, bold=False)
        surf.blit(img, img.get_rect(topright=(x + w, y + 1)))
        y += 24

        # probabilities
        surf.blit(th.text("Xác suất từng lớp  (vạch = ngưỡng)", 13, WHITE), (x, y))
        y += 22
        lab_w, val_w = 138, 40
        bar_x, bar_w = x + lab_w, w - lab_w - val_w
        probs = st.probs if st is not None else None
        for sc in self.cfg.classes:
            p = float(probs[sc.id]) if probs is not None and sc.id < len(probs) else 0.0
            on = ui.enabled[sc.id]
            col = TEXT_DIM if on else TEXT_FAINT
            surf.blit(th.text(_fit(th, sc.label_vi, 12, lab_w - 8), 12, col, bold=False), (x, y + 2))
            bar = pygame.Rect(bar_x, y + 4, bar_w, 12)
            pygame.draw.rect(surf, (44, 48, 58), bar, border_radius=3)
            over = p >= sc.threshold
            fill_col = sc.color if over else darken(sc.color, 0.55)
            if p > 0:
                pygame.draw.rect(surf, fill_col, (bar.x, bar.y, max(2, round(bar.w * min(1.0, p))), bar.h),
                                 border_radius=3)
            if sc.threshold <= 1.0:
                tx = bar.x + round(bar.w * sc.threshold)
                pygame.draw.line(surf, WHITE, (tx, bar.y - 3), (tx, bar.bottom + 2), 2)
            vt = th.text(f"{p:.2f}", 12, WHITE if over else TEXT_DIM, bold=over)
            surf.blit(vt, vt.get_rect(topright=(x + w, y + 2)))
            y += 21
        y += 8

        # spectrogram
        surf.blit(th.text("Log-mel spectrogram", 13, WHITE), (x, y))
        span_s = SPEC_FRAMES * self.cfg.audio.frame_step / self.cfg.audio.sample_rate
        cap = th.text(f"{self.n_mels} mel · {span_s:.1f} s",
                      11, TEXT_FAINT, bold=False)
        surf.blit(cap, cap.get_rect(topright=(x + w, y + 2)))
        y += 20
        spec_rect = pygame.Rect(x, y, w, 96)
        surf.blit(self.spectrogram_surface(spec_rect.size), spec_rect)
        pygame.draw.rect(surf, PANEL_LINE, spec_rect, 1)
        y = spec_rect.bottom + 12

        # stats
        if st is not None:
            rms = f"RMS {st.rms_dbfs:6.1f} dBFS"
            gate = "GATED (bỏ suy luận)" if st.gated else "qua gate"
            gate_col = (230, 170, 60) if st.gated else (90, 200, 140)
            surf.blit(th.text(rms, 13, WHITE), (x, y))
            g = th.text(gate, 12, gate_col)
            surf.blit(g, g.get_rect(topright=(x + w, y + 1)))
            y += 20
            line = f"infer {st.infer_ms:5.1f} ms   ·   FPS {ui.fps:4.1f}   ·   t = {st.t_audio:7.1f} s"
        else:
            surf.blit(th.text("Chưa có dữ liệu từ engine", 13, TEXT_DIM, bold=False), (x, y))
            y += 20
            line = f"FPS {ui.fps:4.1f}"
        surf.blit(th.text(line, 12, TEXT_DIM, bold=False), (x, y))
        y += 26

        # events
        surf.blit(th.text("Sự kiện gần nhất", 13, WHITE), (x, y))
        y += 20
        from soundwatch.ui.screens import hms

        if not ui.event_log:
            surf.blit(th.text("—", 12, TEXT_FAINT, bold=False), (x, y))
        for ev in ui.event_log:
            sc = self.cfg.classes[ev.class_id]
            pygame.draw.circle(surf, sc.color, (x + 5, y + 8), 4)
            surf.blit(th.text(hms(ev.t_wall), 12, TEXT_DIM, bold=False), (x + 16, y))
            surf.blit(th.text(_fit(th, sc.label_vi, 12, 150), 12, WHITE), (x + 82, y))
            extra = f"{ev.confidence:.2f}" + ("  lặp" if ev.repeat else "")
            e = th.text(extra, 12, TEXT_DIM, bold=False)
            surf.blit(e, e.get_rect(topright=(x + w, y)))
            y += 19
        hint = th.text("Space xác nhận · H lịch sử · S cài đặt · Esc về · Q thoát", 11, TEXT_FAINT,
                       bold=False)
        surf.blit(hint, hint.get_rect(bottomleft=(x, rect.bottom - 8)))


def _fit(th: Theme, text: str, size: int, max_w: int) -> str:
    if th.text_width(text, size, False) <= max_w:
        return text
    while text and th.text_width(text + "…", size, False) > max_w:
        text = text[:-1]
    return text.rstrip() + "…"
