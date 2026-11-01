"""Watch UI state machine + the four watch screens (Idle / Alert / History / Settings).

`WatchUI` holds all UI state and is driven from outside, so everything is testable without a
main loop:
    ui.handle_message(msg)   # StatusUpdate / AlertEvent / EngineStopped from the engine
    ui.handle_event(event)   # pygame events (keyboard / mouse)
    ui.update(now)           # timers (alert timeout, flashing, vibration), now = time.monotonic()
    ui.draw(window_surface)  # full window: watch face + vibration timeline + debug panel
    ui.render_face()         # just the 240×240 round display (unscaled)
Commands for the engine (AckCommand / SettingsCommand) are put on `in_queue`.
"""

from __future__ import annotations

import queue
import random
import time
from collections import deque
from dataclasses import dataclass

import pygame

from soundwatch.config import Config
from soundwatch.types import (
    AckCommand,
    AlertEvent,
    EngineStopped,
    Level,
    ModeCommand,
    SettingsCommand,
    SoundClass,
    StatusUpdate,
    event_icon,
    event_label,
)
from soundwatch.ui import widgets as W
from soundwatch.ui.theme import (
    ACCENT,
    BEZEL,
    FACE_BG,
    LEVEL_NAMES_VI,
    TEXT_DIM,
    TEXT_FAINT,
    TRACK,
    WHITE,
    WINDOW_BG,
    Theme,
    darken,
    lighten,
)
from soundwatch.ui.vibration import VibrationPlayer

SCREENS = ("idle", "alert", "history", "settings")
WEEKDAYS_VI = ("Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật")
RMS_FLOOR_DB = -80.0  # bottom of the idle RMS gauge
DEBUG_PANEL_W = 400
JITTER_PX = 3


MODE_NAMES_VI = {"indoor": "Trong nhà", "street": "Ngoài đường"}


@dataclass
class ActiveAlert:
    event: AlertEvent
    shown_at: float  # monotonic time the alert (or its latest repeat) was put on screen


@dataclass(frozen=True)
class HistoryEntry:
    class_id: int
    confidence: float
    t_wall: float
    label: str | None = None  # context-mode display override at alert time (vn-v3)
    icon: str | None = None


def hms(t_wall: float) -> str:
    return time.strftime("%H:%M:%S", time.localtime(t_wall))


class WatchUI:
    def __init__(self, cfg: Config, in_queue: queue.Queue | None = None, show_debug: bool = True,
                 seed: int = 0) -> None:
        self.cfg = cfg
        self.in_queue = in_queue
        self.show_debug = show_debug
        self.theme = Theme(cfg)
        self.size = cfg.ui.watch_size
        self.scale = cfg.ui.scale
        self.face_px = self.size * self.scale
        self.rng = random.Random(seed)

        win_w, win_h = cfg.ui.window
        left_w = win_w - DEBUG_PANEL_W if show_debug else self.face_px + 80
        self.window_size = (win_w if show_debug else left_w, win_h)
        self.face_pos = ((left_w - self.face_px) // 2, 22)
        self.panel_rect = pygame.Rect(left_w, 0, DEBUG_PANEL_W, win_h)

        self.now = time.monotonic()
        self.wall = time.time()
        self.screen = "idle"
        self.return_screen = "idle"
        self.active: ActiveAlert | None = None
        self.pending_danger: dict[int, AlertEvent] = {}
        self.history: deque[HistoryEntry] = deque(maxlen=cfg.ui.history_len)
        self.event_log: deque[AlertEvent] = deque(maxlen=6)
        self.enabled = [c.enabled for c in cfg.classes]
        self.settings_sel = 0
        self.history_scroll = 0.0
        self.settings_scroll = 0.0
        self.status: StatusUpdate | None = None
        self.engine_stopped: str | None = None
        self.vib = VibrationPlayer()
        self.vib_class: SoundClass | None = None
        self.fps = 0.0
        self.quit_requested = False
        self.sent: list[object] = []  # every command sent (also handy for tests)
        self._mask = W.circle_mask(self.size)

        from soundwatch.ui.debug import DebugPanel  # local import: debug imports this module's helpers

        self.debug = DebugPanel(cfg, self.theme)

    # ------------------------------------------------------------ properties
    @property
    def alert_classes(self) -> list[SoundClass]:
        return [c for c in self.cfg.classes if c.id != self.cfg.background_id]

    def _sc(self, class_id: int) -> SoundClass:
        return self.cfg.classes[class_id]

    # -------------------------------------------------------------- commands
    def _send(self, cmd: object) -> None:
        self.sent.append(cmd)
        if self.in_queue is not None:
            try:
                self.in_queue.put_nowait(cmd)
            except queue.Full:  # pragma: no cover - engine queue is unbounded in practice
                pass

    # -------------------------------------------------------------- messages
    def handle_message(self, msg: object, now: float | None = None) -> None:
        if now is not None:
            self.now = now
        if isinstance(msg, StatusUpdate):
            self.status = msg
            self.debug.on_status(msg)
        elif isinstance(msg, AlertEvent):
            self._on_alert(msg)
        elif isinstance(msg, EngineStopped):
            self.engine_stopped = msg.reason

    def _on_alert(self, ev: AlertEvent) -> None:
        if not 0 <= ev.class_id < len(self.cfg.classes):
            return
        sc = self._sc(ev.class_id)
        self.event_log.appendleft(ev)
        if not ev.repeat:
            self.history.appendleft(HistoryEntry(ev.class_id, ev.confidence, ev.t_wall, ev.label, ev.icon))
            self.history_scroll = 0.0
        if sc.level == Level.DANGER:
            self.pending_danger[ev.class_id] = ev
        cur = self.active
        if cur is None or sc.level >= self._sc(cur.event.class_id).level:
            # highest level wins; at equal level the most recent one is shown
            self._show(ev)
            self.vib.start(sc.vibration, self.now, sc.id)
            self.vib_class = sc
        # else: a lower-level alert while a higher one is shown -> history only

    def _show(self, ev: AlertEvent) -> None:
        if self.screen != "alert":
            self.return_screen = self.screen
        self.screen = "alert"
        self.active = ActiveAlert(ev, self.now)

    def _dismiss(self) -> None:
        self.active = None
        if self.pending_danger:  # another unacknowledged DANGER is still waiting
            ev = max(self.pending_danger.values(), key=lambda e: e.t_wall)
            self.screen = "alert"
            self.active = ActiveAlert(ev, self.now)
            return
        self.screen = self.return_screen if self.return_screen != "alert" else "idle"

    def ack(self) -> None:
        """Space / click on the face."""
        if self.screen == "alert" and self.active is not None:
            sc = self._sc(self.active.event.class_id)
            if sc.level == Level.DANGER:
                self._send(AckCommand(sc.id))
                self.pending_danger.pop(sc.id, None)
                if self.vib.class_id == sc.id:
                    self.vib.stop()
            self._dismiss()
        elif self.screen == "idle" and self.pending_danger:
            ev = max(self.pending_danger.values(), key=lambda e: e.t_wall)
            self._show(ev)

    def go(self, screen: str) -> None:
        """Navigate (H / S / Esc). Leaving an alert keeps unacknowledged DANGER pending."""
        assert screen in SCREENS
        if self.screen == "alert":
            self.active = None
        self.screen = screen
        self.return_screen = "idle"

    def toggle_class(self, class_id: int) -> None:
        self.enabled[class_id] = not self.enabled[class_id]
        self._send(SettingsCommand(class_id, self.enabled[class_id]))
        if not self.enabled[class_id]:
            self.pending_danger.pop(class_id, None)

    def cycle_mode(self) -> None:
        """M: next context mode (config `modes:`; no-op without modes), sent as a ModeCommand."""
        m = self.cfg.modes
        if m is None:
            return
        names = m.names
        cur = self.status.mode_selector if self.status is not None and self.status.mode_selector else None
        cur = cur or getattr(self, "_mode_sent", None) or m.default
        nxt = names[(names.index(cur) + 1) % len(names)] if cur in names else names[0]
        self._mode_sent = nxt
        self._send(ModeCommand(nxt))

    # ---------------------------------------------------------------- events
    def face_coords(self, pos: tuple[int, int]) -> tuple[float, float]:
        return ((pos[0] - self.face_pos[0]) / self.scale, (pos[1] - self.face_pos[1]) / self.scale)

    def _on_face(self, pos: tuple[int, int]) -> bool:
        x, y = self.face_coords(pos)
        c = self.size / 2
        return (x - c) ** 2 + (y - c) ** 2 <= c * c

    def handle_event(self, ev: pygame.event.Event, now: float | None = None) -> None:
        if now is not None:
            self.now = now
        if ev.type == pygame.QUIT:
            self.quit_requested = True
        elif ev.type == pygame.KEYDOWN:
            self._on_key(ev.key)
        elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1 and self._on_face(ev.pos):
            if self.screen == "settings":
                hit = self._settings_hit(self.face_coords(ev.pos))
                if hit is not None:
                    self.settings_sel = hit
                    self.toggle_class(self.alert_classes[hit].id)
            else:
                self.ack()
        elif ev.type == pygame.MOUSEWHEEL:
            self._scroll(-ev.y)

    def _on_key(self, key: int) -> None:
        if key in (pygame.K_q,):
            self.quit_requested = True
        elif key == pygame.K_ESCAPE:
            if self.screen == "alert" and self.active is not None \
                    and self._sc(self.active.event.class_id).level != Level.DANGER:
                self._dismiss()
                self.screen = "idle"
            else:
                self.go("idle")
        elif key == pygame.K_h:
            self.go("idle" if self.screen == "history" else "history")
        elif key == pygame.K_s:
            self.go("idle" if self.screen == "settings" else "settings")
        elif key == pygame.K_m:
            self.cycle_mode()
        elif key in (pygame.K_SPACE, pygame.K_RETURN, pygame.K_KP_ENTER):
            if self.screen == "settings":
                self.toggle_class(self.alert_classes[self.settings_sel].id)
            else:
                self.ack()
        elif key in (pygame.K_UP, pygame.K_DOWN):
            self._scroll(-1 if key == pygame.K_UP else 1)

    def _scroll(self, step: int) -> None:
        if self.screen == "history":
            max_scroll = max(0.0, len(self.history) * HIST_ROW - HIST_VIEW)
            self.history_scroll = min(max_scroll, max(0.0, self.history_scroll + step * HIST_ROW))
        elif self.screen == "settings":
            self.settings_sel = min(len(self.alert_classes) - 1, max(0, self.settings_sel + step))

    # ---------------------------------------------------------------- update
    def update(self, now: float, wall: float | None = None) -> None:
        self.now = now
        self.wall = time.time() if wall is None else wall
        if self.active is not None and self.screen == "alert":
            sc = self._sc(self.active.event.class_id)
            if sc.level != Level.DANGER and now - self.active.shown_at >= self.cfg.ui.alert_timeout_s:
                self._dismiss()

    # ------------------------------------------------------------------ draw
    def draw(self, surf: pygame.Surface) -> None:
        surf.fill(WINDOW_BG)
        face = self.render_face()
        vstate = self.vib.state(self.now)
        dx = dy = 0
        if vstate.on:
            dx, dy = self.rng.randint(-JITTER_PX, JITTER_PX), self.rng.randint(-JITTER_PX, JITTER_PX)
        fx, fy = self.face_pos[0] + dx, self.face_pos[1] + dy
        big = pygame.transform.scale(face, (self.face_px, self.face_px))
        # bezel
        cx, cy, r = fx + self.face_px // 2, fy + self.face_px // 2, self.face_px // 2
        surf.blit(big, (fx, fy))
        W.draw_ring(surf, (14, 15, 19), (cx, cy), r + 14, r + 9)
        W.draw_ring(surf, BEZEL, (cx, cy), r + 9, r + 3)
        W.draw_ring(surf, (34, 36, 42), (cx, cy), r + 3, r - 1)
        if vstate.on:
            col = self.vib_class.color if self.vib_class else WHITE
            gx, gy = self.face_pos[0] + self.face_px + 2, self.face_pos[1] + 18
            W.aa_circle(surf, darken(col, 0.3), (gx, gy), 17)
            W.draw_vibration_glyph(surf, (gx, gy), WHITE, 1.3)
        # vibration timeline
        tl = pygame.Rect(self.face_pos[0], self.face_pos[1] + self.face_px + 18, self.face_px, 58)
        sc = self.vib_class
        title = f"Rung: {sc.label_vi}" if sc else "Rung: —"
        W.draw_pattern_timeline(surf, self.theme, tl, sc.vibration if sc else (), vstate,
                                sc.color if sc else TEXT_FAINT, title)
        if not self.show_debug:
            self._draw_hints(surf, pygame.Rect(0, self.window_size[1] - 20, self.window_size[0], 20))
        else:
            self.debug.draw(surf, self.panel_rect, self)

    def _draw_hints(self, surf: pygame.Surface, rect: pygame.Rect) -> None:
        txt = "Space: xác nhận · H: lịch sử · S: cài đặt · Esc: về · Q: thoát"
        if self.cfg.modes is not None:
            txt = txt.replace(" · Esc", " · M: chế độ · Esc")
        img = self.theme.text(txt, 12, TEXT_FAINT, bold=False)
        surf.blit(img, img.get_rect(center=rect.center))

    def render_face(self) -> pygame.Surface:
        # draw on an opaque surface (antialiased primitives need an opaque destination), then
        # copy into a per-pixel-alpha surface and cut it round with the mask
        s = pygame.Surface((self.size, self.size))
        getattr(self, f"_face_{self.screen}")(s)
        out = pygame.Surface((self.size, self.size), pygame.SRCALPHA)
        out.blit(s, (0, 0))
        out.blit(self._mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        return out

    # ---- idle
    def _face_idle(self, s: pygame.Surface) -> None:
        th, S = self.theme, self.size
        c = (S / 2, S / 2)
        s.fill(FACE_BG)
        rms = self.status.rms_dbfs if self.status else RMS_FLOOR_DB
        gated = self.status.gated if self.status else True
        v = (rms - RMS_FLOOR_DB) / -RMS_FLOOR_DB
        W.draw_gauge(s, c, S / 2 - 5, 7, v, (90, 96, 110) if gated else ACCENT)
        # gate threshold tick
        g = (self.cfg.audio.gate_dbfs - RMS_FLOOR_DB) / -RMS_FLOOR_DB
        a = -135 + 270 * g
        p0, p1 = W.polar(c, S / 2 - 3, a), W.polar(c, S / 2 - 14, a)
        pygame.draw.line(s, (200, 204, 214), p0, p1, 2)
        lt = time.localtime(self.wall)
        W.blit_center(s, th.text(time.strftime("%H:%M", lt), 62, WHITE), S / 2, 92)
        date = f"{WEEKDAYS_VI[lt.tm_wday]}, {lt.tm_mday:02d}/{lt.tm_mon:02d}"
        W.blit_center(s, th.text(date, 15, TEXT_DIM, bold=False), S / 2, 133)
        # listening line
        stopped = self.engine_stopped is not None
        txt = "Đã dừng nghe" if stopped else "Đang nghe…"
        if not stopped and self.status is not None and self.status.mode:
            txt = MODE_NAMES_VI.get(self.status.mode, self.status.mode) + (
                " (tự động)" if self.status.mode_selector == "auto" else "")
        col = TEXT_FAINT if stopped else (ACCENT if not gated else TEXT_DIM)
        img = th.text(txt, 16, col)
        ic = th.icon(self._sc(self.cfg.background_id).icon, 18, col)
        total = ic.get_width() + 6 + img.get_width()
        x0 = S / 2 - total / 2
        s.blit(ic, (round(x0), 165 - ic.get_height() // 2))
        s.blit(img, img.get_rect(midleft=(round(x0 + ic.get_width() + 6), 165)))
        # bottom: pending danger, else last event
        if self.pending_danger:
            ev = max(self.pending_danger.values(), key=lambda e: e.t_wall)
            sc = self._sc(ev.class_id)
            pill = pygame.Rect(0, 0, 150, 24)
            pill.center = (S // 2, 199)
            flash = int((self.now * 2) % 2) == 0
            pygame.draw.rect(s, sc.color if flash else darken(sc.color, 0.4), pill, border_radius=12)
            txt = f"Chưa xác nhận: {event_label(ev, sc)}"
            if ev.label:  # mode label overrides can be long (the configured labels keep their old look)
                txt = _ellipsize(th, txt, 11, 144)
            W.blit_center(s, th.text(txt, 11, WHITE), *pill.center)
        elif self.history:
            h = self.history[0]
            sc = self._sc(h.class_id)
            ic = th.icon(h.icon or sc.icon, 16, sc.color)
            txt = f"{h.label or sc.label_vi} · {hms(h.t_wall)[:5]}"
            if h.label:
                txt = _ellipsize(th, txt, 12, 150, bold=False)
            t = th.text(txt, 12, TEXT_DIM, bold=False)
            total = ic.get_width() + 5 + t.get_width()
            x0 = S / 2 - total / 2
            s.blit(ic, (round(x0), 199 - 8))
            s.blit(t, t.get_rect(midleft=(round(x0 + ic.get_width() + 5), 199)))

    # ---- alert
    def _face_alert(self, s: pygame.Surface) -> None:
        th, S = self.theme, self.size
        if self.active is None:
            self._face_idle(s)
            return
        ev = self.active.event
        sc = self._sc(ev.class_id)
        base = th.alert_bg(sc)
        danger = sc.level == Level.DANGER
        bg = base
        if danger:  # 2 Hz flash: 0.25 s bright / 0.25 s dark
            phase = ((self.now - self.active.shown_at) * 2.0) % 1.0
            bg = base if phase < 0.5 else darken(sc.color, 0.82)
        s.fill(bg)
        c = (S / 2, S / 2)
        W.draw_gauge(s, c, S / 2 - 5, 7, ev.confidence, WHITE, track=darken(base, 0.45))
        icon = th.icon(event_icon(ev, sc), 96)
        s.blit(icon, icon.get_rect(midtop=(S // 2, 22)))
        size, lines = W.fit_lines(th, event_label(ev, sc), 196, (30, 28, 26, 24, 22, 20), two_line_max=24)
        if ev.label and th.text_width(lines[-1], size) > 196:  # long mode label (vn-v3): up to 3 lines
            size, lines = 18, _wrap(th, ev.label, 18, 190)[:3]
        top = 122 if len(lines) == 1 else 119
        bottom = W.draw_lines_centered(s, th, lines, size, S / 2, top)
        meta = th.text(f"{hms(ev.t_wall)}  ·  {ev.confidence * 100:.0f}%", 14, lighten(base, 0.75))
        W.blit_center(s, meta, S / 2, bottom + 6)
        if danger:
            hint = th.text("Nhấn để xác nhận", 12, WHITE)
            pill = hint.get_rect(center=(S // 2, 212)).inflate(16, 6)
            pygame.draw.rect(s, darken(sc.color, 0.6), pill, border_radius=pill.height // 2)
            W.blit_center(s, hint, *pill.center)
        else:
            lv = th.text(LEVEL_NAMES_VI[sc.level], 12, lighten(base, 0.6), bold=False)
            W.blit_center(s, lv, S / 2, 212)

    # ---- history
    def _title(self, s: pygame.Surface, text: str, sub: str | None = None) -> None:
        W.blit_center(s, self.theme.text(text, 18), self.size / 2, 26)
        if sub:
            W.blit_center(s, self.theme.text(sub, 11, TEXT_FAINT, bold=False), self.size / 2, 43)

    def _scrollbar(self, s: pygame.Surface, frac0: float, frac1: float) -> None:
        c = (self.size / 2, self.size / 2)
        W.draw_arc(s, TRACK, c, self.size / 2 - 4, 4, 60, 120)
        W.draw_arc(s, TEXT_DIM, c, self.size / 2 - 4, 4, 60 + 60 * frac0, 60 + 60 * frac1)

    def _face_history(self, s: pygame.Surface) -> None:
        th, S = self.theme, self.size
        s.fill(FACE_BG)
        n = len(self.history)
        self._title(s, "Lịch sử", f"{n}/{self.cfg.ui.history_len} sự kiện gần nhất")
        if not self.history:
            W.blit_center(s, th.text("Chưa có sự kiện", 15, TEXT_DIM, bold=False), S / 2, S / 2 + 6)
            return
        clip = pygame.Rect(0, HIST_TOP, S, HIST_VIEW)
        s.set_clip(clip)
        x0 = 34
        for i, h in enumerate(self.history):
            y = HIST_TOP + i * HIST_ROW - self.history_scroll
            if y + HIST_ROW < clip.top or y > clip.bottom:
                continue
            sc = self._sc(h.class_id)
            W.aa_circle(s, darken(sc.color, 0.55), (x0 + 13, round(y + HIST_ROW / 2)), 15)
            s.blit(th.icon(h.icon or sc.icon, 20, WHITE), (x0 + 3, round(y + HIST_ROW / 2) - 10))
            fs, label = _fit_one_line(th, h.label or sc.label_vi, (14, 13, 12), 136)
            s.blit(th.text(label, fs, WHITE), (x0 + 34, round(y + 3 + (14 - fs) / 2)))
            meta = f"{hms(h.t_wall)} · {h.confidence * 100:.0f}%"
            s.blit(th.text(meta, 12, TEXT_DIM, bold=False), (x0 + 34, round(y + 20)))
            if i < n - 1:
                pygame.draw.line(s, (30, 33, 40), (x0 + 34, round(y + HIST_ROW - 1)),
                                 (S - 40, round(y + HIST_ROW - 1)))
        s.set_clip(None)
        total = n * HIST_ROW
        if total > HIST_VIEW:
            self._scrollbar(s, self.history_scroll / total, (self.history_scroll + HIST_VIEW) / total)

    # ---- settings
    def _settings_layout(self) -> list[tuple[int, pygame.Rect]]:
        n = len(self.alert_classes)
        # keep the selected row visible
        sel_top = self.settings_sel * SET_ROW
        if sel_top < self.settings_scroll:
            self.settings_scroll = sel_top
        if sel_top + SET_ROW > self.settings_scroll + SET_VIEW:
            self.settings_scroll = sel_top + SET_ROW - SET_VIEW
        out = []
        for i in range(n):
            y = SET_TOP + i * SET_ROW - self.settings_scroll
            out.append((i, pygame.Rect(30, round(y), self.size - 60, SET_ROW - 2)))
        return out

    def _settings_hit(self, pos: tuple[float, float]) -> int | None:
        if not SET_TOP <= pos[1] <= SET_TOP + SET_VIEW:
            return None
        for i, r in self._settings_layout():
            if r.collidepoint(pos):
                return i
        return None

    def _face_settings(self, s: pygame.Surface) -> None:
        th, S = self.theme, self.size
        s.fill(FACE_BG)
        n_on = sum(self.enabled[c.id] for c in self.alert_classes)
        self._title(s, "Cài đặt", f"Đang bật {n_on}/{len(self.alert_classes)} loại cảnh báo")
        clip = pygame.Rect(0, SET_TOP, S, SET_VIEW)
        s.set_clip(clip)
        for i, r in self._settings_layout():
            if r.bottom < clip.top or r.top > clip.bottom:
                continue
            sc = self.alert_classes[i]
            on = self.enabled[sc.id]
            if i == self.settings_sel:
                pygame.draw.rect(s, (38, 42, 52), r, border_radius=8)
            ic_col = sc.color if on else TEXT_FAINT
            s.blit(th.icon(sc.icon, 20, ic_col), (r.x + 6, r.centery - 10))
            col = WHITE if on else TEXT_FAINT
            max_w = r.width - 74
            fs, lines = W.fit_lines(th, sc.label_vi, max_w, (13, 12, 11), two_line_max=11)
            if len(lines) == 1:
                img = th.text(_ellipsize(th, lines[0], fs, max_w), fs, col)
                s.blit(img, img.get_rect(midleft=(r.x + 32, r.centery)))
            else:
                for k, ln in enumerate(lines):
                    img = th.text(ln, fs, col)
                    s.blit(img, img.get_rect(midleft=(r.x + 32, r.centery + (k - 0.5) * 12)))
            W.draw_toggle(s, pygame.Rect(r.right - 38, r.centery - 8, 32, 16), on, ACCENT)
        s.set_clip(None)
        total = len(self.alert_classes) * SET_ROW
        if total > SET_VIEW:
            self._scrollbar(s, self.settings_scroll / total, (self.settings_scroll + SET_VIEW) / total)


# list geometry (face pixels)
HIST_TOP, HIST_ROW, HIST_VIEW = 54, 38, 152
SET_TOP, SET_ROW, SET_VIEW = 54, 30, 152


def _fit_one_line(theme: Theme, text: str, sizes: tuple[int, ...], max_w: int) -> tuple[int, str]:
    """Largest size at which text fits on one line; ellipsized at the smallest size otherwise."""
    for size in sizes:
        if theme.text_width(text, size) <= max_w:
            return size, text
    return sizes[-1], _ellipsize(theme, text, sizes[-1], max_w)


def _wrap(theme: Theme, text: str, size: int, max_w: int) -> list[str]:
    """Greedy word wrap at one font size."""
    lines: list[str] = []
    for w in text.split():
        if lines and theme.text_width(lines[-1] + " " + w, size, True) <= max_w:
            lines[-1] += " " + w
        else:
            lines.append(w)
    return lines


def _ellipsize(theme: Theme, text: str, size: int, max_w: int, bold: bool = True) -> str:
    if theme.text_width(text, size, bold) <= max_w:
        return text
    while text and theme.text_width(text + "…", size, bold) > max_w:
        text = text[:-1]
    return text.rstrip() + "…"
