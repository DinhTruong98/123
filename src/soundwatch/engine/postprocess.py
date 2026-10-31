"""Probability smoothing + decision logic -> AlertEvents (plan section 4.3). Owner: WP5.

Rules (evaluated once per hop, i.e. per `update(probs, t_audio)` call):

1. Smoothing: `smoothed` = mean of the last `smooth_n` probability vectors (fewer at start-up).
2. Hit: class c is "above" at a hop if smoothed[c] >= threshold[c] (and, only if the optional
   `cfg.bg_margin` is set, smoothed[c] - smoothed[background] >= bg_margin). Class c (not background)
   is *triggered* if it was above in >= k of the last n hops (`k_of_n`, fewer hops at start-up
   count as not-above).
3. Cooldown: a class never emits (fresh or repeat) less than `cooldown_s` after its last emission.
4. Fresh alert (repeat=False): emitted when the class is triggered, enabled, outside its cooldown
   and (DANGER only) not currently in an *active occurrence* (see 5).
4b. DANGER fast path (only if the optional `cfg.danger_fast_threshold` is set): a DANGER class also
   counts as triggered at a hop where smoothed[c] >= danger_fast_threshold (and, if `bg_margin` is
   set, smoothed[c] - smoothed[background] >= bg_margin), without waiting for k-of-n. All other
   conditions of rule 4 still apply (enabled, cooldown, no active occurrence, acked or not). It
   changes nothing else: hit history, last-above time, repeats, ack and clear are as in 2, 3, 5.
   Suggested value: 255/256 = 0.99609375, the saturation level of the int8 output (scale 1/256,
   zero point -128), compared in float32. WARN / INFO classes are not affected.
5. DANGER classes: a fresh alert starts an active occurrence that is *pending ack*. While pending,
   the alert is re-emitted with repeat=True every max(danger_repeat_s, cooldown_s) seconds after the
   previous emission, whether or not the sound is still triggered; no fresh alerts are emitted.
   `ack(c)` clears the pending repeat. The occurrence (pending or acked) ends once smoothed[c] has
   been below threshold continuously for `danger_clear_s` (measured from the last hop where it was
   above). After an ack, the same occurrence does not alert again; a new alert needs the occurrence
   to have ended first (so acking a fire alarm that keeps sounding does not re-alert every
   cooldown_s).
6. Disabled classes (enabled=False) never emit; disabling clears their occurrence state.
7. Background (last class) never emits.
8. All classes triggering in the same hop emit; events are ordered by level desc, confidence desc.
   `confidence` is the smoothed probability at the emitting hop; for repeats it is the peak
   smoothed probability of the occurrence so far (the sound may already have stopped).

9. Context modes (only if a `modes` config is given, see config.ModesConfig; default: none =
   all of the above unchanged). The active mode switches classes off (as `set_enabled(False)`, but
   kept apart from the user's own settings) and overrides class levels (e.g. car_horn DANGER ->
   INFO in `street`: no fast path, no repeats). Level/enable changes take effect at the next hop; a
   class losing DANGER or being disabled has its occurrence cleared. `set_mode(name)` switches at
   runtime; mode 'auto' picks `street` while the running median of the per-hop RMS (`rms_dbfs`
   passed to `update`) over `auto.window_s` is above `auto.street_above_dbfs`, else `indoor`.
   Every switch is appended to `mode_log` as (t_audio, old, new, reason) and logged.

Firmware port: all state is fixed-size (ring of smooth_n vectors + n-bit hit history per class).
"""

from __future__ import annotations

import logging
import time
from collections import deque

import numpy as np

from soundwatch.config import AUTO_HIGH_MODE, AUTO_LOW_MODE, AUTO_MODE, ModesConfig, PostprocessConfig
from soundwatch.types import AlertEvent, Level, SoundClass

log = logging.getLogger(__name__)


class PostProcessor:
    def __init__(self, classes: list[SoundClass] | tuple[SoundClass, ...], cfg: PostprocessConfig,
                 modes: ModesConfig | None = None, mode: str | None = None):
        if not classes:
            raise ValueError("no classes")
        self.classes = tuple(classes)
        self.cfg = cfg
        self.num_classes = len(self.classes)
        self.background_id = self.num_classes - 1
        self.smooth_n = max(1, int(cfg.smooth_n))
        self.k, self.n = (int(v) for v in cfg.k_of_n)
        self._thr = np.array([c.threshold for c in self.classes], dtype=np.float32)
        self._base_level = np.array([int(c.level) for c in self.classes], dtype=np.int64)
        self._level = self._base_level.copy()  # effective level (after mode overrides)
        self._is_danger = self._level == int(Level.DANGER)
        self._cooldown = np.array([c.cooldown_s for c in self.classes], dtype=np.float64)
        self._enabled = np.array([c.enabled for c in self.classes], dtype=bool)  # user settings
        self._mode_off = np.zeros(self.num_classes, dtype=bool)  # switched off by the mode
        self._hist = np.zeros((self.smooth_n, self.num_classes), dtype=np.float32)
        self._hits = np.zeros((self.n, self.num_classes), dtype=bool)
        self.smoothed = np.zeros(self.num_classes, dtype=np.float32)
        self.modes = modes
        self.mode: str | None = None  # active named mode (None = no modes configured)
        self.mode_selector: str | None = None  # requested mode ('auto' or a named mode)
        self.mode_log: list[tuple[float, str | None, str, str]] = []
        self._keys = [c.key for c in self.classes]
        self._rms: deque[tuple[float, float]] = deque()
        self._auto_t0: float | None = None
        self.reset()
        if modes is not None:
            self.set_mode(mode or modes.default, t_audio=0.0, reason="start")
        elif mode is not None:
            raise ValueError(f"mode {mode!r} requested but no modes are configured")

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        self._hist.fill(0.0)
        self._hits.fill(False)
        self.smoothed.fill(0.0)
        self._n_seen = 0
        self._hist_pos = 0
        self._hits_pos = 0
        self._last_emit = np.full(self.num_classes, -np.inf)
        self._last_above = np.full(self.num_classes, -np.inf)
        self._active = np.zeros(self.num_classes, dtype=bool)   # DANGER occurrence in progress
        self._pending = np.zeros(self.num_classes, dtype=bool)  # ... and not yet acked
        self._peak = np.zeros(self.num_classes, dtype=np.float32)  # max smoothed prob in occurrence
        self._rms.clear()
        self._auto_t0 = None

    def ack(self, class_id: int) -> None:
        if 0 <= class_id < self.num_classes:
            self._pending[class_id] = False

    def set_enabled(self, class_id: int, enabled: bool) -> None:
        if 0 <= class_id < self.num_classes:
            self._enabled[class_id] = bool(enabled)
            if not enabled:
                self._active[class_id] = False
                self._pending[class_id] = False

    def is_enabled(self, class_id: int) -> bool:
        """User setting (the active mode may still switch the class off, see `is_active_class`)."""
        return bool(self._enabled[class_id])

    def is_active_class(self, class_id: int) -> bool:
        """Enabled by the user and not switched off by the current mode."""
        return bool(self._enabled[class_id] and not self._mode_off[class_id])

    def level(self, class_id: int) -> Level:
        """Effective level of a class (mode overrides applied)."""
        return Level(int(self._level[class_id]))

    # ------------------------------------------------------------------ modes
    def set_mode(self, name: str, t_audio: float | None = None, reason: str = "manual") -> None:
        """Switch to a named mode or to 'auto' (see module doc, rule 9)."""
        if self.modes is None:
            raise ValueError("no modes configured")
        if name not in self.modes.names:
            raise ValueError(f"unknown mode {name!r} (configured: {self.modes.names})")
        self.mode_selector = name
        if name == AUTO_MODE:
            self._rms.clear()
            self._auto_t0 = None
            target = self.mode if self.mode is not None else AUTO_LOW_MODE
            self._apply_mode(target, t_audio, f"{reason}: auto")
        else:
            self._apply_mode(name, t_audio, reason)

    def _apply_mode(self, name: str, t_audio: float | None, reason: str) -> None:
        assert self.modes is not None
        old = self.mode
        if old == name and self.mode_log:
            return
        spec = self.modes.spec(name)
        off = np.zeros(self.num_classes, dtype=bool)
        for k in spec.disable:
            off[self._keys.index(k)] = True
        level = self._base_level.copy()
        for k, v in spec.level:
            level[self._keys.index(k)] = int(v)
        danger = level == int(Level.DANGER)
        # occurrences of classes that are switched off or stop being DANGER end now
        stop = (off & ~self._mode_off) | (self._is_danger & ~danger)
        self._active[stop] = False
        self._pending[stop] = False
        self._mode_off, self._level, self._is_danger = off, level, danger
        self.mode = name
        t = float(t_audio) if t_audio is not None else float("nan")
        self.mode_log.append((t, old, name, reason))
        log.info("mode %s -> %s at t=%.1fs (%s)", old, name, t, reason)

    def _auto_update(self, t: float, rms_dbfs: float) -> None:
        a = self.modes.auto if self.modes is not None else None
        if a is None:
            return
        self._rms.append((t, float(rms_dbfs)))
        while self._rms and self._rms[0][0] <= t - a.window_s:
            self._rms.popleft()
        if self._auto_t0 is None:
            self._auto_t0 = t
        if t - self._auto_t0 + 1e-9 < min(a.min_s, a.window_s):
            return
        med = float(np.median([v for _, v in self._rms]))
        h = a.hysteresis_db / 2.0
        if self.mode != AUTO_HIGH_MODE and med > a.street_above_dbfs + h:
            self._apply_mode(AUTO_HIGH_MODE, t, f"auto: median RMS {med:.1f} dBFS")
        elif self.mode != AUTO_LOW_MODE and med < a.street_above_dbfs - h:
            self._apply_mode(AUTO_LOW_MODE, t, f"auto: median RMS {med:.1f} dBFS")

    def is_pending(self, class_id: int) -> bool:
        """True while a DANGER alert of this class waits for ack (and is being repeated)."""
        return bool(self._pending[class_id])

    # ------------------------------------------------------------------ update
    def update(self, probs: np.ndarray, t_audio: float, t_wall: float | None = None,
               rms_dbfs: float | None = None) -> list[AlertEvent]:
        """`rms_dbfs` (level of the analysed window) is only used by the 'auto' mode."""
        p = np.asarray(probs, dtype=np.float32).reshape(-1)
        if p.shape[0] != self.num_classes:
            raise ValueError(f"expected {self.num_classes} probs, got {p.shape[0]}")
        t = float(t_audio)
        tw = time.time() if t_wall is None else float(t_wall)
        if self.mode_selector == AUTO_MODE and rms_dbfs is not None:
            self._auto_update(t, rms_dbfs)

        self._hist[self._hist_pos] = p
        self._hist_pos = (self._hist_pos + 1) % self.smooth_n
        self._n_seen += 1
        m = min(self._n_seen, self.smooth_n)
        self.smoothed[:] = self._hist.sum(axis=0) / m  # unfilled rows are zero

        above = self.smoothed >= self._thr
        margin = getattr(self.cfg, "bg_margin", None)
        if margin is not None:
            above &= (self.smoothed - self.smoothed[self.background_id]) >= float(margin)
        self._hits[self._hits_pos] = above
        self._hits_pos = (self._hits_pos + 1) % self.n
        triggered = self._hits.sum(axis=0) >= self.k
        fast = getattr(self.cfg, "danger_fast_threshold", None)
        if fast is not None:
            fast_hit = self._is_danger & (self.smoothed >= np.float32(fast))
            if margin is not None:
                fast_hit &= (self.smoothed - self.smoothed[self.background_id]) >= float(margin)
            triggered |= fast_hit
        self._last_above[above] = t

        events: list[AlertEvent] = []
        for c in range(self.num_classes):
            if c == self.background_id:
                continue
            if not self._enabled[c] or self._mode_off[c]:
                continue
            danger = bool(self._is_danger[c])
            if danger and self._active[c] and t - self._last_above[c] >= self.cfg.danger_clear_s:
                self._active[c] = False
                self._pending[c] = False
            conf = float(self.smoothed[c])
            if danger and self._active[c]:
                self._peak[c] = max(self._peak[c], conf)
            since = t - self._last_emit[c]
            if since < self._cooldown[c]:
                continue
            if danger and self._active[c]:
                if self._pending[c] and since >= self.cfg.danger_repeat_s:
                    events.append(AlertEvent(c, float(self._peak[c]), t, tw, repeat=True))
                    self._last_emit[c] = t
                continue
            if triggered[c]:
                events.append(AlertEvent(c, conf, t, tw, repeat=False))
                self._last_emit[c] = t
                if danger:
                    self._active[c] = True
                    self._pending[c] = True
                    self._peak[c] = conf
        events.sort(key=lambda e: (-int(self._level[e.class_id]), -e.confidence))
        return events
