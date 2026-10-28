"""Probability smoothing + decision logic -> AlertEvents (plan section 4.3). Owner: WP5.

Rules (evaluated once per hop, i.e. per `update(probs, t_audio)` call):

1. Smoothing: `smoothed` = mean of the last `smooth_n` probability vectors (fewer at start-up).
2. Hit: class c is "above" at a hop if smoothed[c] >= threshold[c]. Class c (not background)
   is *triggered* if it was above in >= k of the last n hops (`k_of_n`, fewer hops at start-up
   count as not-above).
3. Cooldown: a class never emits (fresh or repeat) less than `cooldown_s` after its last emission.
4. Fresh alert (repeat=False): emitted when the class is triggered, enabled, outside its cooldown
   and (DANGER only) not currently in an *active occurrence* (see 5).
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

Firmware port: all state is fixed-size (ring of smooth_n vectors + n-bit hit history per class).
"""

from __future__ import annotations

import time

import numpy as np

from soundwatch.config import PostprocessConfig
from soundwatch.types import AlertEvent, Level, SoundClass


class PostProcessor:
    def __init__(self, classes: list[SoundClass] | tuple[SoundClass, ...], cfg: PostprocessConfig):
        if not classes:
            raise ValueError("no classes")
        self.classes = tuple(classes)
        self.cfg = cfg
        self.num_classes = len(self.classes)
        self.background_id = self.num_classes - 1
        self.smooth_n = max(1, int(cfg.smooth_n))
        self.k, self.n = (int(v) for v in cfg.k_of_n)
        self._thr = np.array([c.threshold for c in self.classes], dtype=np.float32)
        self._is_danger = np.array([c.level == Level.DANGER for c in self.classes], dtype=bool)
        self._cooldown = np.array([c.cooldown_s for c in self.classes], dtype=np.float64)
        self._enabled = np.array([c.enabled for c in self.classes], dtype=bool)
        self._hist = np.zeros((self.smooth_n, self.num_classes), dtype=np.float32)
        self._hits = np.zeros((self.n, self.num_classes), dtype=bool)
        self.smoothed = np.zeros(self.num_classes, dtype=np.float32)
        self.reset()

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
        return bool(self._enabled[class_id])

    def is_pending(self, class_id: int) -> bool:
        """True while a DANGER alert of this class waits for ack (and is being repeated)."""
        return bool(self._pending[class_id])

    # ------------------------------------------------------------------ update
    def update(self, probs: np.ndarray, t_audio: float, t_wall: float | None = None) -> list[AlertEvent]:
        p = np.asarray(probs, dtype=np.float32).reshape(-1)
        if p.shape[0] != self.num_classes:
            raise ValueError(f"expected {self.num_classes} probs, got {p.shape[0]}")
        t = float(t_audio)
        tw = time.time() if t_wall is None else float(t_wall)

        self._hist[self._hist_pos] = p
        self._hist_pos = (self._hist_pos + 1) % self.smooth_n
        self._n_seen += 1
        m = min(self._n_seen, self.smooth_n)
        self.smoothed[:] = self._hist.sum(axis=0) / m  # unfilled rows are zero

        above = self.smoothed >= self._thr
        self._hits[self._hits_pos] = above
        self._hits_pos = (self._hits_pos + 1) % self.n
        triggered = self._hits.sum(axis=0) >= self.k
        self._last_above[above] = t

        events: list[AlertEvent] = []
        for c in range(self.num_classes):
            if c == self.background_id:
                continue
            if not self._enabled[c]:
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
        events.sort(key=lambda e: (-int(self.classes[e.class_id].level), -e.confidence))
        return events
