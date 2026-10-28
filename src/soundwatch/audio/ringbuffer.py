"""Fixed-size ring buffer that never reallocates (mirrors firmware). Owner: WP5.

Firmware equivalent: a static int16[16000] circular buffer filled by the PDM/I2S DMA callback.
"""

from __future__ import annotations

import numpy as np


class RingBuffer:
    """Preallocated float32 circular buffer of `capacity` samples.

    `write` accepts blocks of any size (also larger than `capacity`, in which case only the last
    `capacity` samples are kept). `latest` copies the most recent `capacity` samples in
    chronological order into `out` (or a new array if `out` is None). No allocation happens in
    `write`, nor in `latest` when `out` is given.
    """

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self._cap = int(capacity)
        self._buf = np.zeros(self._cap, dtype=np.float32)
        self._pos = 0  # next write index == index of the oldest sample once full
        self._total = 0

    @property
    def capacity(self) -> int:
        return self._cap

    @property
    def total_written(self) -> int:
        return self._total

    def clear(self) -> None:
        self._buf.fill(0.0)
        self._pos = 0
        self._total = 0

    def write(self, block: np.ndarray) -> None:
        x = np.asarray(block).reshape(-1)
        n = x.shape[0]
        if n == 0:
            return
        cap = self._cap
        if n >= cap:
            self._buf[:] = x[n - cap:]
            self._pos = 0
        else:
            first = min(n, cap - self._pos)
            self._buf[self._pos:self._pos + first] = x[:first]
            rest = n - first
            if rest:
                self._buf[:rest] = x[first:]
            self._pos = (self._pos + n) % cap
        self._total += n

    def latest(self, out: np.ndarray | None = None) -> np.ndarray:
        """Most recent `capacity` samples in chronological order (zero-padded at the start if not full)."""
        if out is None:
            out = np.empty(self._cap, dtype=np.float32)
        elif out.shape != (self._cap,):
            raise ValueError(f"out must have shape ({self._cap},), got {out.shape}")
        tail = self._cap - self._pos
        out[:tail] = self._buf[self._pos:]
        out[tail:] = self._buf[:self._pos]
        return out
