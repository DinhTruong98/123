"""Fixed-size ring buffer that never reallocates (mirrors firmware). Owner: WP5."""

from __future__ import annotations

import numpy as np


class RingBuffer:
    def __init__(self, capacity: int):
        raise NotImplementedError("WP5")

    def write(self, block: np.ndarray) -> None:
        raise NotImplementedError("WP5")

    @property
    def total_written(self) -> int:
        raise NotImplementedError("WP5")

    def latest(self, out: np.ndarray | None = None) -> np.ndarray:
        """Most recent `capacity` samples in chronological order (zero-padded at the start if not full)."""
        raise NotImplementedError("WP5")
