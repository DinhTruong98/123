"""Probability smoothing + decision logic -> AlertEvents (plan section 4.3). Owner: WP5."""

from __future__ import annotations

import numpy as np

from soundwatch.config import PostprocessConfig
from soundwatch.types import AlertEvent, SoundClass


class PostProcessor:
    def __init__(self, classes: list[SoundClass] | tuple[SoundClass, ...], cfg: PostprocessConfig):
        raise NotImplementedError("WP5")

    def update(self, probs: np.ndarray, t_audio: float) -> list[AlertEvent]:
        raise NotImplementedError("WP5")

    def ack(self, class_id: int) -> None:
        raise NotImplementedError("WP5")

    def set_enabled(self, class_id: int, enabled: bool) -> None:
        raise NotImplementedError("WP5")

    def reset(self) -> None:
        raise NotImplementedError("WP5")
