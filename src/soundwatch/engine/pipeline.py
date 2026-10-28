"""Real-time engine: source -> ring buffer -> gate -> features -> classifier -> postprocess. Owner: WP5.

Runs in a worker thread. Puts StatusUpdate / AlertEvent / EngineStopped on `out_queue`; reads
AckCommand / SettingsCommand from `in_queue`.
"""

from __future__ import annotations

import queue

from soundwatch.audio.sources import AudioSource
from soundwatch.config import Config
from soundwatch.engine.classifier import Classifier


class Engine:
    def __init__(self, cfg: Config, source: AudioSource, classifier: Classifier, stats: dict,
                 out_queue: queue.Queue, in_queue: queue.Queue | None = None,
                 event_log_path=None):
        raise NotImplementedError("WP5")

    def start(self) -> None:
        """Start worker thread."""
        raise NotImplementedError("WP5")

    def run(self) -> None:
        """Blocking loop (used by start() and by --fast evaluation)."""
        raise NotImplementedError("WP5")

    def stop(self) -> None:
        raise NotImplementedError("WP5")

    def join(self, timeout: float | None = None) -> None:
        raise NotImplementedError("WP5")
