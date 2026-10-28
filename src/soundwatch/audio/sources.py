"""Audio sources yielding float32 mono blocks at the configured sample rate. Owner: WP5."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol

import numpy as np


class AudioSource(Protocol):
    sample_rate: int

    def blocks(self) -> Iterator[np.ndarray]: ...

    def close(self) -> None: ...


class MicSource:
    """sounddevice.InputStream (import sounddevice lazily: PortAudio may be missing)."""

    def __init__(self, sample_rate: int = 16000, blocksize: int = 1600, device: int | str | None = None):
        raise NotImplementedError("WP5")


class FileSource:
    """WAV/FLAC file -> 16 kHz mono float32 blocks. realtime=True paces blocks at wall-clock speed."""

    def __init__(self, path, sample_rate: int = 16000, blocksize: int = 1600, realtime: bool = True):
        raise NotImplementedError("WP5")


class ArraySource:
    """In-memory array source, for tests."""

    def __init__(self, samples: np.ndarray, sample_rate: int = 16000, blocksize: int = 1600,
                 realtime: bool = False):
        raise NotImplementedError("WP5")
