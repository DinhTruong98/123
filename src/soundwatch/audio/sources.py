"""Audio sources yielding float32 mono blocks at the configured sample rate. Owner: WP5.

All sources implement the `AudioSource` protocol: `blocks()` yields 1-D float32 arrays in [-1, 1]
at `sample_rate`; `close()` is idempotent, thread-safe and makes a running `blocks()` iterator end
promptly (the engine calls it from `Engine.stop()`).
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Iterator
from math import gcd
from pathlib import Path
from typing import Protocol

import numpy as np


class AudioSource(Protocol):
    sample_rate: int

    def blocks(self) -> Iterator[np.ndarray]: ...

    def close(self) -> None: ...


def to_mono_float32(x: np.ndarray) -> np.ndarray:
    """(n,) or (n, channels) -> (n,) float32 (channel mean)."""
    x = np.asarray(x)
    if x.ndim == 2:
        x = x.mean(axis=1)
    elif x.ndim != 1:
        raise ValueError(f"expected 1-D or 2-D audio, got shape {x.shape}")
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float32) / float(np.iinfo(x.dtype).max)
    return np.ascontiguousarray(x, dtype=np.float32)


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Polyphase resampling (scipy.signal.resample_poly) -> float32."""
    if sr_in == sr_out:
        return np.asarray(x, dtype=np.float32)
    from scipy.signal import resample_poly

    g = gcd(int(sr_in), int(sr_out))
    y = resample_poly(np.asarray(x, dtype=np.float64), int(sr_out) // g, int(sr_in) // g)
    return y.astype(np.float32)


class MicSource:
    """sounddevice.InputStream (import sounddevice lazily: PortAudio may be missing).

    The PortAudio callback copies each block into an internal queue; `blocks()` yields them.
    If the consumer falls behind, the queue grows (up to `max_queue` blocks, then the oldest
    blocks are dropped and counted in `dropped_blocks`).
    """

    def __init__(self, sample_rate: int = 16000, blocksize: int = 1600, device: int | str | None = None,
                 max_queue: int = 200):
        try:
            import sounddevice as sd
        except (OSError, ImportError) as e:
            raise RuntimeError(
                "Microphone input needs the 'sounddevice' package and the PortAudio library "
                "(Debian/Ubuntu: `sudo apt install libportaudio2`). Use a WAV file source instead "
                f"(--source path/to/file.wav). Original error: {e}"
            ) from e
        self._sd = sd
        self.sample_rate = int(sample_rate)
        self.blocksize = int(blocksize)
        self.device = device
        self._q: queue.Queue[np.ndarray] = queue.Queue(maxsize=max_queue)
        self._closed = threading.Event()
        self._stream = None
        self.dropped_blocks = 0
        self.status_flags = 0  # count of callbacks reporting over/underflow

    def _callback(self, indata, frames, time_info, status) -> None:  # PortAudio thread
        if status:
            self.status_flags += 1
        block = np.array(indata[:, 0], dtype=np.float32, copy=True)
        try:
            self._q.put_nowait(block)
        except queue.Full:
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            self.dropped_blocks += 1
            try:
                self._q.put_nowait(block)
            except queue.Full:
                pass

    def blocks(self) -> Iterator[np.ndarray]:
        self._stream = self._sd.InputStream(
            samplerate=self.sample_rate, blocksize=self.blocksize, channels=1, dtype="float32",
            device=self.device, callback=self._callback,
        )
        self._stream.start()
        try:
            while not self._closed.is_set():
                try:
                    yield self._q.get(timeout=0.1)
                except queue.Empty:
                    continue
        finally:
            self._stop_stream()

    def _stop_stream(self) -> None:
        s, self._stream = self._stream, None
        if s is not None:
            try:
                s.stop()
                s.close()
            except Exception:  # best effort on shutdown
                pass

    def close(self) -> None:
        self._closed.set()


class ArraySource:
    """In-memory array source, for tests. `realtime=True` paces blocks at wall-clock speed."""

    def __init__(self, samples: np.ndarray, sample_rate: int = 16000, blocksize: int = 1600,
                 realtime: bool = False):
        if blocksize <= 0:
            raise ValueError("blocksize must be > 0")
        self.samples = to_mono_float32(samples)
        self.sample_rate = int(sample_rate)
        self.blocksize = int(blocksize)
        self.realtime = bool(realtime)
        self._closed = threading.Event()

    @property
    def num_samples(self) -> int:
        return int(self.samples.shape[0])

    @property
    def duration_s(self) -> float:
        return self.num_samples / self.sample_rate

    def blocks(self) -> Iterator[np.ndarray]:
        x, n, bs, sr = self.samples, self.samples.shape[0], self.blocksize, self.sample_rate
        t0 = time.monotonic()
        for start in range(0, n, bs):
            if self._closed.is_set():
                return
            end = min(start + bs, n)
            if self.realtime:
                # Block [start, end) is "available" once its last sample has been recorded.
                # Absolute monotonic schedule -> no cumulative drift from sleep overshoot.
                delay = t0 + end / sr - time.monotonic()
                while delay > 0 and not self._closed.is_set():
                    time.sleep(min(delay, 0.05))
                    delay = t0 + end / sr - time.monotonic()
                if self._closed.is_set():
                    return
            yield x[start:end]

    def close(self) -> None:
        self._closed.set()


class FileSource(ArraySource):
    """WAV/FLAC file -> 16 kHz mono float32 blocks. realtime=True paces blocks at wall-clock speed.

    The file is read with soundfile, mixed to mono (channel mean) and resampled with
    scipy.signal.resample_poly if its rate differs from `sample_rate`.
    """

    def __init__(self, path, sample_rate: int = 16000, blocksize: int = 1600, realtime: bool = True):
        import soundfile as sf

        self.path = Path(path)
        data, sr_file = sf.read(str(self.path), dtype="float32", always_2d=True)
        self.file_sample_rate = int(sr_file)
        mono = to_mono_float32(data)
        super().__init__(resample(mono, sr_file, sample_rate), sample_rate=sample_rate,
                         blocksize=blocksize, realtime=realtime)
