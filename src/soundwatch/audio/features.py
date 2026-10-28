"""Log-mel feature extraction in pure numpy (contract: plan section 4.2). Owner: WP2.

Must not use librosa / tf.signal at runtime: this is the reference for the C/CMSIS-DSP port.
"""

from __future__ import annotations

import numpy as np

from soundwatch.config import AudioConfig


def mel_filterbank(sr: int = 16000, n_fft: int = 512, n_mels: int = 40,
                   fmin: float = 20.0, fmax: float = 8000.0) -> np.ndarray:
    """HTK-mel triangular filterbank, shape (n_fft // 2 + 1, n_mels), float32."""
    raise NotImplementedError("WP2")


def compute_logmel(window: np.ndarray, cfg: AudioConfig | None = None) -> np.ndarray:
    """(window_samples,) float32 in [-1, 1] -> (n_frames, n_mels) float32 log-mel, NOT normalized."""
    raise NotImplementedError("WP2")


def compute_feature_stats(logmels: np.ndarray) -> dict:
    """Scalar mean/std over a (N, n_frames, n_mels) array -> {"mean": float, "std": float}."""
    raise NotImplementedError("WP2")


def load_stats(path) -> dict:
    raise NotImplementedError("WP2")


def save_stats(stats: dict, path) -> None:
    raise NotImplementedError("WP2")


def normalize(logmel: np.ndarray, stats: dict) -> np.ndarray:
    """(logmel - mean) / std, float32. Works on (49, 40) or batched (N, 49, 40)."""
    raise NotImplementedError("WP2")


def rms_dbfs(x: np.ndarray) -> float:
    """RMS level in dBFS (full scale = 1.0); returns -120.0 for silence."""
    raise NotImplementedError("WP2")
