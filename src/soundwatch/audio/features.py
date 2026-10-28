"""Log-mel feature extraction in pure numpy (contract: plan section 4.2). Owner: WP2.

Must not use librosa / tf.signal at runtime: this is the reference for the C/CMSIS-DSP port.

Exact feature definition (the firmware must reproduce this bit-for-bit up to float rounding):

1. Input: one window of ``window_samples`` (16000) mono samples. float32/float64 in [-1, 1],
   float16 (cast to float32) or int16 (divided by 32768). Shorter windows are zero-padded at
   the end; longer windows raise ``ValueError``.
2. Framing: ``n_frames = 1 + (window_samples - n_fft) // frame_step`` frames of ``n_fft`` (512)
   samples, hop ``frame_step`` (320), no centering / no padding -> 49 frames. Frame ``t`` covers
   samples ``[t*320, t*320 + 512)``.
3. Window: **periodic Hann** of length N = n_fft:
   ``w[n] = 0.5 - 0.5 * cos(2*pi*n / N)``, n = 0..N-1  (== ``scipy.signal.get_window('hann', N)``,
   == librosa ``window='hann'``). Note: w[0] = 0, w[N/2] = 1, NOT symmetric (``np.hanning``).
4. Power spectrum: ``P = |rfft(frame * w)|^2`` (no scaling), n_fft // 2 + 1 = 257 bins.
5. Mel: ``mel = P @ fb`` with ``fb`` the (257, 40) HTK-mel triangular filterbank from
   :func:`mel_filterbank`: ``n_mels + 2`` points equally spaced on the HTK mel scale
   ``mel(f) = 2595 * log10(1 + f / 700)`` between fmin=20 Hz and fmax=8000 Hz; bin k has
   frequency ``k * sr / n_fft``; triangle m rises from point m to m+1 and falls to m+2.
   **norm=None**: triangle peaks are 1.0 (no Slaney area normalization). Identical to
   ``librosa.filters.mel(..., htk=True, norm=None)``.
6. ``logmel = log(mel + eps)``, natural log, eps = 1e-6 (``AudioConfig.log_eps``).
7. Normalization (separate step, :func:`normalize`): ``(logmel - mean) / std`` with scalar
   mean/std from ``feature_stats.json``.

All arithmetic is float32 (as on the MCU with an FPU); the filterbank is built in float64 and
then cast to float32.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from soundwatch.config import AudioConfig

# ---- plan 4.2 constants (used when no AudioConfig is given) ----------------------------------
SAMPLE_RATE = 16000
WINDOW_SAMPLES = 16000
N_FFT = 512
FRAME_STEP = 320
N_MELS = 40
FMIN = 20.0
FMAX = 8000.0
LOG_EPS = 1e-6
N_FRAMES = 1 + (WINDOW_SAMPLES - N_FFT) // FRAME_STEP  # 49

#: Window type, documented for the C port: periodic Hann, w[n] = 0.5 - 0.5*cos(2*pi*n/N).
WINDOW_TYPE = "hann_periodic"
#: Mel filterbank normalization: None -> triangle peaks are 1.0 (librosa norm=None, htk=True).
MEL_NORM = None
MEL_SCALE = "htk"

SILENCE_DBFS = -120.0
_BATCH_CHUNK = 32  # windows per chunk in compute_logmel_batch (~7 MB temporaries, cache friendly)


def feature_params(cfg: AudioConfig | None = None) -> dict:
    """All parameters that define the feature (stored in feature_stats.json and C headers)."""
    if cfg is None:
        p = dict(sample_rate=SAMPLE_RATE, window_samples=WINDOW_SAMPLES, n_fft=N_FFT,
                 frame_step=FRAME_STEP, n_frames=N_FRAMES, n_mels=N_MELS, fmin=FMIN, fmax=FMAX,
                 log_eps=LOG_EPS)
    else:
        p = dict(sample_rate=cfg.sample_rate, window_samples=cfg.window_samples, n_fft=cfg.n_fft,
                 frame_step=cfg.frame_step, n_frames=cfg.n_frames, n_mels=cfg.n_mels,
                 fmin=cfg.fmin, fmax=cfg.fmax, log_eps=cfg.log_eps)
    p.update(window=WINDOW_TYPE, mel_scale=MEL_SCALE, mel_norm=MEL_NORM, log="natural")
    return p


# ---- constants: window + filterbank (cached) -------------------------------------------------
@lru_cache(maxsize=8)
def _hann_cached(n: int) -> np.ndarray:
    w = (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n, dtype=np.float64) / n)).astype(np.float32)
    w.setflags(write=False)
    return w


def hann_window(n: int = N_FFT) -> np.ndarray:
    """Periodic Hann window (float32, read-only)."""
    return _hann_cached(int(n))


def hz_to_mel_htk(f: np.ndarray | float) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def mel_to_hz_htk(m: np.ndarray | float) -> np.ndarray:
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


@lru_cache(maxsize=8)
def _mel_fb_cached(sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    if not (0.0 <= fmin < fmax <= sr / 2):
        raise ValueError(f"need 0 <= fmin < fmax <= sr/2, got fmin={fmin}, fmax={fmax}, sr={sr}")
    n_bins = n_fft // 2 + 1
    fft_freqs = np.linspace(0.0, sr / 2.0, n_bins)  # == k * sr / n_fft
    mel_pts = np.linspace(hz_to_mel_htk(fmin), hz_to_mel_htk(fmax), n_mels + 2)
    hz_pts = mel_to_hz_htk(mel_pts)
    fdiff = np.diff(hz_pts)  # (n_mels + 1,)
    ramps = hz_pts[:, None] - fft_freqs[None, :]  # (n_mels + 2, n_bins)
    lower = -ramps[:-2] / fdiff[:-1, None]
    upper = ramps[2:] / fdiff[1:, None]
    fb = np.maximum(0.0, np.minimum(lower, upper))  # (n_mels, n_bins), norm=None
    fb = np.ascontiguousarray(fb.T.astype(np.float32))
    fb.setflags(write=False)
    return fb


def mel_filterbank(sr: int = 16000, n_fft: int = 512, n_mels: int = 40,
                   fmin: float = 20.0, fmax: float = 8000.0) -> np.ndarray:
    """HTK-mel triangular filterbank, shape (n_fft // 2 + 1, n_mels), float32.

    norm=None (triangle peaks 1.0). The returned array is cached and read-only; copy before
    modifying.
    """
    return _mel_fb_cached(int(sr), int(n_fft), int(n_mels), float(fmin), float(fmax))


def mel_filterbank_sparse(fb: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sparse form for the firmware: per mel band (start_bin, length) into a flat weight array.

    Returns ``(start (n_mels,) int, length (n_mels,) int, weights (sum(length),) float32)``;
    band m uses bins ``start[m] .. start[m]+length[m]-1`` with weights
    ``weights[offset[m] : offset[m]+length[m]]``, offset = cumsum of lengths.
    """
    fb = mel_filterbank() if fb is None else fb
    starts, lengths, weights = [], [], []
    for m in range(fb.shape[1]):
        nz = np.nonzero(fb[:, m])[0]
        if nz.size == 0:
            starts.append(0)
            lengths.append(0)
            continue
        s, e = int(nz[0]), int(nz[-1]) + 1
        starts.append(s)
        lengths.append(e - s)
        weights.append(fb[s:e, m])
    w = np.concatenate(weights).astype(np.float32) if weights else np.zeros(0, np.float32)
    return np.asarray(starts, np.int32), np.asarray(lengths, np.int32), w


# ---- core -------------------------------------------------------------------------------------
def _params(cfg: AudioConfig | None) -> tuple[int, int, int, int, int, float, float, float]:
    if cfg is None:
        return SAMPLE_RATE, WINDOW_SAMPLES, N_FFT, FRAME_STEP, N_MELS, FMIN, FMAX, LOG_EPS
    return (cfg.sample_rate, cfg.window_samples, cfg.n_fft, cfg.frame_step, cfg.n_mels,
            cfg.fmin, cfg.fmax, cfg.log_eps)


def to_float32(x: np.ndarray) -> np.ndarray:
    """int16 -> /32768; other ints -> error; floats -> float32."""
    x = np.asarray(x)
    if x.dtype == np.int16:
        return x.astype(np.float32) * np.float32(1.0 / 32768.0)
    if not np.issubdtype(x.dtype, np.floating):
        raise TypeError(f"unsupported audio dtype {x.dtype} (expected float32/float16/int16)")
    return x.astype(np.float32, copy=False)


def _fit_length(x: np.ndarray, n: int) -> np.ndarray:
    """Zero-pad the last axis to n samples; raise if longer."""
    L = x.shape[-1]
    if L > n:
        raise ValueError(f"window has {L} samples, expected at most {n}")
    if L < n:
        pad = [(0, 0)] * (x.ndim - 1) + [(0, n - L)]
        x = np.pad(x, pad)
    return x


def _logmel_core(x: np.ndarray, n_fft: int, step: int, fb: np.ndarray, eps: float) -> np.ndarray:
    """x (..., window_samples) float32 -> (..., n_frames, n_mels) float32."""
    frames = sliding_window_view(x, n_fft, axis=-1)[..., ::step, :]  # view, no copy
    spec = np.fft.rfft(frames * hann_window(n_fft), axis=-1)  # complex64 for float32 input
    power = spec.real * spec.real + spec.imag * spec.imag
    mel = power @ fb
    return np.log(mel + np.float32(eps)).astype(np.float32, copy=False)


def compute_logmel(window: np.ndarray, cfg: AudioConfig | None = None) -> np.ndarray:
    """(window_samples,) float32 in [-1, 1] -> (n_frames, n_mels) float32 log-mel, NOT normalized."""
    sr, n_win, n_fft, step, n_mels, fmin, fmax, eps = _params(cfg)
    x = to_float32(window)
    if x.ndim != 1:
        raise ValueError(f"expected a 1-D window, got shape {x.shape}; use compute_logmel_batch")
    x = _fit_length(x, n_win)
    return _logmel_core(x, n_fft, step, mel_filterbank(sr, n_fft, n_mels, fmin, fmax), eps)


def compute_logmel_batch(waves: np.ndarray, cfg: AudioConfig | None = None,
                         chunk: int = _BATCH_CHUNK) -> np.ndarray:
    """(N, window_samples) float32/float16/int16 -> (N, n_frames, n_mels) float32.

    Processed in chunks of ``chunk`` windows to bound memory; identical to calling
    :func:`compute_logmel` on each row.
    """
    sr, n_win, n_fft, step, n_mels, fmin, fmax, eps = _params(cfg)
    waves = np.asarray(waves)
    if waves.ndim != 2:
        raise ValueError(f"expected (N, samples), got shape {waves.shape}")
    if waves.shape[1] > n_win:
        raise ValueError(f"windows have {waves.shape[1]} samples, expected at most {n_win}")
    fb = mel_filterbank(sr, n_fft, n_mels, fmin, fmax)
    n_frames = 1 + (n_win - n_fft) // step
    out = np.empty((waves.shape[0], n_frames, n_mels), np.float32)
    for i in range(0, waves.shape[0], max(1, chunk)):
        x = _fit_length(to_float32(waves[i:i + chunk]), n_win)
        out[i:i + chunk] = _logmel_core(x, n_fft, step, fb, eps)
    return out


# ---- normalization stats ----------------------------------------------------------------------
def compute_feature_stats(logmels: np.ndarray, cfg: AudioConfig | None = None) -> dict:
    """Scalar mean/std over a (N, n_frames, n_mels) array -> {"mean": float, "std": float}.

    Also returns ``n_windows`` and ``params`` (feature parameters, see :func:`feature_params`).
    Accumulates in float64 chunk by chunk (safe for large N).
    """
    logmels = np.asarray(logmels)
    if logmels.ndim == 2:
        logmels = logmels[None]
    if logmels.size == 0:
        raise ValueError("cannot compute stats of an empty array")
    s = ss = 0.0
    for i in range(0, logmels.shape[0], 4096):
        c = logmels[i:i + 4096].astype(np.float64)
        s += float(c.sum())
        ss += float(np.square(c).sum())
    n = logmels.size
    mean = s / n
    var = max(ss / n - mean * mean, 0.0)
    return {"mean": float(mean), "std": float(np.sqrt(var)), "n_windows": int(logmels.shape[0]),
            "params": feature_params(cfg)}


def save_stats(stats: dict, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"mean": float(stats["mean"]), "std": float(stats["std"]),
            "n_windows": int(stats.get("n_windows", 0)), "params": stats.get("params", feature_params())}
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def load_stats(path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "mean" not in data or "std" not in data:
        raise ValueError(f"{path}: feature stats must contain 'mean' and 'std'")
    data["mean"] = float(data["mean"])
    data["std"] = float(data["std"])
    if not np.isfinite(data["mean"]) or not np.isfinite(data["std"]) or data["std"] <= 0:
        raise ValueError(f"{path}: invalid stats mean={data['mean']} std={data['std']}")
    return data


def normalize(logmel: np.ndarray, stats: dict) -> np.ndarray:
    """(logmel - mean) / std, float32. Works on (49, 40) or batched (N, 49, 40)."""
    mean = np.float32(stats["mean"])
    inv_std = np.float32(1.0 / max(float(stats["std"]), 1e-12))
    return ((np.asarray(logmel, dtype=np.float32) - mean) * inv_std).astype(np.float32, copy=False)


def rms_dbfs(x: np.ndarray) -> float:
    """RMS level in dBFS (full scale = 1.0); returns -120.0 for silence."""
    x = to_float32(x)
    if x.size == 0:
        return SILENCE_DBFS
    ms = float(np.mean(np.square(x, dtype=np.float64)))
    if ms <= 1e-12:
        return SILENCE_DBFS
    return max(SILENCE_DBFS, 10.0 * float(np.log10(ms)))
