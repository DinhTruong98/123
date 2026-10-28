"""OPTIONAL (WP2): fixed-point emulation of the log-mel pipeline, for MCUs without an FPU.

Not used by the engine or training; it only estimates the error a q15 port would add relative
to :func:`soundwatch.audio.features.compute_logmel`.

Emulated pipeline (per frame, all integer except the final log):
1. int16 input x, periodic Hann in Q15: ``f = (x * w_q15 + 2^14) >> 15``  (int16).
2. Block floating point: shift ``f`` left by ``s`` bits so that max|f| uses 14 bits (headroom
   for the FFT), s >= 0 (per frame).
3. FFT: CMSIS ``arm_rfft_q15`` style, output scaled by 1/N and rounded to int16. (Modelled as an
   exact FFT followed by a single rounding; real per-stage truncation is a bit worse.)
4. power = re^2 + im^2 in int32.
5. mel = sum(power * fb_q15) >> 15 in int64 (uint32 range on the MCU after the shift).
6. logmel = log(mel * N^2 / (32768^2 * 4^s) + eps) — on the MCU via CLZ + small log2 table;
   here with float log, since that step is cheap to make accurate.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from soundwatch.audio import features as F
from soundwatch.config import AudioConfig


def to_q15(x: np.ndarray) -> np.ndarray:
    return np.clip(np.round(np.asarray(x, np.float64) * 32768.0), -32768, 32767).astype(np.int16)


def compute_logmel_fixed(window: np.ndarray, cfg: AudioConfig | None = None,
                         headroom_bits: int = 14) -> np.ndarray:
    """(window_samples,) int16 (or float in [-1, 1]) -> (n_frames, n_mels) float32 log-mel."""
    sr, n_win, n_fft, step, n_mels, fmin, fmax, eps = F._params(cfg)
    x = np.asarray(window)
    x = x.astype(np.int16) if x.dtype == np.int16 else to_q15(F.to_float32(x))
    x = F._fit_length(x, n_win).astype(np.int32)

    w_q15 = np.minimum(to_q15(F.hann_window(n_fft)).astype(np.int32), 32767)
    frames = sliding_window_view(x, n_fft)[::step]  # (T, N) int32
    f = (frames * w_q15 + (1 << 14)) >> 15  # int16 range

    peak = np.max(np.abs(f), axis=1)
    bits = np.where(peak > 0, np.floor(np.log2(np.maximum(peak, 1))).astype(np.int64) + 1, headroom_bits)
    s = np.clip(headroom_bits - bits, 0, 15)  # per-frame left shift
    f = f.astype(np.int64) << s[:, None]

    spec = np.fft.rfft(f.astype(np.float64), axis=1) / n_fft
    re = np.clip(np.round(spec.real), -32768, 32767).astype(np.int64)
    im = np.clip(np.round(spec.imag), -32768, 32767).astype(np.int64)
    power = re * re + im * im  # <= 2^31

    fb_q15 = np.round(F.mel_filterbank(sr, n_fft, n_mels, fmin, fmax).astype(np.float64) * 32768.0)
    fb_q15 = np.minimum(fb_q15, 32767).astype(np.int64)
    mel = (power @ fb_q15) >> 15  # int64 (fits uint32 for these sizes)

    scale = (n_fft * n_fft) / (32768.0 * 32768.0) / (4.0 ** s.astype(np.float64))
    return np.log(mel.astype(np.float64) * scale[:, None] + eps).astype(np.float32)
