"""WP1: training-time (online) augmentation. Pure functions, explicit ``np.random.Generator``.

Waveform ops take/return 1-D float32 arrays in [-1, 1] (same length as the input).
``spec_augment`` works on un-normalized or normalized log-mel ``(T, F)`` or ``(T, F, 1)``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample

from soundwatch import paths

EPS = 1e-12


def _clip(x: np.ndarray) -> np.ndarray:
    return np.clip(x, -1.0, 1.0).astype(np.float32, copy=False)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


class NoiseBank:
    """Background clips (16 kHz wav) loaded lazily; ``sample`` returns random crops.

    Files are read on demand with partial reads (cheap for wav); set ``preload=True``
    to keep every clip in RAM (float16) for faster sampling.
    """

    def __init__(
        self,
        directory: str | Path | None = None,
        sample_rate: int = 16000,
        preload: bool = False,
        min_rms_db: float = -60.0,
        retries: int = 5,
    ) -> None:
        self.directory = Path(directory) if directory else paths.NOISE / "train"
        self.sample_rate = sample_rate
        self.preload = preload
        self.min_rms_db = min_rms_db
        self.retries = retries
        self._files: list[Path] | None = None
        self._frames: dict[int, int] = {}
        self._cache: dict[int, np.ndarray] = {}

    @property
    def files(self) -> list[Path]:
        if self._files is None:
            self._files = sorted(self.directory.glob("*.wav"))
            if not self._files:
                raise FileNotFoundError(f"no noise wavs in {self.directory} (run build-dataset)")
        return self._files

    def __len__(self) -> int:
        return len(self.files)

    def _n_frames(self, i: int) -> int:
        if i not in self._frames:
            self._frames[i] = sf.info(str(self.files[i])).frames
        return self._frames[i]

    def _read(self, i: int, start: int, stop: int) -> np.ndarray:
        if self.preload:
            if i not in self._cache:
                self._cache[i] = sf.read(str(self.files[i]), dtype="float32")[0].astype(np.float16)
            return self._cache[i][start:stop].astype(np.float32)
        return sf.read(str(self.files[i]), start=start, stop=stop, dtype="float32")[0]

    def _crop(self, rng: np.random.Generator, n: int) -> np.ndarray:
        i = int(rng.integers(len(self.files)))
        total = self._n_frames(i)
        if total <= n:
            x = self._read(i, 0, total)
            return np.pad(x, (0, n - len(x))) if len(x) < n else x
        start = int(rng.integers(total - n + 1))
        return self._read(i, start, start + n)

    def sample(self, rng: np.random.Generator, n_samples: int | None = None) -> np.ndarray:
        """Random crop of ``n_samples`` (default 1 s); avoids near-silent crops when possible."""
        n = n_samples or self.sample_rate
        x = self._crop(rng, n)
        for _ in range(self.retries):
            if 20 * np.log10(rms(x) + EPS) >= self.min_rms_db:
                break
            x = self._crop(rng, n)
        return x.astype(np.float32, copy=False)


def mix_at_snr(signal: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """``signal + g*noise`` with g chosen so that 10*log10(P_signal / P_scaled_noise) == snr_db."""
    if len(noise) != len(signal):
        noise = np.resize(noise, len(signal))
    ps, pn = rms(signal), rms(noise)
    if ps < 1e-6 or pn < 1e-9:  # silent signal or noise: nothing sensible to scale against
        return _clip(signal + (noise if ps < 1e-6 else 0.0))
    g = ps / pn / (10.0 ** (snr_db / 20.0))
    return _clip(signal + g * noise)


def mix_noise(
    wave: np.ndarray,
    noise_bank: NoiseBank | np.ndarray,
    snr_db_range: tuple[float, float] = (0.0, 20.0),
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Add background noise at a random SNR in ``snr_db_range`` (uniform, dB)."""
    rng = rng if rng is not None else np.random.default_rng()
    if isinstance(noise_bank, np.ndarray):
        if len(noise_bank) > len(wave):
            s = int(rng.integers(len(noise_bank) - len(wave) + 1))
            noise = noise_bank[s : s + len(wave)]
        else:
            noise = noise_bank
    else:
        noise = noise_bank.sample(rng, len(wave))
    snr = float(rng.uniform(*snr_db_range))
    return mix_at_snr(wave, noise, snr)


def random_gain(wave: np.ndarray, rng: np.random.Generator, max_db: float = 6.0) -> np.ndarray:
    return _clip(wave * 10.0 ** (rng.uniform(-max_db, max_db) / 20.0))


def time_shift(
    wave: np.ndarray, rng: np.random.Generator, max_s: float = 0.2, sample_rate: int = 16000
) -> np.ndarray:
    """Shift by up to ±max_s seconds, zero-filling (no wrap-around)."""
    m = int(round(max_s * sample_rate))
    k = int(rng.integers(-m, m + 1))
    out = np.zeros_like(wave, dtype=np.float32)
    if k > 0:
        out[k:] = wave[:-k]
    elif k < 0:
        out[:k] = wave[-k:]
    else:
        out[:] = wave
    return out


def pitch_shift(wave: np.ndarray, rng: np.random.Generator, max_semitones: float = 2.0) -> np.ndarray:
    """Cheap pitch shift by FFT resampling (also changes tempo), cropped/padded to the input length."""
    semis = rng.uniform(-max_semitones, max_semitones)
    ratio = 2.0 ** (semis / 12.0)
    n = len(wave)
    y = resample(wave.astype(np.float64), max(1, int(round(n / ratio))))
    if len(y) >= n:
        s = (len(y) - n) // 2
        y = y[s : s + n]
    else:
        y = np.pad(y, (0, n - len(y)))
    return _clip(y)


def spec_augment(
    logmel: np.ndarray,
    rng: np.random.Generator,
    n_time_masks: int = 2,
    max_t: int = 8,
    n_freq_masks: int = 2,
    max_f: int = 6,
    fill: float | None = None,
) -> np.ndarray:
    """Time/frequency masking on ``(T, F[, 1])``. Masks are filled with ``fill`` (default: input mean)."""
    out = np.array(logmel, dtype=np.float32, copy=True)
    t_len, f_len = out.shape[0], out.shape[1]
    v = float(out.mean()) if fill is None else fill
    for _ in range(n_time_masks):
        w = int(rng.integers(0, max_t + 1))
        if w:
            s = int(rng.integers(0, max(1, t_len - w + 1)))
            out[s : s + w] = v
    for _ in range(n_freq_masks):
        w = int(rng.integers(0, max_f + 1))
        if w:
            s = int(rng.integers(0, max(1, f_len - w + 1)))
            out[:, s : s + w] = v
    return out


def augment_wave(
    wave: np.ndarray,
    rng: np.random.Generator,
    noise_bank: NoiseBank | np.ndarray | None = None,
    p_noise: float = 0.8,
    p_gain: float = 1.0,
    p_shift: float = 0.5,
    p_pitch: float = 0.0,
    snr_db_range: tuple[float, float] = (0.0, 20.0),
    sample_rate: int = 16000,
) -> np.ndarray:
    """Convenience chain for training: shift -> pitch -> noise -> gain. Returns float32 (len(wave),)."""
    x = np.asarray(wave, dtype=np.float32)
    if rng.random() < p_shift:
        x = time_shift(x, rng, 0.2, sample_rate)
    if rng.random() < p_pitch:
        x = pitch_shift(x, rng)
    if noise_bank is not None and rng.random() < p_noise:
        x = mix_noise(x, noise_bank, snr_db_range, rng)
    if rng.random() < p_gain:
        x = random_gain(x, rng)
    return _clip(x)
