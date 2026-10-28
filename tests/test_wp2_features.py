"""WP2 tests: log-mel feature extraction (plan 4.2)."""

from __future__ import annotations

import sys
import time

import numpy as np
import pytest

from soundwatch.audio import features as F
from soundwatch.config import load_config

SR = 16000


def _sine(freq: float, amp: float = 0.5, n: int = 16000) -> np.ndarray:
    return (amp * np.sin(2 * np.pi * freq * np.arange(n) / SR)).astype(np.float32)


def _noise(seed: int = 0, amp: float = 0.5, shape=(16000,)) -> np.ndarray:
    return (amp * np.random.default_rng(seed).uniform(-1, 1, shape)).astype(np.float32)


@pytest.fixture(scope="module")
def audio_cfg():
    return load_config().audio


def test_runtime_module_does_not_import_librosa():
    import subprocess

    code = ("import sys, soundwatch.audio.features as f, numpy as np; f.compute_logmel(np.zeros(16000));"
            "assert 'librosa' not in sys.modules and 'tensorflow' not in sys.modules")
    subprocess.run([sys.executable, "-c", code], check=True)


def test_shapes_dtypes(audio_cfg):
    lm = F.compute_logmel(_noise())
    assert lm.shape == (49, 40) and lm.dtype == np.float32
    lm2 = F.compute_logmel(_noise(), audio_cfg)
    np.testing.assert_array_equal(lm, lm2)
    fb = F.mel_filterbank()
    assert fb.shape == (257, 40) and fb.dtype == np.float32
    assert F.mel_filterbank() is fb  # cached
    assert (fb.sum(axis=0) > 0).all()  # no empty band
    assert fb.max() <= 1.0 and fb.min() >= 0.0


def test_hann_is_periodic():
    w = F.hann_window(512)
    assert w.dtype == np.float32 and w.shape == (512,)
    assert w[0] == 0.0 and w[256] == pytest.approx(1.0)
    np.testing.assert_allclose(w[1:], w[1:][::-1], atol=1e-7)  # periodic symmetry
    scipy_signal = pytest.importorskip("scipy.signal")
    np.testing.assert_allclose(w, scipy_signal.get_window("hann", 512), atol=1e-7)


def test_zeros_finite():
    lm = F.compute_logmel(np.zeros(16000, np.float32))
    assert np.isfinite(lm).all()
    np.testing.assert_allclose(lm, np.log(np.float32(1e-6)), rtol=1e-6)


def test_input_dtypes_and_padding():
    x = _noise(1)
    x16 = np.clip(np.round(x * 32768), -32768, 32767).astype(np.int16)
    a = F.compute_logmel(x16)
    b = F.compute_logmel(x16.astype(np.float32) / 32768)
    np.testing.assert_allclose(a, b, atol=1e-5)
    c = F.compute_logmel(x.astype(np.float16))
    assert c.dtype == np.float32 and np.abs(c - F.compute_logmel(x)).max() < 0.05
    short = F.compute_logmel(x[:10000])
    padded = np.concatenate([x[:10000], np.zeros(6000, np.float32)])
    np.testing.assert_array_equal(short, F.compute_logmel(padded))
    with pytest.raises(ValueError):
        F.compute_logmel(np.zeros(16001, np.float32))
    with pytest.raises(ValueError):
        F.compute_logmel_batch(np.zeros((2, 16001), np.float32))


@pytest.mark.parametrize("name", ["noise", "sine", "chirp", "quiet"])
def test_matches_librosa(name):
    librosa = pytest.importorskip("librosa")
    t = np.arange(16000) / SR
    y = {
        "noise": _noise(0),
        "sine": _sine(1000.0),
        "chirp": (0.5 * np.sin(2 * np.pi * (100 * t + 3450 * t * t))).astype(np.float32),
        "quiet": _noise(3, amp=1e-3),
    }[name]
    S = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=512, hop_length=320, win_length=512, window="hann", center=False,
        power=2.0, n_mels=40, fmin=20, fmax=8000, htk=True, norm=None)
    ref = np.log(S.T + 1e-6)
    ours = F.compute_logmel(y)
    diff = float(np.abs(ours - ref).max())
    print(f"librosa max abs diff ({name}): {diff:.3g}")
    assert diff < 1e-3

    fb_ref = librosa.filters.mel(sr=SR, n_fft=512, n_mels=40, fmin=20, fmax=8000, htk=True, norm=None)
    np.testing.assert_allclose(F.mel_filterbank(), fb_ref.T, atol=1e-6)


def test_sine_1khz_peak_band():
    lm = F.compute_logmel(_sine(1000.0))
    band = int(np.argmax(lm.mean(axis=0)))
    # mel band whose triangle covers 1 kHz with the largest weight
    k = round(1000 * 512 / SR)  # FFT bin of 1 kHz (= 32)
    expected = int(np.argmax(F.mel_filterbank()[k]))
    assert band == expected
    centers = F.mel_to_hz_htk(np.linspace(F.hz_to_mel_htk(20), F.hz_to_mel_htk(8000), 42))[1:-1]
    assert abs(centers[band] - 1000) < 100


def test_batch_equals_single():
    waves = _noise(5, shape=(70, 16000))
    batch = F.compute_logmel_batch(waves, chunk=16)
    assert batch.shape == (70, 49, 40) and batch.dtype == np.float32
    for i in (0, 15, 16, 69):
        np.testing.assert_allclose(batch[i], F.compute_logmel(waves[i]), atol=1e-6)
    b16 = F.compute_logmel_batch(waves.astype(np.float16))
    np.testing.assert_allclose(b16[3], F.compute_logmel(waves[3].astype(np.float16)), atol=1e-6)
    short = F.compute_logmel_batch(waves[:, :12000])
    np.testing.assert_allclose(short[2], F.compute_logmel(waves[2, :12000]), atol=1e-6)


def test_stats_roundtrip_and_normalize(tmp_path, audio_cfg):
    lms = F.compute_logmel_batch(_noise(7, shape=(20, 16000)) * np.linspace(0.01, 1, 20)[:, None])
    st = F.compute_feature_stats(lms, audio_cfg)
    assert st["n_windows"] == 20
    assert st["mean"] == pytest.approx(float(lms.astype(np.float64).mean()), rel=1e-6)
    assert st["std"] == pytest.approx(float(lms.astype(np.float64).std()), rel=1e-5)
    assert st["params"]["n_mels"] == 40 and st["params"]["window"] == "hann_periodic"
    p = tmp_path / "sub" / "feature_stats.json"
    F.save_stats(st, p)
    st2 = F.load_stats(p)
    assert st2["mean"] == st["mean"] and st2["std"] == st["std"] and st2["n_windows"] == 20
    z = F.normalize(lms, st2)
    assert z.dtype == np.float32 and z.shape == lms.shape
    assert abs(float(z.mean())) < 1e-4 and abs(float(z.std()) - 1) < 1e-4
    np.testing.assert_allclose(F.normalize(lms[4], st2), z[4], atol=1e-6)
    p.write_text('{"mean": 1.0, "std": 0.0}')
    with pytest.raises(ValueError):
        F.load_stats(p)


def test_rms_dbfs():
    assert F.rms_dbfs(np.zeros(16000, np.float32)) == -120.0
    assert F.rms_dbfs(np.ones(100, np.float32)) == pytest.approx(0.0, abs=1e-6)
    assert F.rms_dbfs(_sine(440, amp=1.0)) == pytest.approx(-3.0103, abs=0.01)
    assert F.rms_dbfs(np.full(100, 16384, np.int16)) == pytest.approx(-6.0206, abs=0.01)
    assert F.rms_dbfs(np.full(100, 1e-9, np.float32)) == -120.0


def test_timing():
    x = _noise(9)
    for _ in range(10):
        F.compute_logmel(x)
    n = 200
    t0 = time.perf_counter()
    for _ in range(n):
        F.compute_logmel(x)
    single = (time.perf_counter() - t0) / n * 1e3
    waves = _noise(10, shape=(512, 16000)).astype(np.float16)
    t0 = time.perf_counter()
    F.compute_logmel_batch(waves)
    batched = (time.perf_counter() - t0) / len(waves) * 1e3
    print(f"\ncompute_logmel: {single:.3f} ms/window single, {batched:.3f} ms/window batched")
    assert single < 10.0 and batched < 10.0
