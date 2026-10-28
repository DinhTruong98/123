import numpy as np
import pytest
import soundfile as sf

from soundwatch.data import augment as A


def _sig(rng, n=16000, amp=0.1):
    t = np.arange(n) / 16000
    return (amp * np.sin(2 * np.pi * 440 * t) + 0.01 * rng.standard_normal(n)).astype(np.float32)


@pytest.mark.parametrize("snr", [0.0, 5.0, 10.0, 20.0])
def test_mix_at_snr_hits_target(snr):
    rng = np.random.default_rng(0)
    s = _sig(rng)
    n = (0.05 * rng.standard_normal(16000)).astype(np.float32)
    y = A.mix_at_snr(s, n, snr)
    got = 20 * np.log10(A.rms(s) / A.rms(y - s))
    assert abs(got - snr) < 0.5


def test_mix_noise_with_bank_hits_sampled_snr(tmp_path):
    rng = np.random.default_rng(1)
    for i in range(3):
        sf.write(tmp_path / f"n{i}.wav", 0.05 * rng.standard_normal(40000), 16000, subtype="PCM_16")
    bank = A.NoiseBank(tmp_path)
    assert len(bank) == 3 and bank.sample(rng).shape == (16000,)
    s = _sig(rng)
    for seed in range(5):
        r1, r2 = np.random.default_rng(seed), np.random.default_rng(seed)
        y = A.mix_noise(s, bank, (3.0, 3.0), r1)
        assert abs(20 * np.log10(A.rms(s) / A.rms(y - s)) - 3.0) < 0.5
        np.testing.assert_array_equal(y, A.mix_noise(s, bank, (3.0, 3.0), r2))  # deterministic


def test_shape_dtype_and_range():
    rng = np.random.default_rng(2)
    s = _sig(rng, amp=0.9)
    noise = rng.standard_normal(32000).astype(np.float32)
    outs = [
        A.mix_noise(s, noise, (0, 20), rng),
        A.random_gain(s, rng),
        A.time_shift(s, rng),
        A.pitch_shift(s, rng),
        A.augment_wave(s, rng, noise, p_pitch=1.0),
    ]
    for y in outs:
        assert y.shape == s.shape and y.dtype == np.float32
        assert np.all(np.abs(y) <= 1.0)


def test_time_shift_bounds():
    rng = np.random.default_rng(3)
    s = np.zeros(16000, np.float32)
    s[8000] = 1.0
    for _ in range(20):
        y = A.time_shift(s, rng)
        assert abs(int(np.argmax(y)) - 8000) <= 3200


def test_spec_augment():
    rng = np.random.default_rng(4)
    x = rng.standard_normal((49, 40)).astype(np.float32)
    y = A.spec_augment(x, rng, fill=0.0)
    assert y.shape == x.shape and y.dtype == np.float32
    assert not np.shares_memory(x, y)
    changed = y != x
    assert changed.sum() <= 2 * 8 * 40 + 2 * 6 * 49
    y3 = A.spec_augment(x[..., None], rng)
    assert y3.shape == (49, 40, 1)
