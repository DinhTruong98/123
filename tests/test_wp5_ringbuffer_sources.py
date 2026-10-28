import time

import numpy as np
import pytest
import soundfile as sf

from soundwatch.audio.ringbuffer import RingBuffer
from soundwatch.audio.sources import ArraySource, FileSource, MicSource


@pytest.mark.parametrize("capacity", [16000, 7, 1])
def test_ringbuffer_matches_reference_slice(capacity):
    rng = np.random.default_rng(0)
    src = rng.standard_normal(60000).astype(np.float32)
    rb = RingBuffer(capacity)
    out = np.empty(capacity, np.float32)
    pos = 0
    sizes = list(rng.integers(0, 3 * capacity + 5, size=200)) + [capacity, capacity + 1, 0, 1]
    for n in sizes:
        n = int(min(n, src.size - pos))
        before = rb._buf  # noqa: SLF001 - check no reallocation
        rb.write(src[pos:pos + n])
        assert rb._buf is before
        pos += n
        assert rb.total_written == pos
        got = rb.latest(out)
        assert got is out
        ref = src[max(0, pos - capacity):pos]
        expected = np.concatenate([np.zeros(capacity - ref.size, np.float32), ref])
        np.testing.assert_array_equal(got, expected)
        if pos >= src.size:
            break


def test_ringbuffer_latest_alloc_and_shape_check():
    rb = RingBuffer(4)
    rb.write(np.array([1, 2, 3, 4, 5, 6], np.float32))
    np.testing.assert_array_equal(rb.latest(), [3, 4, 5, 6])
    with pytest.raises(ValueError):
        rb.latest(np.empty(3, np.float32))


def test_array_source_blocks_and_stereo():
    x = np.arange(5000, dtype=np.float32) / 5000
    blocks = list(ArraySource(x, blocksize=1600).blocks())
    assert [b.size for b in blocks] == [1600, 1600, 1600, 200]
    np.testing.assert_array_equal(np.concatenate(blocks), x)
    st = np.stack([x, -x], axis=1)
    np.testing.assert_allclose(np.concatenate(list(ArraySource(st).blocks())), 0.0)


def test_array_source_realtime_pacing_and_close():
    x = np.zeros(16000 * 3 // 10, np.float32)  # 0.3 s
    t0 = time.monotonic()
    n = sum(b.size for b in ArraySource(x, blocksize=800, realtime=True).blocks())
    dt = time.monotonic() - t0
    assert n == x.size
    assert 0.27 <= dt < 1.0
    src = ArraySource(np.zeros(16000 * 10, np.float32), blocksize=1600, realtime=True)
    it = src.blocks()
    next(it)
    src.close()
    assert list(it) == []


def test_file_source_resamples_44k_stereo(tmp_path):
    sr_in, dur, f0 = 44100, 2.0, 440.0
    t = np.arange(int(sr_in * dur)) / sr_in
    sig = 0.5 * np.sin(2 * np.pi * f0 * t)
    p = tmp_path / "a.wav"
    sf.write(p, np.stack([sig, 0.5 * sig], axis=1), sr_in, subtype="PCM_16")
    src = FileSource(p, sample_rate=16000, blocksize=1600, realtime=False)
    assert src.file_sample_rate == sr_in
    y = np.concatenate(list(src.blocks()))
    assert y.dtype == np.float32
    assert abs(y.size - 32000) <= 1
    # mono mix = mean(0.5, 0.25) * sin = 0.375 * sin; dominant frequency preserved
    spec = np.abs(np.fft.rfft(y[2000:-2000]))
    freq = np.fft.rfftfreq(y[2000:-2000].size, 1 / 16000)
    assert abs(freq[spec.argmax()] - f0) < 2.0
    assert np.sqrt(np.mean(y[2000:-2000] ** 2)) == pytest.approx(0.375 / np.sqrt(2), rel=0.02)


def test_mic_source_error_is_clear_without_portaudio():
    try:
        import sounddevice  # noqa: F401
    except OSError:
        with pytest.raises(RuntimeError, match="PortAudio"):
            MicSource()
    else:
        pytest.skip("PortAudio available")
