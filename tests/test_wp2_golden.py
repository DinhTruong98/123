"""WP2 tests: golden vector export and (optional) fixed-point emulation."""

from __future__ import annotations

import argparse
import json
import re

import numpy as np
import pytest

from soundwatch.audio import features as F
from soundwatch.audio.features_fixed import compute_logmel_fixed
from soundwatch.config import load_config
from soundwatch.tools import export_golden as G


@pytest.fixture(scope="module")
def golden(tmp_path_factory):
    out = tmp_path_factory.mktemp("golden")
    manifest = G.export(out, load_config().audio, windows_dir=out / "no_windows_here")
    return out, manifest


def test_golden_files(golden):
    out, manifest = golden
    assert len(manifest) == 5
    names = [m["name"] for m in manifest]
    assert names == ["silence", "sine_1khz", "white_noise", "chirp", "sum_of_sines"]
    for m in manifest:
        d = out / m["dir"]
        x, sr = G.read_wav(d / "input.wav")
        assert sr == 16000 and x.dtype == np.int16 and x.shape == (16000,)
        x32 = np.load(d / "input_f32.npy")
        np.testing.assert_array_equal(x32, x.astype(np.float32) / 32768)
        lm = np.load(d / "logmel.npy")
        assert lm.shape == (49, 40) and lm.dtype == np.float32
        np.testing.assert_array_equal(lm, F.compute_logmel(x32))
    for f in ("golden_features.h", "mel_filterbank.h", "README.md", "manifest.json"):
        assert (out / f).exists()
    assert json.loads((out / "manifest.json").read_text())["params"]["n_frames"] == 49


def test_golden_deterministic(golden, tmp_path):
    out, _ = golden
    G.export(tmp_path, load_config().audio, windows_dir=tmp_path / "none")
    for f in ("golden_features.h", "mel_filterbank.h"):
        assert (tmp_path / f).read_text() == (out / f).read_text()


def _parse_array(text: str, name: str) -> np.ndarray:
    m = re.search(rf"{name}(?:\[\d+\])+ = \{{(.*?)\}};", text, re.S)
    assert m, name
    return np.array([float(v.rstrip("f")) for v in m.group(1).replace("\n", " ").split(",") if v.strip()])


def test_headers_reconstruct_reference(golden):
    """Re-implement the C algorithm from the header constants only and compare with the golden logmel."""
    out, manifest = golden
    fbh = (out / "mel_filterbank.h").read_text()
    gh = (out / "golden_features.h").read_text()
    hann = _parse_array(fbh, "sw_hann_window").astype(np.float32)
    start = _parse_array(fbh, "sw_mel_start").astype(int)
    length = _parse_array(fbh, "sw_mel_len").astype(int)
    offset = _parse_array(fbh, "sw_mel_offset").astype(int)
    weights = _parse_array(fbh, "sw_mel_weights").astype(np.float32)
    xs = _parse_array(gh, "sw_golden_input").reshape(5, 16000)
    lms = _parse_array(gh, "sw_golden_logmel").reshape(5, 49, 40).astype(np.float32)
    np.testing.assert_array_equal(hann, F.hann_window())
    for k in range(5):
        x = (xs[k] / 32768.0).astype(np.float32)
        out_lm = np.empty((49, 40), np.float32)
        for t in range(49):
            spec = np.fft.rfft(x[t * 320:t * 320 + 512] * hann)
            p = (spec.real ** 2 + spec.imag ** 2).astype(np.float32)
            for m in range(40):
                w = weights[offset[m]:offset[m] + length[m]]
                out_lm[t, m] = np.log(np.dot(p[start[m]:start[m] + length[m]], w) + np.float32(1e-6))
        np.testing.assert_allclose(lms[k], np.load(out / manifest[k]["dir"] / "logmel.npy"), atol=1e-6)
        assert np.abs(out_lm - lms[k]).max() < 1e-3


def test_golden_uses_test_window(tmp_path):
    wdir = tmp_path / "windows"
    wdir.mkdir()
    wave = (np.random.default_rng(1).uniform(-0.3, 0.3, (3, 16000))).astype(np.float16)
    np.savez(wdir / "test.npz", wave=wave, label=np.zeros(3, int))
    cases = G.make_cases(load_config().audio, windows_dir=wdir)
    assert cases[4][0] == "test_window0"
    np.testing.assert_allclose(cases[4][2] / 32768.0, wave[0].astype(np.float32), atol=1e-4)


def test_cli_main(tmp_path):
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    G.add_arguments(p)
    assert G.main(p.parse_args(["--out", str(tmp_path)])) == 0
    assert len(list(tmp_path.glob("case_*/logmel.npy"))) == 5


def test_fixed_point_error(golden):
    """OPTIONAL fixed-point emulation: report error vs float reference."""
    out, manifest = golden
    for m in manifest:
        x, _ = G.read_wav(out / m["dir"] / "input.wav")
        ref = np.load(out / m["dir"] / "logmel.npy")
        fx = compute_logmel_fixed(x)
        assert fx.shape == ref.shape and np.isfinite(fx).all()
        err = np.abs(fx - ref)
        # only bands with meaningful energy (> -6 nats below the window max) matter for the model
        loud = ref > ref.max() - 6.0
        print(f"fixed-point {m['name']:<13} max abs err {err.max():.4f}  (loud bins {err[loud].max():.4f})")
        if m["name"] == "silence":
            assert err.max() < 1e-5
        else:
            assert err[loud].max() < 0.1
