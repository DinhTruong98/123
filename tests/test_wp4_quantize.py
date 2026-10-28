"""WP4: int8 conversion of a tiny DS-CNN + C export."""

from __future__ import annotations

import json
import re

import numpy as np
import pytest

pytest.importorskip("tensorflow")

from soundwatch.config import load_config  # noqa: E402
from soundwatch.models import export_c, quantize  # noqa: E402
from soundwatch.models.dscnn import build_dscnn, with_softmax  # noqa: E402


@pytest.fixture(scope="module")
def tiny_int8() -> tuple[bytes, bytes]:
    m = with_softmax(build_dscnn((49, 40, 1), 8, width=8, n_blocks=1))
    rep = np.random.default_rng(0).normal(size=(20, 49, 40, 1)).astype(np.float32)
    return quantize.convert(m), quantize.convert(m, rep)


def test_int8_conversion_ops_and_io(tiny_int8):
    float_tfl, int8_tfl = tiny_int8
    info = quantize.inspect_tflite(int8_tfl)
    assert info["unsupported_ops"] == []
    assert {"CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED", "SOFTMAX"} <= set(info["op_set"])
    assert info["input"]["dtype"] == "int8" and info["output"]["dtype"] == "int8"
    assert info["input"]["shape"] == [1, 49, 40, 1]
    assert 0 < info["activation_peak_bytes"] <= info["arena_estimate_bytes"]
    x = np.random.default_rng(1).normal(size=(4, 49, 40, 1)).astype(np.float32)
    p8 = quantize.tflite_predict(int8_tfl, x)
    pf = quantize.tflite_predict(float_tfl, x)
    assert p8.shape == pf.shape == (4, 8)
    np.testing.assert_allclose(p8.sum(1), 1.0, atol=0.05)
    np.testing.assert_allclose(pf.sum(1), 1.0, atol=1e-4)


def test_export_c(tmp_path, tiny_int8):
    _, int8_tfl = tiny_int8
    model = tmp_path / "m.tflite"
    model.write_bytes(int8_tfl)
    stats = tmp_path / "stats.json"
    stats.write_text(json.dumps({"mean": -5.5, "std": 3.25}))
    cfg = load_config()
    files = export_c.export_all(cfg, model, stats, tmp_path / "fw")
    cc = files["model_data.cc"].read_text()
    assert "alignas(16) const unsigned char g_model[]" in cc
    assert f"const unsigned int g_model_len = {len(int8_tfl)};" in cc
    body = cc.split("g_model[] = {")[1].split("};")[0]
    assert len(re.findall(r"0x[0-9a-f]{2}", body)) == len(int8_tfl)
    assert "extern const unsigned char g_model[];" in files["model_data.h"].read_text()
    fp = files["feature_params.h"].read_text()
    for key in ("SW_SAMPLE_RATE      16000", "SW_N_FFT            512", "SW_FRAME_STEP       320",
                "SW_N_FRAMES         49", "SW_N_MELS           40", "SW_FEATURE_MEAN     -5.5f",
                "SW_FEATURE_STD      3.25f", "SW_INPUT_SCALE", "SW_INPUT_ZERO_POINT", "SW_OUTPUT_SCALE"):
        assert key in fp
    lab = files["labels.h"].read_text()
    assert f"#define SW_NUM_CLASSES {cfg.num_classes}" in lab
    assert '"Bao chay"' in lab and '"Coi xe"' in lab and '"Cho sua"' in lab
    assert "kVibration[" in lab and "kCooldownMs" in lab
    lab.encode("ascii")  # UTF-8 labels are octal-escaped: header is pure ASCII


def test_c_string_roundtrip():
    s = "Gõ cửa / Chuông cửa"
    lit = export_c.c_string(s)
    decoded = bytes(lit[1:-1], "ascii").decode("unicode_escape").encode("latin-1").decode("utf-8")
    assert decoded == s
    assert export_c.ascii_fold("Đèn đỏ Còi cấp cứu") == "Den do Coi cap cuu"
