import json

import numpy as np
import pytest

from soundwatch import paths
from soundwatch.config import load_config
from soundwatch.eval import benchmark as bm
from soundwatch.eval import offline as off

SR = 16000


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def tiny_models(tmp_path_factory, cfg):
    """Tiny random Keras student + its int8 TFLite conversion."""
    import keras
    from keras import layers

    from soundwatch.models.quantize import convert

    keras.utils.set_random_seed(0)
    d = tmp_path_factory.mktemp("models")
    inp = keras.Input((49, 40, 1))
    x = layers.Conv2D(4, (10, 4), strides=(2, 2), padding="same", activation="relu")(inp)
    x = layers.DepthwiseConv2D((3, 3), padding="same", activation="relu")(x)
    x = layers.Conv2D(8, (1, 1), activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    out = layers.Dense(cfg.num_classes, activation="softmax")(x)
    model = keras.Model(inp, out)
    model.save(d / "student_distill.keras")
    rep = np.random.default_rng(0).standard_normal((16, 49, 40, 1)).astype(np.float32)
    (d / "student_int8.tflite").write_bytes(convert(model, rep))
    (d / "feature_stats.json").write_text(json.dumps({"mean": -5.0, "std": 4.0}))
    return d, model


def test_metrics_from_predictions(cfg):
    y = np.array([0, 0, 1, 1, 2, 2, 7, 7, 3, 4, 5, 6])
    p = np.array([0, 1, 1, 1, 2, 7, 7, 7, 3, 4, 5, 6])
    m = off.metrics_from_predictions(y, p, cfg)
    assert m["accuracy"] == pytest.approx(10 / 12)
    assert m["danger_recall"] == {"smoke_alarm": 0.5, "car_horn": 1.0, "siren": 0.5}
    assert m["danger_recall_min"] == 0.5 and m["danger_recall_mean"] == pytest.approx(2 / 3)
    assert m["per_class"]["car_horn"]["precision"] == pytest.approx(2 / 3)
    cmn = np.asarray(m["confusion_normalized"])
    assert np.allclose(cmn.sum(1), 1.0)
    assert 0 < m["macro_f1"] < 1


def test_confusion_png(tmp_path, cfg):
    y = np.repeat(np.arange(cfg.num_classes), 3)
    m = off.metrics_from_predictions(y, y[::-1], cfg)
    off.plot_confusion(m, cfg, tmp_path / "cm.png", "test")
    assert (tmp_path / "cm.png").stat().st_size > 5000


def test_evaluate_with_tiny_models(tmp_path, cfg, tiny_models, monkeypatch):
    mdir, _ = tiny_models
    rng = np.random.default_rng(0)
    wins = tmp_path / "windows"
    wins.mkdir()
    n = 24
    np.savez(
        wins / "test.npz",
        wave=(0.1 * rng.standard_normal((n, SR))).astype(np.float16),
        label=(np.arange(n) % cfg.num_classes).astype(np.int16),
        source_clip=np.array([f"c{i}" for i in range(n)]),
    )
    monkeypatch.setattr(paths, "WINDOWS", wins)
    monkeypatch.setattr(paths, "CACHE", tmp_path / "cache")
    res = off.evaluate(cfg, "test", models_dir=mdir, out_dir=tmp_path / "rep", verbose=False)
    assert set(res["models"]) == {"student_distill", "student_int8"}
    assert "teacher" in res["missing"] and "student_nodistill" in res["missing"]
    assert res["models"]["student_int8"]["n"] == n
    assert "int8_drop" in res and res["int8_drop"]["reference"] == "student_distill"
    assert (tmp_path / "rep" / "offline_student_int8.png").exists()
    saved = json.loads((tmp_path / "rep" / "offline_metrics.json").read_text())
    assert saved["models"]["student_distill"]["confusion_png"] == "offline_student_distill.png"


def test_evaluate_nothing_available(tmp_path, cfg, monkeypatch):
    monkeypatch.setattr(paths, "WINDOWS", tmp_path / "none")
    res = off.evaluate(cfg, "test", models_dir=tmp_path / "none", out_dir=tmp_path / "rep", verbose=False)
    assert res["models"] == {} and "error" in res


def test_tflite_macs_match_keras(tiny_models):
    from soundwatch.models.dscnn import count_macs

    mdir, model = tiny_models
    t = bm.tflite_macs_params(mdir / "student_int8.tflite")
    assert t["macs"] == count_macs(model)
    assert abs(t["params"] - model.count_params()) <= 16  # same weights (+ axis/shape consts)
    assert {"CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED"} <= set(t["ops"])


def test_mcu_estimates():
    e = bm.mcu_estimates(5_000_000)
    m4 = e[0]
    assert m4["latency_ms_worst"] == pytest.approx(5e6 / (0.25 * 64e6) * 1000, rel=1e-3)
    assert m4["latency_ms_best"] == pytest.approx(5e6 / (0.5 * 64e6) * 1000, rel=1e-3)
    assert all(x["estimate"] for x in e)


def test_benchmark_run(tmp_path, cfg, tiny_models):
    import soundfile as sf

    mdir, _ = tiny_models
    scen = tmp_path / "scen"
    scen.mkdir()
    x = np.zeros(SR * 6, np.float32)
    x[SR * 3 :] = 0.05
    sf.write(str(scen / "bg_00.wav"), x, SR)
    res = bm.run(
        cfg,
        [mdir / "student_int8.tflite", mdir / "student_distill.keras", mdir / "missing.tflite"],
        runs=30,
        models_dir=mdir,
        reports_dir=tmp_path / "rep",
        scen_dir=scen,
        verbose=False,
    )
    q = res["models"]["student_int8"]
    assert q["size_bytes"] > 0 and q["macs"] > 0 and q["arena_estimate_bytes"] > 0
    assert q["infer_ms"]["n"] == 30 and q["infer_ms"]["median"] > 0
    assert q["mcu_estimates"][0]["macs"] == q["macs"]
    assert res["models"]["student_distill"]["macs"] == q["macs"]
    assert res["missing"] == ["missing.tflite"]
    assert 0.3 < res["gating"]["gated_fraction"] < 0.8
    assert json.loads((tmp_path / "rep" / "benchmark.json").read_text())["runs"] == 30
