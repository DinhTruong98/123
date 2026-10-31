"""vn-v2: sigmoid head, mix augmentation, PCEN front-end, calibration in train / quantize."""

from __future__ import annotations

import argparse
import json

import numpy as np
import pytest

pytest.importorskip("keras")

import test_wp3_train as wp3  # noqa: E402
from soundwatch import paths  # noqa: E402

TINY, _args = wp3.TINY, wp3._args
tiny_env = wp3.tiny_env  # shared fixture


def _teacher(tiny_env):
    from soundwatch.models import common

    logits = np.random.default_rng(1).normal(size=(48, 8)).astype(np.float32)
    np.save(tiny_env / "cache" / "teacher_logits_train.npy", logits)
    common.mark_cache(tiny_env / "cache" / "teacher_logits_train.npy", "train", len(logits))


def test_train_sigmoid_pcen_mix_distill(tiny_env):
    from soundwatch.audio.features import load_stats, model_meta
    from soundwatch.models import train

    _teacher(tiny_env)
    m = train.run(_args(["--quick", "--distill", "--output", "sigmoid", "--frontend", "pcen",
                         "--p-mix", "0.5", "--tag", "ml", *TINY]))
    assert m["output"] == "sigmoid" and m["frontend"] == "pcen"
    meta = model_meta(load_stats(tiny_env / "models" / "feature_stats_ml_quick.json"))
    assert meta.frontend == "pcen" and meta.output == "sigmoid" and meta.window_s == 1.0
    assert (tiny_env / "models" / "student_ml_quick.keras").exists()


def test_p_mix_requires_sigmoid(tiny_env):
    from soundwatch.models import train

    with pytest.raises(ValueError):
        train.run(_args(["--quick", "--no-distill", "--p-mix", "0.5", *TINY]))


def test_default_run_stats_unchanged(tiny_env):
    """The default recipe writes the historical feature_stats params (no front-end keys)."""
    from soundwatch.audio.features import feature_params
    from soundwatch.models import train

    train.run(_args(["--quick", "--no-distill", *TINY]))
    st = json.loads((tiny_env / "models" / "feature_stats_quick.json").read_text())
    assert st["params"] == json.loads(json.dumps(feature_params()))
    assert "frontend" not in st["params"]


def _seq(p_mix: float, teacher=None, alpha=1.0, ls=0.0):
    from soundwatch.models.train import StudentSequence

    rng = np.random.default_rng(0)
    labels = np.array([0, 1, 2, 7, 7, 3, 0, 7] * 4)
    x = rng.normal(-3, 1, size=(len(labels), 49, 40)).astype(np.float32)
    ml = {"num_classes": 8, "background_id": 7, "alpha": alpha, "label_smoothing": ls,
          "teacher_probs": teacher, "p_mix": p_mix, "mix_event_db": 6.0, "mix_p_background": 0.3}
    seq = StudentSequence([x], np.zeros((len(labels), 16), np.float32), np.ones(len(labels), np.float32),
                          {"mean": 0.0, "std": 1.0}, 8, 0, 0.0, labels=labels, multilabel=ml)
    return seq, labels


def test_mix_targets_union_and_background_rule():
    seq, labels = _seq(1.0)
    for b in range(len(seq)):
        x, t, w = seq[b]
        assert x.shape == (len(t), 49, 40, 1) and t.shape[1] == 8
        sel = seq.order[b * 8:(b + 1) * 8]
        for i, j in enumerate(sel):
            assert t[i, labels[j]] == 1.0 or labels[j] == 7  # own label kept (bg may be replaced)
            ev = t[i, :7].max() > 0
            assert t[i, 7] == (0.0 if ev else 1.0)
            assert t[i].sum() <= 2.0
    # p_mix = 0: plain one-hot
    seq0, labels0 = _seq(0.0)
    x, t, _ = seq0[0]
    np.testing.assert_array_equal(t, np.eye(8)[labels0[seq0.order[:8]]])


def test_mix_adds_power():
    seq, labels = _seq(1.0)
    sel = seq.order[:8]
    x0 = seq._gather(sel, np.zeros(8, int))
    x1 = x0.copy()
    seq._multilabel_batch(sel, x1, np.random.default_rng(3))
    assert np.all(x1 >= x0 - 1e-5)  # adding mel power never lowers any bin
    assert np.any(x1 > x0 + 1e-3)


def test_distill_soft_targets_only_unmixed():
    teacher = np.full((32, 8), 1 / 8, np.float32)
    seq, labels = _seq(0.0, teacher=teacher, alpha=0.5)
    _, t, _ = seq[0]
    lab = labels[seq.order[:8]]
    np.testing.assert_allclose(t[np.arange(8), lab], 0.5 + 0.5 / 8, atol=1e-6)


def test_quantize_sigmoid_calibrated(tiny_env):
    from soundwatch.engine.classifier import TFLiteClassifier, load_model
    from soundwatch.models import quantize, train

    train.run(_args(["--quick", "--no-distill", "--output", "sigmoid", "--frontend", "pcen",
                     "--tag", "q", *TINY]))
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    quantize.add_arguments(p)
    out = tiny_env / "out"
    args = p.parse_args(["--tag", "q_quick", "--no-copy", "--calibrate", "--out-dir", str(out),
                         "--n-rep", "8"])
    assert quantize.main(args) == 0
    q = json.loads((out / "quant_metrics.json").read_text())
    assert "LOGISTIC" in q["int8"]["ops"] and not q["int8"]["unsupported_ops"]
    assert len(q["calibration"]["temperature"]) == 8
    assert {"int8_before", "int8_after", "float_before", "float_after"} <= set(q["calibration"])
    clf = TFLiteClassifier(out / "student_int8.tflite")
    assert clf.output == "sigmoid"  # detected from the LOGISTIC op
    clf2, stats, meta = load_model(out / "student_int8.tflite", out / "feature_stats.json")
    assert meta.frontend == "pcen" and clf2.output == "sigmoid"
    probs = clf2.predict(np.zeros((49, 40), np.float32))
    assert probs.shape == (8,) and probs.min() >= 0 and probs.max() <= 1
    assert not (paths.ROOT / "artifacts" / "models" / "student_q_quick.keras").exists()


def test_fit_temperature_recovers_and_reduces_nll():
    from soundwatch.eval import calibration as cal

    rng = np.random.default_rng(0)
    n, c = 4000, 8
    y = rng.integers(0, c, n)
    true_logits = rng.normal(0, 1, (n, c))
    true_logits[np.arange(n), y] += 2.0
    # sample labels from the calibrated softmax, then make the model over-confident (x3)
    p = cal.probs_from_logits(true_logits, "softmax")
    y = np.array([rng.choice(c, p=pi) for pi in p])
    over = 3.0 * true_logits
    t = cal.fit_temperature(over, y, "softmax")
    assert t.shape == (1,) and 2.5 < t[0] < 3.5
    assert cal.nll(over, y, "softmax", t) < cal.nll(over, y, "softmax")
    ts = cal.fit_temperature(over, y, "sigmoid")
    assert ts.shape == (c,) and np.all(ts > 1.0)
    r0 = cal.calibration_report(cal.probs_from_logits(over, "softmax"), y)
    r1 = cal.calibration_report(cal.probs_from_logits(over, "softmax", t), y)
    assert r1["ece_top_label"] < r0["ece_top_label"]
    assert r0["saturated_windows"] >= r1["saturated_windows"]
    assert sum(b["n"] for b in r1["reliability"]) == n


def test_fold_temperature_into_dense():
    from soundwatch.eval import calibration as cal
    from soundwatch.models.dscnn import build_dscnn

    m = build_dscnn((49, 40, 1), 8, 8, 1)
    x = np.random.default_rng(0).normal(size=(4, 49, 40, 1)).astype(np.float32)
    z = np.asarray(m(x, training=False))
    t = np.linspace(0.5, 3.0, 8)
    cal.fold_temperature(m, t)
    np.testing.assert_allclose(np.asarray(m(x, training=False)), z / t, rtol=1e-4, atol=1e-5)


def test_build_dataset_windows_dir_keeps_noise(tmp_path, monkeypatch):
    from soundwatch.config import load_config
    from soundwatch.data import build_dataset as bd

    cfg = load_config()
    wav = tmp_path / "a.wav"
    import soundfile as sf

    sf.write(wav, 0.1 * np.random.default_rng(0).standard_normal(48000), 16000)
    recs = [bd.ClipRecord("x/a.wav", str(wav), 0, "train"), bd.ClipRecord("x/b.wav", str(wav), 7, "val")]
    noise = tmp_path / "noise"
    (noise / "keep").mkdir(parents=True)
    out = tmp_path / "w2"
    bd.build(recs, cfg, out_dir=out, noise_dir=None, events_dir=None, workers=1, verbose=False)
    assert (out / "train.npz").exists() and (noise / "keep").exists()


def test_data_dirs_override(tmp_path, monkeypatch):
    from soundwatch.models import common

    monkeypatch.setattr(paths, "WINDOWS", paths.WINDOWS)
    monkeypatch.setattr(paths, "CACHE", paths.CACHE)
    p = argparse.ArgumentParser()
    common.add_data_dir_args(p)
    common.apply_data_dirs(p.parse_args(["--windows-dir", str(tmp_path / "w"), "--cache-dir",
                                         str(tmp_path / "c")]))
    assert paths.WINDOWS == (tmp_path / "w").resolve() and (tmp_path / "c").is_dir()
