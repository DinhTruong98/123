"""vn-v2: per-model front-end metadata, PCEN, sigmoid outputs, context modes, exports."""

from __future__ import annotations

import copy
import dataclasses
import json
import queue

import numpy as np
import pytest
import yaml

from soundwatch import paths
from soundwatch.audio import features as F
from soundwatch.config import ConfigError, load_config, with_mode
from soundwatch.engine.classifier import as_probs, to_probs
from soundwatch.engine.postprocess import PostProcessor
from soundwatch.types import Level, ModeCommand

SR = 16000
MODES = {
    "default": "indoor",
    "street": {"disable": ["dog_bark", "baby_cry"], "level": {"car_horn": 0}},
    "indoor": {},
    "auto": {"street_above_dbfs": -35, "window_s": 10, "min_s": 2},
}


def _noise(seed=0, amp=0.3):
    return (amp * np.random.default_rng(seed).standard_normal(SR)).astype(np.float32)


def _cfg_with_modes(tmp_path, modes=MODES, mutate=None):
    raw = yaml.safe_load(paths.CONFIG.read_text(encoding="utf-8"))
    raw["modes"] = copy.deepcopy(modes)
    if mutate:
        mutate(raw)
    p = tmp_path / "modes.yaml"
    p.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    return load_config(p)


# ---------------------------------------------------------------- front-end metadata
def test_model_meta_defaults_for_historical_stats():
    for p in ("assets/models/feature_stats.json", "assets/models/vn/feature_stats.json"):
        meta = F.model_meta(F.load_stats(paths.ROOT / p))
        assert meta.frontend == "logmel" and meta.output == "softmax" and meta.window_s is None
        assert meta.is_default
    assert F.model_meta({"mean": 0, "std": 1}).is_default
    assert F.model_meta(None).is_default


def test_feature_params_unchanged_without_meta():
    p = F.feature_params()
    assert "frontend" not in p and "output" not in p and "pcen" not in p
    stats = json.loads((paths.ROOT / "assets/models/feature_stats.json").read_text())
    assert stats["params"] == json.loads(json.dumps(p))


def test_meta_roundtrip(tmp_path):
    meta = F.ModelMeta.make("pcen", "sigmoid", 1.0, {"gain": 0.9})
    st = F.compute_feature_stats(np.ones((2, 49, 40), np.float32), None, meta)
    F.save_stats({**st, "std": 1.0}, tmp_path / "s.json")
    back = F.model_meta(F.load_stats(tmp_path / "s.json"))
    assert back == meta and back.pcen_params["gain"] == 0.9 and back.pcen_params["bias"] == 2.0
    params = json.loads((tmp_path / "s.json").read_text())["params"]
    assert params["frontend"] == "pcen" and params["output"] == "sigmoid"
    assert params["pcen"]["s"] == pytest.approx(F.pcen_smoothing_coef(0.4))
    with pytest.raises(ValueError):
        F.ModelMeta("mfcc")


def test_compute_features_logmel_is_identity():
    x = _noise()
    np.testing.assert_array_equal(F.compute_features(x), F.compute_logmel(x))
    np.testing.assert_array_equal(F.compute_features(x, None, F.ModelMeta()), F.compute_logmel(x))
    xb = np.stack([x, _noise(1)])
    np.testing.assert_array_equal(F.compute_features_batch(xb), F.compute_logmel_batch(xb))


def test_apply_meta_to_config():
    cfg = load_config()
    assert F.apply_meta_to_config(cfg, F.ModelMeta()) is cfg
    assert F.apply_meta_to_config(cfg, F.ModelMeta.make(window_s=1.0)) is cfg
    c2 = F.apply_meta_to_config(cfg, F.ModelMeta.make(window_s=2.0))
    assert c2.audio.window_samples == 32000 and c2.audio.n_frames == 99


# ---------------------------------------------------------------- PCEN
def test_pcen_smoothing_coef():
    # librosa: b = (sqrt(1 + 4 T^2) - 1) / (2 T^2), T = 0.4 s * 16000 / 320 = 20 frames
    assert F.pcen_smoothing_coef(0.4) == pytest.approx((np.sqrt(1 + 1600) - 1) / 800)


def test_pcen_matches_librosa():
    librosa = pytest.importorskip("librosa")
    from scipy.signal import lfilter_zi

    meta = F.ModelMeta.make("pcen")
    for seed, amp in ((0, 0.3), (1, 0.01), (2, 0.8)):
        x = _noise(seed, amp)
        x[SR // 2:] *= 0.05  # level step inside the window
        e = F.compute_mel(x)
        s = F.pcen_smoothing_coef(0.4)
        zi = lfilter_zi([s], [1, s - 1])[None, :] * e[0][:, None]  # smoother starts at E[0]
        ref = librosa.pcen(e.T.astype(np.float64), sr=SR, hop_length=320, gain=0.98, bias=2.0, power=0.5,
                           time_constant=0.4, eps=1e-6, max_size=1, zi=zi).T
        got = F.compute_features(x, None, meta)
        assert got.shape == (49, 40) and got.dtype == np.float32
        assert np.abs(got - ref).max() < 1e-3


def test_pcen_batch_and_from_logmel():
    meta = F.ModelMeta.make("pcen")
    xb = np.stack([_noise(0), _noise(1, 0.05), np.zeros(SR, np.float32)])
    pb = F.compute_features_batch(xb, None, meta)
    for i in range(3):
        np.testing.assert_allclose(pb[i], F.compute_features(xb[i], None, meta), atol=1e-6)
    assert np.isfinite(pb).all()
    lm = F.compute_logmel_batch(xb[:2])
    np.testing.assert_allclose(F.logmel_to_features(lm, meta), pb[:2], atol=1e-4)
    np.testing.assert_array_equal(F.logmel_to_features(lm, F.ModelMeta()), lm)


def test_pcen_is_stateless_per_window():
    meta = F.ModelMeta.make("pcen")
    x = _noise(3)
    a = F.compute_features(x, None, meta)
    F.compute_features(_noise(4, 0.9), None, meta)
    np.testing.assert_array_equal(a, F.compute_features(x, None, meta))


def test_export_golden_pcen(tmp_path):
    from soundwatch.tools import export_golden

    cfg = load_config()
    m = export_golden.export(tmp_path / "plain", cfg.audio, windows_dir=tmp_path)
    assert not (tmp_path / "plain" / "golden_pcen.h").exists()
    assert "frontend" not in json.loads((tmp_path / "plain" / "manifest.json").read_text())["params"]
    meta = F.ModelMeta.make("pcen")
    m = export_golden.export(tmp_path / "pcen", cfg.audio, windows_dir=tmp_path, meta=meta)
    h = (tmp_path / "pcen" / "golden_pcen.h").read_text()
    assert "sw_golden_pcen[5][49][40]" in h and "SW_PCEN_S" in h
    d = tmp_path / "pcen" / m[1]["dir"]
    x = np.load(d / "input_f32.npy")
    np.testing.assert_array_equal(np.load(d / "pcen.npy"), F.compute_features(x, cfg.audio, meta))
    assert json.loads((tmp_path / "pcen" / "manifest.json").read_text())["params"]["frontend"] == "pcen"


# ---------------------------------------------------------------- sigmoid outputs
def test_sigmoid_outputs_are_not_softmaxed():
    y = np.array([0.9, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05], np.float32)  # sums to 1.75
    np.testing.assert_allclose(to_probs(y, "sigmoid"), y)
    assert abs(float(as_probs(y).sum()) - 1.0) < 1e-5  # the historical path would renormalize
    np.testing.assert_array_equal(to_probs(y, None), as_probs(y))
    np.testing.assert_array_equal(to_probs(y, "softmax"), as_probs(y))


def test_tflite_sigmoid_detected(tmp_path):
    keras = pytest.importorskip("keras")
    from soundwatch.engine.classifier import TFLiteClassifier, load_classifier
    from soundwatch.models.dscnn import build_dscnn, with_output
    from soundwatch.models.quantize import convert

    base = build_dscnn((49, 40, 1), 8, 8, 1)
    rep = np.random.default_rng(0).normal(size=(8, 49, 40, 1)).astype(np.float32)
    for out in ("sigmoid", "softmax"):
        p = tmp_path / f"{out}.tflite"
        p.write_bytes(convert(with_output(base, out), rep))
        clf = TFLiteClassifier(p)
        assert clf.output == ("sigmoid" if out == "sigmoid" else None)
        assert load_classifier(p, output="sigmoid").output == "sigmoid"
    del keras


def test_load_classifier_rejects_unknown_output(tmp_path):
    from soundwatch.engine.classifier import TFLiteClassifier

    with pytest.raises(ValueError):
        TFLiteClassifier(paths.ROOT / "assets/models/student_int8.tflite", output="relu")


# ---------------------------------------------------------------- engine with a PCEN model
def test_engine_uses_model_frontend():
    from soundwatch.audio.sources import ArraySource
    from soundwatch.engine.classifier import DummyClassifier
    from soundwatch.engine.pipeline import run_offline

    cfg = load_config()
    x = np.concatenate([_noise(5), _noise(6)])
    stats = {"mean": 0.0, "std": 1.0,
             "params": F.feature_params(cfg.audio, F.ModelMeta.make("pcen", "sigmoid", 1.0))}
    _, summ, _ = run_offline(cfg, ArraySource(x), DummyClassifier(8), stats, keep_logmel=True)
    np.testing.assert_allclose(summ.logmels[0], F.compute_features(x[:SR], cfg.audio, F.model_meta(stats)))
    _, summ0, _ = run_offline(cfg, ArraySource(x), DummyClassifier(8), {"mean": 0.0, "std": 1.0},
                              keep_logmel=True)
    np.testing.assert_array_equal(summ0.logmels[0], F.compute_logmel(x[:SR], cfg.audio))


# ---------------------------------------------------------------- modes: config
def test_config_without_modes():
    for c in ("config/soundwatch.yaml", "config/soundwatch_vn.yaml"):
        cfg = load_config(paths.ROOT / c)
        assert cfg.modes is None
        assert with_mode(cfg, None) is cfg
        with pytest.raises(ConfigError):
            with_mode(cfg, "street")


def test_config_modes_parse(tmp_path):
    cfg = _cfg_with_modes(tmp_path)
    m = cfg.modes
    assert m.default == "indoor" and set(m.names) == {"street", "indoor", "auto"}
    assert m.spec("street").disable == ("dog_bark", "baby_cry")
    assert m.spec("street").level_map == {"car_horn": 0}
    assert m.auto.street_above_dbfs == -35 and m.auto.window_s == 10
    assert with_mode(cfg, "street").modes.default == "street"
    with pytest.raises(ConfigError):
        with_mode(cfg, "beach")


@pytest.mark.parametrize("bad", [
    {"street": {"disable": ["cat"]}},
    {"street": {"level": {"siren": 5}}},
    {"street": {"mute": ["siren"]}},
    {"default": "beach", "street": {}},
    {"auto": {"window_s": 0}},
])
def test_config_modes_invalid(tmp_path, bad):
    with pytest.raises(ConfigError):
        _cfg_with_modes(tmp_path, bad)


def test_auto_adds_indoor_street(tmp_path):
    cfg = _cfg_with_modes(tmp_path, {"default": "auto", "auto": {}})
    assert set(cfg.modes.names) == {"indoor", "street", "auto"}


# ---------------------------------------------------------------- modes: PostProcessor
def _p(cfg, key, v=1.0):
    p = np.zeros(cfg.num_classes, np.float32)
    p[cfg.class_keys.index(key)] = v
    p[-1] = 1.0 - v
    return p


def _run(pp, seq, t0=1.0, rms=None):
    out = []
    for i, p in enumerate(seq):
        out += pp.update(p, t0 + 0.5 * i, t_wall=0.0, rms_dbfs=None if rms is None else rms[i])
    return out


def test_no_modes_identical_events():
    cfg = load_config()
    rng = np.random.default_rng(0)
    seq = [rng.dirichlet(np.ones(8) * 0.1).astype(np.float32) for _ in range(400)]
    a = _run(PostProcessor(cfg.classes, cfg.postprocess), seq)
    b = _run(PostProcessor(cfg.classes, cfg.postprocess, None, None), seq, rms=[-20.0] * 400)
    assert a == b and len(a) > 0


def test_indoor_mode_identical(tmp_path):
    cfg = _cfg_with_modes(tmp_path)
    rng = np.random.default_rng(1)
    seq = [rng.dirichlet(np.ones(8) * 0.1).astype(np.float32) for _ in range(400)]
    a = _run(PostProcessor(cfg.classes, cfg.postprocess), seq)
    b = _run(PostProcessor(cfg.classes, cfg.postprocess, cfg.modes), seq)
    assert a == [dataclasses.replace(e, level=None) for e in b]
    assert all(e.level == int(cfg.classes[e.class_id].level) for e in b)


def test_street_mode_disables_and_downgrades(tmp_path):
    cfg = _cfg_with_modes(tmp_path)
    pp = PostProcessor(cfg.classes, cfg.postprocess, cfg.modes, "street")
    dog, horn = cfg.class_keys.index("dog_bark"), cfg.class_keys.index("car_horn")
    assert pp.mode == "street" and not pp.is_active_class(dog) and pp.is_enabled(dog)
    assert pp.level(horn) == Level.INFO
    assert _run(pp, [_p(cfg, "dog_bark")] * 6) == []
    # car_horn as INFO: no fast path on a single saturated hop, no repeats
    pp = PostProcessor(cfg.classes, cfg.postprocess, cfg.modes, "street")
    ev = _run(pp, [_p(cfg, "car_horn"), _p(cfg, "background")] * 3)
    assert ev == []
    ev = _run(pp, [_p(cfg, "car_horn")] * 30, t0=10.0)
    assert ev and not any(e.repeat for e in ev)
    # indoor: same input alerts at once (fast path) and repeats
    pp = PostProcessor(cfg.classes, cfg.postprocess, cfg.modes, "indoor")
    ev = _run(pp, [_p(cfg, "car_horn")] + [_p(cfg, "background")] * 20)
    assert ev[0].class_id == horn and any(e.repeat for e in ev)


def test_mode_switch_clears_occurrence(tmp_path):
    cfg = _cfg_with_modes(tmp_path)
    horn = cfg.class_keys.index("car_horn")
    pp = PostProcessor(cfg.classes, cfg.postprocess, cfg.modes)
    _run(pp, [_p(cfg, "car_horn")] * 2)
    assert pp.is_pending(horn)
    pp.set_mode("street", t_audio=2.0)
    assert not pp.is_pending(horn)
    assert pp.mode_log[-1][1:] == ("indoor", "street", "manual")
    with pytest.raises(ValueError):
        pp.set_mode("beach")


def test_auto_mode_follows_running_median(tmp_path):
    cfg = _cfg_with_modes(tmp_path, mutate=lambda r: r["modes"].update(default="auto"))
    pp = PostProcessor(cfg.classes, cfg.postprocess, cfg.modes)
    assert pp.mode_selector == "auto" and pp.mode == "indoor"
    bg = _p(cfg, "background")
    n = 60
    rms = [-50.0] * n + [-25.0] * n + [-50.0] * n  # 30 s quiet, 30 s loud, 30 s quiet
    _run(pp, [bg] * (3 * n), rms=rms)
    switches = [(round(t, 1), new) for t, _, new, why in pp.mode_log if why.startswith("auto:")]
    assert [s for _, s in switches] == ["street", "indoor"]
    # median over 10 s: flips about 5 s after each level change
    assert 30 + 4 < switches[0][0] < 30 + 7 and 60 + 4 < switches[1][0] < 60 + 7
    # a short loud burst (< half the window) does not switch
    pp2 = PostProcessor(cfg.classes, cfg.postprocess, cfg.modes)
    _run(pp2, [bg] * 60, rms=[-50.0] * 20 + [-20.0] * 8 + [-50.0] * 32)
    assert pp2.mode == "indoor"


def test_engine_mode_command_and_stats(tmp_path):
    from soundwatch.audio.sources import ArraySource
    from soundwatch.engine.classifier import DummyClassifier
    from soundwatch.engine.pipeline import Engine, replay_postprocess, run_offline

    cfg = _cfg_with_modes(tmp_path)
    x = 0.1 * np.random.default_rng(0).standard_normal(SR * 6).astype(np.float32)
    inq: queue.Queue = queue.Queue()
    inq.put(ModeCommand("street"))
    out: queue.Queue = queue.Queue()
    eng = Engine(cfg, ArraySource(x), DummyClassifier(8), {"mean": 0.0, "std": 1.0}, out, inq)
    eng.run()
    assert eng.stats["mode"] == "street"
    assert [m[2] for m in eng.stats["mode_changes"]] == ["indoor", "street"]
    # replay honours cfg.modes: street drops dog_bark alerts
    dog = cfg.class_keys.index("dog_bark")
    clf = DummyClassifier(8, [(1.0, 6.0, dog)], confidence=1.0)
    ev, summ, _ = run_offline(cfg, ArraySource(x), clf, {"mean": 0.0, "std": 1.0})
    assert any(e.class_id == dog for e in ev)
    assert not replay_postprocess(with_mode(cfg, "street"), summ)
    assert replay_postprocess(cfg, summ) == [dataclasses.replace(e, t_wall=0.0) for e in ev]


def test_cli_run_mode(tmp_path, capsys):
    import soundfile as sf

    from soundwatch.__main__ import main

    raw = yaml.safe_load(paths.CONFIG.read_text(encoding="utf-8"))
    raw["modes"] = MODES
    cp = tmp_path / "c.yaml"
    cp.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    wav = tmp_path / "a.wav"
    sf.write(wav, 0.1 * np.random.default_rng(0).standard_normal(SR * 3), SR)
    assert main(["run", "--config", str(cp), "--source", str(wav), "--headless", "--fast", "--dummy",
                 "--no-log", "--mode", "street"]) == 0
    assert "mode None -> street" in capsys.readouterr().out
    assert main(["run", "--source", str(wav), "--headless", "--fast", "--dummy", "--no-log",
                 "--mode", "street"]) == 2


# ---------------------------------------------------------------- exports
def test_headers_modes_and_frontend(tmp_path):
    from soundwatch.models import export_c

    cfg = load_config()
    assert "kMode" not in export_c.labels_header(cfg)
    assert export_c._frontend_lines({"mean": 0, "std": 1, "params": F.feature_params()}) == []
    h = export_c.labels_header(_cfg_with_modes(tmp_path))
    assert "#define SW_NUM_MODES 2" in h and "kModeDisabled" in h and "SW_AUTO_WINDOW_MS   10000" in h
    assert "{0, 0, 0, 1, 0, 1, 0, 0}" in h  # street: baby_cry, dog_bark off
    assert "{2, 0, 2, 1, 1, 0, 0, 0}" in h  # street levels: car_horn -> INFO
    lines = export_c._frontend_lines({"params": F.feature_params(None, F.ModelMeta.make("pcen", "sigmoid"))})
    txt = "\n".join(lines)
    assert "SW_FRONTEND_PCEN    1" in txt and "SW_OUTPUT_SIGMOID   1" in txt and "SW_PCEN_S" in txt
    assert "SW_PCEN_BIAS        2.0f" in txt  # valid C float literal


def test_ios_export_extras(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("eia", paths.ROOT / "ios/scripts/export_ios_assets.py")
    eia = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eia)
    cfg = load_config()
    stats = F.load_stats(paths.ROOT / "assets/models/feature_stats.json")
    assert eia.export_extras(cfg, stats) == {}
    assert "modes" not in eia.export_config(cfg, stats)
    st2 = {**stats, "params": F.feature_params(cfg.audio, F.ModelMeta.make("pcen", "sigmoid", 1.0))}
    ex = eia.export_extras(_cfg_with_modes(tmp_path), st2)
    assert ex["model_frontend"]["frontend"] == "pcen" and ex["modes"]["default"] == "indoor"
    assert ex["modes"]["modes"]["street"]["level"] == {"car_horn": 0}


def test_custom_max_windows_train_only():
    from soundwatch.data.build_dataset import custom_max_windows

    assert custom_max_windows("smoke_alarm", "train") == 2
    assert custom_max_windows("smoke_alarm", "val") == 6 == custom_max_windows("smoke_alarm", "test")
    assert custom_max_windows("car_horn", "train") == 4
    assert custom_max_windows("siren", "train") == 6


def test_fast_path_below_threshold_keeps_occurrence():
    """danger_fast_threshold < threshold: the fast-path alert starts an occurrence that is kept alive
    by further fast hits (no second fresh alert after the cooldown, repeats instead)."""
    cfg = load_config()
    siren = cfg.class_keys.index("siren")
    cls = tuple(dataclasses.replace(c, threshold=0.99609375) if c.key == "siren" else c for c in cfg.classes)
    pp = PostProcessor(cls, dataclasses.replace(cfg.postprocess, danger_fast_threshold=0.95))
    seq = [_p(cfg, "siren", 0.96)] + [_p(cfg, "siren", 0.99609375)] * 6 + [_p(cfg, "siren", 0.6)]
    ev = _run(pp, seq)
    fresh = [e for e in ev if not e.repeat and e.class_id == siren]
    assert len(fresh) == 1 and pp.is_pending(siren)


def test_alert_event_level_with_modes(tmp_path):
    from soundwatch.engine.logger import event_to_dict
    from soundwatch.engine.run import format_alert

    cfg = _cfg_with_modes(tmp_path)
    ev = _run(PostProcessor(cfg.classes, cfg.postprocess, cfg.modes, "street"), [_p(cfg, "car_horn")] * 3)
    assert ev and all(e.level == int(Level.INFO) for e in ev)
    assert format_alert(ev[0], cfg).split()[1] == "INFO"
    assert event_to_dict(ev[0], cfg.classes)["level"] == "INFO"
    ev0 = _run(PostProcessor(cfg.classes, cfg.postprocess), [_p(cfg, "car_horn")] * 3)
    assert ev0 and all(e.level is None for e in ev0)
    assert format_alert(ev0[0], cfg).split()[1] == "DANGER"
