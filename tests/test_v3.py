"""vn-v3: relabel overrides, per-mode display labels/icons, street-sound evidence for `auto`, exports."""

from __future__ import annotations

import copy
import dataclasses
import importlib.util
import json
import queue

import numpy as np
import pytest
import yaml

from soundwatch import paths
from soundwatch.config import AutoModeConfig, ConfigError, load_config
from soundwatch.data import build_dataset as B
from soundwatch.engine.postprocess import PostProcessor
from soundwatch.types import AlertEvent, ModeCommand, event_icon, event_label

V3 = paths.ROOT / "config" / "soundwatch_vn_v3.yaml"
MODES = {
    "default": "indoor",
    "indoor": {"label": {"siren": "Còi hú báo cháy"}, "icon": {"siren": "fire.png"}},
    "street": {"disable": ["dog_bark"], "level": {"car_horn": 0}, "label": {"siren": "Xe ưu tiên"}},
    "auto": {"street_above_dbfs": -35, "window_s": 30, "hysteresis_db": 3},
}


def _cfg(tmp_path, modes=MODES):
    raw = yaml.safe_load((paths.ROOT / "config" / "soundwatch.yaml").read_text(encoding="utf-8"))
    raw["modes"] = copy.deepcopy(modes)
    p = tmp_path / "m.yaml"
    p.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    return load_config(p)


def _p(cfg, key, v=1.0):
    p = np.zeros(cfg.num_classes, np.float32)
    p[cfg.class_keys.index(key)] = v
    p[-1] += 1.0 - v
    return p


def _run(pp, seq, t0=1.0, dt=0.5, rms=None):
    ev = []
    for i, p in enumerate(seq):
        ev += pp.update(p, t0 + dt * i, t_wall=0.0, rms_dbfs=rms if rms is None else float(rms))
    return ev


# ---------------------------------------------------------------- relabel
def _rec(sid, key, split="train", cfg=None):
    cfg = cfg or load_config()
    return B.ClipRecord(sid, f"/x/{sid}", cfg.class_keys.index(key), split, B.custom_max_windows(key, split))


def test_relabel_csv_and_apply(tmp_path):
    cfg = load_config()
    p = tmp_path / "r.csv"
    p.write_text("file,old_class,new_class,reason\n"
                 "smoke_alarm/a.wav,smoke_alarm,siren,wail\n"
                 "smoke_alarm/b.wav,smoke_alarm,drop,both\n"
                 "fsd50k/dev/1.wav,smoke_alarm,siren,wail\n"
                 "smoke_alarm/low.wav,smoke_alarm,siren,skipped low\n", encoding="utf-8")
    rel = B.load_relabel(p)
    assert rel["vn/smoke_alarm/a.wav"] == ("smoke_alarm", "siren")
    assert "fsd50k/dev/1.wav" in rel
    recs = [_rec("vn/smoke_alarm/a.wav", "smoke_alarm"), _rec("vn/smoke_alarm/b.wav", "smoke_alarm"),
            _rec("vn/smoke_alarm/c.wav", "smoke_alarm"),
            B.ClipRecord("fsd50k/dev/1.wav", "/x", cfg.class_keys.index("smoke_alarm"), "train", 8)]
    out, info = B.apply_relabel(recs, rel, cfg)
    by = {r.source_clip: r for r in out}
    assert set(by) == {"vn/smoke_alarm/a.wav", "vn/smoke_alarm/c.wav", "fsd50k/dev/1.wav"}
    assert cfg.class_keys[by["vn/smoke_alarm/a.wav"].label] == "siren"
    # the VN train cap follows the new class (smoke_alarm 2 per clip in train, siren 6); FSD50K keeps its cap
    assert by["vn/smoke_alarm/a.wav"].max_windows == B.custom_max_windows("siren", "train") == 6
    assert by["vn/smoke_alarm/c.wav"].max_windows == 2
    assert by["fsd50k/dev/1.wav"].max_windows == 8
    assert info["applied"] == {"smoke_alarm->siren": 2, "smoke_alarm->drop": 1}
    assert info["not_in_records"] == ["vn/smoke_alarm/low.wav"]
    assert B.apply_relabel(recs, {}, cfg)[0] == recs


def test_relabel_rejects_wrong_old_class_and_unknown_class(tmp_path):
    cfg = load_config()
    recs = [_rec("vn/s/a.wav", "siren")]
    with pytest.raises(ValueError, match="old_class"):
        B.apply_relabel(recs, {"vn/s/a.wav": ("smoke_alarm", "siren")}, cfg)
    with pytest.raises(ValueError, match="unknown new_class"):
        B.apply_relabel(recs, {"vn/s/a.wav": ("siren", "ambulance")}, cfg)
    p = tmp_path / "dup.csv"
    p.write_text("file,old_class,new_class\na.wav,siren,drop\na.wav,siren,drop\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        B.load_relabel(p)


def test_relabel_v3_file_consistent_with_manifest():
    import csv

    man = {r["file"]: r for r in csv.DictReader(open(paths.ROOT / "data/vn/manifest.csv", encoding="utf-8"))}
    rows = list(csv.DictReader(open(paths.ROOT / "data/vn/relabel_v3.csv", encoding="utf-8")))
    assert rows and {r["new_class"] for r in rows} <= {"siren", "drop"}
    for r in rows:
        assert man[r["file"]]["class_key"] == r["old_class"] == "smoke_alarm"
        assert r["reason"].strip()
    # every VN smoke_alarm clip that YAMNet hears as more siren than alarm is decided (relabelled or dropped)
    decided = {r["file"] for r in rows}
    wails = {f for f, r in man.items() if r["class_key"] == "smoke_alarm"
             and float(r["siren_score"]) > float(r["alarm_score"])}
    assert wails <= decided


def test_build_parser_has_custom_relabel():
    import argparse

    ap = argparse.ArgumentParser()
    B.add_arguments(ap)
    assert ap.parse_args(["--custom-relabel", "x.csv"]).custom_relabel == "x.csv"


# ---------------------------------------------------------------- config: display overrides, auto evidence
def test_mode_display_overrides_parse(tmp_path):
    cfg = _cfg(tmp_path)
    m = cfg.modes
    assert m.spec("indoor").label_map == {"siren": "Còi hú báo cháy"}
    assert m.spec("indoor").icon_map == {"siren": "fire.png"}
    assert m.spec("street").icon_map == {}
    assert m.auto.uses_rms and not m.auto.uses_sound


@pytest.mark.parametrize("bad", [
    {"indoor": {"label": {"ambulance": "x"}}},
    {"indoor": {"label": {"siren": "  "}}},
    {"indoor": {"icon": {"background": "ear.png"}}},
    {"auto": {"street_above_dbfs": None}},
    {"auto": {"street_evidence": {"truck": 1.0}}},
    {"auto": {"street_evidence": {"car_horn": 1.0}, "evidence_window_s": 0}},
    {"auto": {"street_evidence": [1, 2]}},
])
def test_mode_display_and_auto_invalid(tmp_path, bad):
    with pytest.raises(ConfigError):
        _cfg(tmp_path, bad)


def test_mode_icon_checked_with_check_icons(tmp_path):
    cfg = _cfg(tmp_path, {"indoor": {"icon": {"siren": "nope.png"}}})
    assert cfg.modes.spec("indoor").icon_map == {"siren": "nope.png"}
    with pytest.raises(ConfigError, match="icon"):
        load_config(tmp_path / "m.yaml", check_icons=True)


def test_v3_config_loads_with_icons():
    cfg = load_config(V3, check_icons=True)
    assert cfg.class_keys == load_config().class_keys  # 8-class contract unchanged
    assert cfg.class_by_key("siren").label_vi == "Còi hú"
    assert cfg.modes.spec("street").label_map["siren"].startswith("Xe ưu tiên")
    assert cfg.modes.auto.uses_sound and cfg.modes.auto.uses_rms
    assert cfg.modes.auto.evidence_map["car_horn"] > 0 > cfg.modes.auto.evidence_map["baby_cry"]


# ---------------------------------------------------------------- PostProcessor: rule 9b
def test_events_carry_mode_display(tmp_path):
    cfg = _cfg(tmp_path)
    si, horn = cfg.class_keys.index("siren"), cfg.class_keys.index("car_horn")
    pp = PostProcessor(cfg.classes, cfg.postprocess, cfg.modes)
    ev = _run(pp, [_p(cfg, "siren")] * 3)
    assert ev and all(e.label == "Còi hú báo cháy" and e.icon == "fire.png" for e in ev if e.class_id == si)
    assert pp.display(si) == ("Còi hú báo cháy", "fire.png")
    assert pp.display(horn) == (cfg.classes[horn].label_vi, cfg.classes[horn].icon)
    pp.set_mode("street", t_audio=3.0)
    assert pp.display(si) == ("Xe ưu tiên", cfg.classes[si].icon)
    ev = _run(pp, [_p(cfg, "background")] * 30 + [_p(cfg, "siren")] * 3, t0=4.0)
    fresh = [e for e in ev if e.class_id == si and not e.repeat]
    assert fresh and fresh[-1].label == "Xe ưu tiên" and fresh[-1].icon is None
    sc = cfg.classes[si]
    assert event_label(fresh[-1], sc) == "Xe ưu tiên" and event_icon(fresh[-1], sc) == sc.icon
    plain = AlertEvent(si, 0.9, 1.0, 0.0)
    assert event_label(plain, sc) == sc.label_vi and event_icon(plain, sc) == sc.icon


def test_no_display_overrides_identical_events(tmp_path):
    base = _cfg(tmp_path, {"default": "indoor", "indoor": {}, "street": {"disable": ["dog_bark"]}})
    disp = _cfg(tmp_path, {"default": "indoor", "indoor": {"label": {"siren": "X"}},
                           "street": {"disable": ["dog_bark"]}})
    rng = np.random.default_rng(3)
    seq = [rng.dirichlet(np.ones(8) * 0.1).astype(np.float32) for _ in range(400)]
    a = _run(PostProcessor(base.classes, base.postprocess, base.modes), seq)
    b = _run(PostProcessor(disp.classes, disp.postprocess, disp.modes), seq)
    assert a == [dataclasses.replace(e, label=None) for e in b]
    assert all(e.label is None for e in a) and all(e.icon is None for e in a + b)
    c = _run(PostProcessor(base.classes, base.postprocess), seq)  # no modes at all
    assert [dataclasses.replace(e, level=None) for e in a] == c


# ---------------------------------------------------------------- auto: street-sound evidence
def _auto_cfg(tmp_path, **auto):
    m = copy.deepcopy(MODES)
    m["default"] = "auto"
    m["auto"] = auto
    return _cfg(tmp_path, m)


def test_auto_sound_evidence_switches_with_hysteresis(tmp_path):
    cfg = _auto_cfg(tmp_path, street_above_dbfs=None, street_evidence={"car_horn": 1.0, "baby_cry": -1.0},
                    street_score_above=0.1, score_hysteresis=0.05, evidence_window_s=20, min_s=5)
    pp = PostProcessor(cfg.classes, cfg.postprocess, cfg.modes)
    assert pp.mode == "indoor" and pp.mode_selector == "auto"
    horn, bg, baby = _p(cfg, "car_horn", 0.6), _p(cfg, "background"), _p(cfg, "baby_cry", 0.3)
    _run(pp, [bg] * 40)  # 20 s of quiet indoor sound
    assert pp.mode == "indoor" and pp.street_score == 0.0
    _run(pp, ([horn] + [bg] * 3) * 20, t0=21.0)  # horn every 2 s: mean evidence 0.15
    assert pp.mode == "street" and pp.street_score > 0.1
    assert 21.0 < pp.mode_log[-1][0] <= 61.0 and "street score" in pp.mode_log[-1][3]
    _run(pp, [bg] * 10, t0=61.0)  # score decays but stays above 0.05 for a while (hysteresis)
    assert pp.mode == "street"
    _run(pp, [baby] * 60, t0=66.0)  # household sound pulls the score below 0.05
    assert pp.mode == "indoor" and pp.street_score < 0.05


def test_auto_min_dwell_and_rms_veto(tmp_path):
    cfg = _auto_cfg(tmp_path, street_above_dbfs=-35, window_s=10, hysteresis_db=4,
                    street_evidence={"car_horn": 1.0}, street_score_above=0.2, score_hysteresis=0.1,
                    evidence_window_s=5, min_s=2, min_dwell_s=30)
    horn, bg = _p(cfg, "car_horn", 0.9), _p(cfg, "background")
    # loud street with horns -> street; then quiet without horns: stays street for >= 30 s (dwell)
    pp = PostProcessor(cfg.classes, cfg.postprocess, cfg.modes)
    _run(pp, [horn, bg] * 10, rms=-25)
    assert pp.mode == "street"
    t_sw = pp.mode_log[-1][0]
    _run(pp, [bg] * 40, t0=11.0, rms=-50)
    assert pp.mode == "street" or pp.mode_log[-1][0] - t_sw >= 30
    _run(pp, [bg] * 80, t0=31.0, rms=-50)
    assert pp.mode == "indoor" and pp.mode_log[-1][0] - t_sw >= 30
    # quiet room hearing horns (e.g. through a window): the level vetoes street
    pp = PostProcessor(cfg.classes, cfg.postprocess, cfg.modes)
    _run(pp, [horn, bg] * 40, rms=-55)
    assert pp.mode == "indoor"
    # loud room without horns (TV, karaoke): no switch to street
    pp = PostProcessor(cfg.classes, cfg.postprocess, cfg.modes)
    _run(pp, [bg] * 80, rms=-20)
    assert pp.mode == "indoor"


def test_auto_rms_only_unchanged(tmp_path):
    """RMS-only auto (every vn-v2 config) behaves as before: switch on the running median with hysteresis."""
    cfg = _auto_cfg(tmp_path, street_above_dbfs=-35, window_s=10, hysteresis_db=2, min_s=5)
    pp = PostProcessor(cfg.classes, cfg.postprocess, cfg.modes)
    bg = _p(cfg, "background")
    _run(pp, [bg] * 20, rms=-30)
    assert pp.mode == "street"
    _run(pp, [bg] * 30, t0=11.0, rms=-35.5)  # inside the hysteresis band: stays
    assert pp.mode == "street"
    _run(pp, [bg] * 30, t0=26.0, rms=-40)
    assert pp.mode == "indoor"
    assert pp.street_score == 0.0


# ---------------------------------------------------------------- outputs: log, console, engine status
def test_logger_and_console_use_effective_label(tmp_path):
    from soundwatch.engine.logger import event_to_dict
    from soundwatch.engine.run import format_alert

    cfg = _cfg(tmp_path)
    si = cfg.class_keys.index("siren")
    d0 = event_to_dict(AlertEvent(si, 0.9, 1.0, 0.0), cfg.classes)
    assert d0["label_vi"] == cfg.classes[si].label_vi and "icon" not in d0
    ev = AlertEvent(si, 0.9, 1.0, 0.0, level=2, label="Xe ưu tiên", icon="siren.png")
    d = event_to_dict(ev, cfg.classes)
    assert d["label_vi"] == "Xe ưu tiên" and d["icon"] == "siren.png"
    assert "Xe ưu tiên" in format_alert(ev, cfg)


def test_engine_status_reports_mode(tmp_path):
    from soundwatch.audio.sources import ArraySource
    from soundwatch.engine.classifier import DummyClassifier
    from soundwatch.engine.pipeline import Engine
    from soundwatch.types import StatusUpdate

    cfg = _cfg(tmp_path)
    q: queue.Queue = queue.Queue()
    x = (np.random.default_rng(0).standard_normal(16000 * 3) * 0.1).astype(np.float32)
    eng = Engine(cfg, ArraySource(x), DummyClassifier(8), {"mean": 0.0, "std": 1.0}, out_queue=q)
    eng.in_queue.put(ModeCommand("street"))
    eng.run()
    st = [m for m in list(q.queue) if isinstance(m, StatusUpdate)]
    assert st and st[-1].mode == "street" and st[-1].mode_selector == "street"
    cfg0 = load_config()
    q0: queue.Queue = queue.Queue()
    Engine(cfg0, ArraySource(x), DummyClassifier(8), {"mean": 0.0, "std": 1.0}, out_queue=q0).run()
    assert all(m.mode is None for m in list(q0.queue) if isinstance(m, StatusUpdate))


# ---------------------------------------------------------------- exports
def _ios():
    path = paths.ROOT / "ios/scripts/export_ios_assets.py"
    spec = importlib.util.spec_from_file_location("export_ios_assets", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_ios_export_modes_display_and_auto():
    ios = _ios()
    stats = json.loads((paths.ROOT / "assets/models/vn_v2/feature_stats.json").read_text())
    v2 = ios.export_extras(load_config(paths.ROOT / "config/soundwatch_vn_v2.yaml"), stats)["modes"]
    assert all(set(m) == {"disable", "level"} for m in v2["modes"].values())  # vn-v2 export unchanged
    assert set(v2["auto"]) == {"street_above_dbfs", "window_s", "hysteresis_db", "min_s"}
    v3 = ios.export_extras(load_config(V3), stats)["modes"]
    st = v3["modes"]["street"]
    assert st["label"]["siren"].startswith("Xe ưu tiên") and st["icon"] == {"siren": "siren.png"}
    assert st["sf_symbol"] == {"siren": "light.beacon.max.fill"}
    assert v3["modes"]["indoor"]["sf_symbol"] == {"siren": "flame.fill"}
    assert v3["auto"]["street_evidence"]["car_horn"] == 1.0 and "street_score_above" in v3["auto"]


def test_labels_header_mode_display_and_sound():
    from soundwatch.models.export_c import labels_header

    h2 = labels_header(load_config(paths.ROOT / "config/soundwatch_vn_v2.yaml"))
    assert "kModeLabelUtf8" not in h2 and "SW_AUTO_USE_SOUND" not in h2
    h3 = labels_header(load_config(V3))
    assert "kModeLabelUtf8" in h3 and "kModeIconName" in h3 and "kModeLabelAscii" in h3
    assert "SW_AUTO_USE_SOUND 1" in h3 and "kAutoEvidenceWeight" in h3
    assert '"Xe uu tien' in h3


# ---------------------------------------------------------------- train: shared stats only for default runs
def test_writes_shared_stats_only_for_default_runs():
    from soundwatch.models.train import writes_shared_stats

    assert writes_shared_stats("distill", False) and writes_shared_stats("nodistill", False)
    assert writes_shared_stats("v3_sm_lm_quick", True)
    assert not writes_shared_stats("v3_sm_lm", False)


def test_auto_config_dataclass_defaults():
    a = AutoModeConfig()
    assert a.uses_rms and not a.uses_sound and a.min_dwell_s == 0.0


# ---------------------------------------------------------------- simulator UI
def test_ui_shows_mode_label_and_cycles_modes(tmp_path):
    import os

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame

    from soundwatch.types import StatusUpdate
    from soundwatch.ui.screens import WatchUI

    pygame.init()
    cfg = load_config(V3)
    ui = WatchUI(cfg, queue.Queue(), show_debug=False)
    si = cfg.class_keys.index("siren")
    label = cfg.modes.spec("street").label_map["siren"]
    ev = AlertEvent(si, 0.97, 1.0, 1.0, level=2, label=label, icon="siren.png")
    ui.handle_message(StatusUpdate(1.0, -30.0, np.zeros(8, np.float32), False, 1.0, mode="street",
                                   mode_selector="auto"))
    ui.handle_message(ev)
    assert ui.screen == "alert" and ui.history[0].label == label and ui.history[0].icon == "siren.png"
    surf = pygame.Surface(ui.window_size)
    ui.draw(surf)  # long override label renders (wrapped / shrunk)
    ui.go("idle")
    ui.draw(surf)
    ui.cycle_mode()
    assert isinstance(ui.sent[-1], ModeCommand) and ui.sent[-1].mode in cfg.modes.names
    ui0 = WatchUI(load_config(), queue.Queue(), show_debug=False)
    ui0.cycle_mode()
    assert ui0.sent == []


# ---------------------------------------------------------------- rule 10: flip guard
def test_flip_guard(tmp_path):
    raw = yaml.safe_load((paths.ROOT / "config" / "soundwatch.yaml").read_text(encoding="utf-8"))
    for c in raw["classes"]:
        c["threshold"] = 0.5
    raw["postprocess"].update({"k_of_n": [1, 1], "flip_guard_hops": 3, "flip_guard_margin": 0.2})
    p = tmp_path / "fg.yaml"
    p.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    cfg = load_config(p)
    si, sm = cfg.class_keys.index("siren"), cfg.class_keys.index("smoke_alarm")

    def mix(s, m):
        x = np.zeros(8, np.float32)
        x[si], x[sm], x[-1] = s, m, 1 - s - m
        return x

    seq = [mix(0.9, 0.05)] * 4 + [mix(0.4, 0.55)] * 2 + [mix(0.1, 0.85)] * 6
    pp = PostProcessor(cfg.classes, cfg.postprocess)
    ev = _run(pp, seq)
    fresh = [(e.class_id, e.t_audio) for e in ev if not e.repeat]
    # smoke above threshold from hop 4 but only alerts after it dominated by > 0.2 for 3 hops (hops 6, 7, 8)
    assert fresh[0] == (si, 1.0) and (sm, 1.0 + 0.5 * 8) in fresh
    assert not any(c == sm and t < 1.0 + 0.5 * 8 for c, t in fresh) and pp.flip_suppressed > 0
    off = load_config()  # without the option nothing is withheld
    raw["postprocess"].pop("flip_guard_hops")
    p.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    cfg0 = load_config(p)
    ev0 = _run(PostProcessor(cfg0.classes, cfg0.postprocess), seq)
    assert (sm, 3.0) in [(e.class_id, e.t_audio) for e in ev0 if not e.repeat]
    assert off.postprocess.flip_guard_hops is None
