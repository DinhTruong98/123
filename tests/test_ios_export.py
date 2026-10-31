"""ios/scripts/export_ios_assets.py --suffix: a second model's files next to the default ones."""

from __future__ import annotations

import importlib.util
import json

from soundwatch import paths


def _load_script():
    spec = importlib.util.spec_from_file_location("eia", paths.ROOT / "ios/scripts/export_ios_assets.py")
    eia = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eia)
    return eia


def test_out_name():
    eia = _load_script()
    assert eia.out_name("soundwatch_config", ".json", None) == "soundwatch_config.json"
    assert eia.out_name("soundwatch_config", ".json", "vn2") == "soundwatch_config_vn2.json"


def test_suffix_writes_only_suffixed_files(tmp_path, monkeypatch):
    eia = _load_script()
    monkeypatch.setattr(eia, "OUT", tmp_path)
    assert eia.main([
        "--suffix", "vn2", "--config", str(paths.ROOT / "config/soundwatch_vn_v2.yaml"),
        "--model", str(paths.ROOT / "assets/models/vn_v2/student_int8.tflite"),
        "--stats", str(paths.ROOT / "assets/models/vn_v2/feature_stats.json"),
    ]) == 0
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "demo_expected_events_vn2.json", "golden_features_vn2.json",
        "soundwatch_config_vn2.json", "student_int8_vn2.tflite",
    ]
    cfg = json.loads((tmp_path / "soundwatch_config_vn2.json").read_text(encoding="utf-8"))
    assert cfg["modes"]["default"] == "indoor"
    assert cfg["postprocess"]["k_of_n"] == [2, 3] and cfg["postprocess"]["danger_fast_threshold"] == 0.95
    assert abs(cfg["feature_stats"]["mean"] - -2.148079443279617) < 1e-12


def test_no_suffix_keeps_default_names(tmp_path, monkeypatch):
    eia = _load_script()
    monkeypatch.setattr(eia, "OUT", tmp_path)
    assert eia.main([]) == 0
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "demo_expected_events.json", "demo_scenario.wav", "golden_features.json",
        "soundwatch_config.json", "student_int8.tflite",
    ]
    assert "modes" not in json.loads((tmp_path / "soundwatch_config.json").read_text(encoding="utf-8"))
