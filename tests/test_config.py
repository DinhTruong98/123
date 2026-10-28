import pytest
import yaml

from soundwatch import paths
from soundwatch.config import ConfigError, load_config
from soundwatch.types import Level


def test_default_config_loads():
    cfg = load_config()
    assert cfg.num_classes == 8
    assert cfg.classes[-1].key == "background"
    assert cfg.background_id == 7
    assert cfg.audio.window_samples == 16000
    assert cfg.audio.hop_samples == 8000
    assert cfg.audio.n_frames == 49
    assert cfg.classes[0].level is Level.DANGER
    assert all(c.id == i for i, c in enumerate(cfg.classes))


def _write(tmp_path, mutate):
    raw = yaml.safe_load(paths.CONFIG.read_text(encoding="utf-8"))
    mutate(raw)
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    return p


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r["classes"].reverse(),  # background not last
        lambda r: r["classes"][0].update(threshold=0.0),
        lambda r: r["classes"][0].update(vibration=[100, 100]),
        lambda r: r["classes"][1].update(key="smoke_alarm"),
        lambda r: r["postprocess"].update(k_of_n=[4, 3]),
    ],
)
def test_invalid_configs_rejected(tmp_path, mutate):
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, mutate))


def test_cli_help(capsys):
    from soundwatch.__main__ import main

    assert main(["--help"]) == 0
    assert "train-student" in capsys.readouterr().out
