from soundwatch.config import load_config
from soundwatch.data import label_map as lm


def test_every_target_class_covered():
    cfg = load_config()
    ids = {lm.map_esc50(c, cfg) for c in lm.ESC50_CATEGORIES}
    ids |= {lm.map_us8k(c, cfg) for c in lm.US8K_TO_KEY}
    ids.discard(None)
    assert ids == set(range(cfg.num_classes))


def test_specific_mappings():
    cfg = load_config()
    bg = cfg.background_id
    assert lm.map_esc50("cat", cfg) == bg
    assert lm.map_esc50("clock_alarm", cfg) == cfg.class_keys.index("smoke_alarm")
    assert lm.map_esc50("toilet_flush", cfg) == cfg.class_keys.index("water_running")
    assert lm.map_esc50("dog", cfg) == lm.map_us8k("dog_bark", cfg)
    assert lm.map_us8k("gun_shot", cfg) is None
    assert lm.map_us8k("jackhammer", cfg) == bg
    assert len(lm.ESC50_CATEGORIES) == 50 and len(lm.US8K_TO_KEY) == 10
    assert set(lm.ESC50_TO_KEY) <= set(lm.ESC50_CATEGORIES)
    assert {k for k in lm.ESC50_TO_KEY.values()} | {k for k in lm.US8K_TO_KEY.values() if k} <= set(
        cfg.class_keys
    )
