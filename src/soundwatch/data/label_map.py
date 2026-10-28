"""WP1: map source-dataset class names to SoundWatch target classes.

Class names verified against ``ESC-50/meta/esc50.csv`` (column ``category``) and
``UrbanSound8K/metadata/UrbanSound8K.csv`` (column ``class``).

Target indices follow ``Config.class_keys`` (background is always last).
A mapping function returns ``None`` for source classes that are dropped (US8K ``gun_shot``).
ESC-50 ``cat`` is deliberately kept in background so the model learns cat != baby.

ESC-50 ``clock_alarm`` is a *proxy* for smoke_alarm (default). Once enough real smoke-alarm clips
exist (FSD50K, see :mod:`soundwatch.data.fsd50k`) it can be sent to background instead with
``clock_alarm_background=True`` (``build-dataset --clock-alarm background``).
FSD50K clips are pre-selected per target class, so :func:`map_fsd50k` maps a target key directly.
"""

from __future__ import annotations

from soundwatch.config import Config, load_config

BACKGROUND = "background"
DROP = None

# ESC-50 category -> target key (anything not listed is background)
ESC50_TO_KEY: dict[str, str] = {
    "clock_alarm": "smoke_alarm",  # proxy, see plan section 10
    "car_horn": "car_horn",
    "siren": "siren",
    "crying_baby": "baby_cry",
    "door_wood_knock": "door_knock",
    "dog": "dog_bark",
    "pouring_water": "water_running",
    "water_drops": "water_running",
    "toilet_flush": "water_running",
}

ESC50_CATEGORIES: tuple[str, ...] = (
    "airplane", "breathing", "brushing_teeth", "can_opening", "car_horn", "cat", "chainsaw",
    "chirping_birds", "church_bells", "clapping", "clock_alarm", "clock_tick", "coughing", "cow",
    "crackling_fire", "crickets", "crow", "crying_baby", "dog", "door_wood_creaks", "door_wood_knock",
    "drinking_sipping", "engine", "fireworks", "footsteps", "frog", "glass_breaking", "hand_saw",
    "helicopter", "hen", "insects", "keyboard_typing", "laughing", "mouse_click", "pig",
    "pouring_water", "rain", "rooster", "sea_waves", "sheep", "siren", "sneezing", "snoring",
    "thunderstorm", "toilet_flush", "train", "vacuum_cleaner", "washing_machine", "water_drops", "wind",
)  # fmt: skip

# UrbanSound8K class -> target key, or None to drop
US8K_TO_KEY: dict[str, str | None] = {
    "air_conditioner": BACKGROUND,
    "car_horn": "car_horn",
    "children_playing": BACKGROUND,
    "dog_bark": "dog_bark",
    "drilling": BACKGROUND,
    "engine_idling": BACKGROUND,
    "gun_shot": DROP,
    "jackhammer": BACKGROUND,
    "siren": "siren",
    "street_music": BACKGROUND,
}

_cfg: Config | None = None


def _config(cfg: Config | None) -> Config:
    global _cfg
    if cfg is not None:
        return cfg
    if _cfg is None:
        _cfg = load_config()
    return _cfg


def key_to_id(key: str, cfg: Config | None = None) -> int:
    return _config(cfg).class_keys.index(key)


FSD50K_TARGETS: tuple[str, ...] = ("baby_cry", "smoke_alarm")


def esc50_key(category: str, clock_alarm_background: bool = False) -> str:
    if category not in ESC50_CATEGORIES:
        raise KeyError(f"unknown ESC-50 category: {category!r}")
    if clock_alarm_background and category == "clock_alarm":
        return BACKGROUND
    return ESC50_TO_KEY.get(category, BACKGROUND)


def us8k_key(cls: str) -> str | None:
    if cls not in US8K_TO_KEY:
        raise KeyError(f"unknown UrbanSound8K class: {cls!r}")
    return US8K_TO_KEY[cls]


def map_esc50(category: str, cfg: Config | None = None, clock_alarm_background: bool = False) -> int | None:
    """ESC-50 ``category`` -> target class id (never None: unmapped -> background)."""
    return key_to_id(esc50_key(category, clock_alarm_background), cfg)


def map_fsd50k(target: str, cfg: Config | None = None) -> int:
    """FSD50K selection target (``baby_cry`` / ``smoke_alarm``) -> target class id."""
    if target not in FSD50K_TARGETS:
        raise KeyError(f"unknown FSD50K target: {target!r}")
    return key_to_id(target, cfg)


def map_us8k(cls: str, cfg: Config | None = None) -> int | None:
    """UrbanSound8K ``class`` -> target class id, or None if dropped (gun_shot)."""
    k = us8k_key(cls)
    return None if k is None else key_to_id(k, cfg)


def map_label(dataset: str, name: str, cfg: Config | None = None) -> int | None:
    if dataset == "esc50":
        return map_esc50(name, cfg)
    if dataset == "us8k":
        return map_us8k(name, cfg)
    if dataset == "fsd50k":
        return map_fsd50k(name, cfg)
    raise KeyError(f"unknown dataset {dataset!r}")


def is_alert(class_id: int, cfg: Config | None = None) -> bool:
    return class_id != _config(cfg).background_id
