"""WP5/WP7-era post-processing settings, so the engine/post-processor tests keep exercising
smoothing + k-of-n independently of the values tuned into config/soundwatch.yaml (WP8)."""

import dataclasses

from soundwatch.config import Config, load_config

WP7_THRESHOLDS = {
    "smoke_alarm": 0.60,
    "car_horn": 0.60,
    "siren": 0.60,
    "baby_cry": 0.70,
    "door_knock": 0.70,
    "dog_bark": 0.75,
    "water_running": 0.75,
}


def legacy_config() -> Config:
    cfg = load_config()
    pp = dataclasses.replace(cfg.postprocess, smooth_n=3, k_of_n=(2, 3), bg_margin=None)
    classes = tuple(
        dataclasses.replace(c, threshold=WP7_THRESHOLDS[c.key]) if c.key in WP7_THRESHOLDS else c
        for c in cfg.classes
    )
    return dataclasses.replace(cfg, postprocess=pp, classes=classes)
