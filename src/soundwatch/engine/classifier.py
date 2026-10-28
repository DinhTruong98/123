"""Classifiers: (n_frames, n_mels) normalized features -> probabilities. Owner: WP5."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Classifier(Protocol):
    num_classes: int

    def predict(self, features: np.ndarray) -> np.ndarray: ...


class TFLiteClassifier:
    """Runs a .tflite model; quantizes int8 input / dequantizes int8 output using the model's params.
    Output is a probability vector (apply softmax only if the model outputs logits)."""

    def __init__(self, model_path, num_threads: int = 1):
        raise NotImplementedError("WP5")


class KerasClassifier:
    def __init__(self, model_path):
        raise NotImplementedError("WP5")


class DummyClassifier:
    """Deterministic fake. `schedule` is a list of (t_start, t_end, class_id); outside every interval
    it predicts background. The caller advances time via `set_time(t_audio)` before `predict`."""

    def __init__(self, num_classes: int, schedule: list[tuple[float, float, int]] | None = None,
                 confidence: float = 0.9):
        raise NotImplementedError("WP5")
