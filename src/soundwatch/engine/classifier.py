"""Classifiers: (n_frames, n_mels) normalized features -> probabilities. Owner: WP5."""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Protocol

import numpy as np


class Classifier(Protocol):
    num_classes: int

    def predict(self, features: np.ndarray) -> np.ndarray: ...


def softmax(x: np.ndarray) -> np.ndarray:
    z = np.asarray(x, dtype=np.float64)
    z = z - z.max()
    e = np.exp(z)
    return (e / e.sum()).astype(np.float32)


def as_probs(y: np.ndarray, tol: float = 0.05) -> np.ndarray:
    """Return `y` as a probability vector: unchanged if it already looks like one
    (non-negative, sums to 1 +- tol), else softmax(y)."""
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if np.all(y >= -1e-6) and abs(float(y.sum()) - 1.0) <= tol:
        return np.clip(y, 0.0, 1.0)
    return softmax(y)


def _make_interpreter(model_path: str, num_threads: int):
    """LiteRT interpreter if `ai_edge_litert` is installed, else tf.lite.Interpreter (its
    deprecation warning is silenced; the API is identical)."""
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        import tensorflow as tf

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return tf.lite.Interpreter(model_path=model_path, num_threads=num_threads)
    return Interpreter(model_path=model_path, num_threads=num_threads)


class TFLiteClassifier:
    """Runs a .tflite model; quantizes int8 input / dequantizes int8 output using the model's params.
    Output is a probability vector (apply softmax only if the model outputs logits).

    The (n_frames, n_mels) feature array is reshaped to the model's input shape (e.g. (1, 49, 40, 1)).
    `last_infer_ms` holds the duration of the last `predict` (quantize + invoke + dequantize).
    """

    def __init__(self, model_path, num_threads: int = 1):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(self.model_path)
        self._interp = _make_interpreter(str(self.model_path), num_threads)
        self._interp.allocate_tensors()
        inp = self._interp.get_input_details()[0]
        out = self._interp.get_output_details()[0]
        self._in_index = inp["index"]
        self._out_index = out["index"]
        self.input_shape = tuple(int(v) for v in inp["shape"])
        self.input_dtype = np.dtype(inp["dtype"])
        self.output_dtype = np.dtype(out["dtype"])
        self._in_scale, self._in_zp = (float(v) for v in inp["quantization"])
        self._out_scale, self._out_zp = (float(v) for v in out["quantization"])
        self.num_classes = int(out["shape"][-1])
        self._in_buf = np.zeros(self.input_shape, dtype=self.input_dtype)
        self.last_infer_ms = 0.0

    @property
    def is_quantized(self) -> bool:
        return np.issubdtype(self.input_dtype, np.integer)

    def _quantize_input(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=np.float32)
        if x.size != self._in_buf.size:
            raise ValueError(f"features of shape {x.shape} do not fit model input {self.input_shape}")
        x = x.reshape(self.input_shape)
        if self.is_quantized:
            info = np.iinfo(self.input_dtype)
            scale = self._in_scale if self._in_scale else 1.0
            q = np.round(x / scale + self._in_zp)
            np.clip(q, info.min, info.max, out=q)
            self._in_buf[...] = q.astype(self.input_dtype)
        else:
            self._in_buf[...] = x
        return self._in_buf

    def predict(self, features: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        self._interp.set_tensor(self._in_index, self._quantize_input(features))
        self._interp.invoke()
        y = self._interp.get_tensor(self._out_index).reshape(-1)
        if np.issubdtype(self.output_dtype, np.integer):
            scale = self._out_scale if self._out_scale else 1.0
            y = (y.astype(np.float32) - self._out_zp) * scale
        probs = as_probs(y)
        self.last_infer_ms = (time.perf_counter() - t0) * 1000.0
        return probs


class KerasClassifier:
    """Float Keras model (.keras / .h5), for comparison with the int8 TFLite model."""

    def __init__(self, model_path):
        import keras

        self.model_path = Path(model_path)
        self._model = keras.models.load_model(str(self.model_path), compile=False)
        shape = self._model.input_shape
        if isinstance(shape, list):
            shape = shape[0]
        self.input_shape = (1, *[int(v) for v in shape[1:]])
        self.num_classes = int(self._model.output_shape[-1])
        self.last_infer_ms = 0.0

    def predict(self, features: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        x = np.asarray(features, dtype=np.float32).reshape(self.input_shape)
        y = np.asarray(self._model(x, training=False)).reshape(-1)
        probs = as_probs(y)
        self.last_infer_ms = (time.perf_counter() - t0) * 1000.0
        return probs


class DummyClassifier:
    """Deterministic fake. `schedule` is a list of (t_start, t_end, class_id); outside every interval
    it predicts background. The caller advances time via `set_time(t_audio)` before `predict`.

    The scheduled class (first matching interval, t_start <= t < t_end) gets probability
    `confidence`; the remaining `1 - confidence` is spread evenly over the other classes.
    Background is the last class (num_classes - 1).
    """

    def __init__(self, num_classes: int, schedule: list[tuple[float, float, int]] | None = None,
                 confidence: float = 0.9):
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2")
        self.num_classes = int(num_classes)
        self.schedule = [(float(a), float(b), int(c)) for a, b, c in (schedule or [])]
        for _, _, c in self.schedule:
            if not 0 <= c < self.num_classes:
                raise ValueError(f"class id {c} out of range")
        self.confidence = float(confidence)
        self.t = 0.0
        self.last_infer_ms = 0.0

    @property
    def background_id(self) -> int:
        return self.num_classes - 1

    def set_time(self, t: float) -> None:
        self.t = float(t)

    def class_at(self, t: float) -> int:
        for a, b, c in self.schedule:
            if a <= t < b:
                return c
        return self.background_id

    def predict(self, features: np.ndarray | None = None) -> np.ndarray:
        c = self.class_at(self.t)
        rest = (1.0 - self.confidence) / (self.num_classes - 1)
        p = np.full(self.num_classes, rest, dtype=np.float32)
        p[c] = self.confidence
        return p


def load_classifier(model_path, num_threads: int = 1) -> TFLiteClassifier | KerasClassifier:
    """.tflite -> TFLiteClassifier, anything else (.keras / .h5) -> KerasClassifier."""
    p = Path(model_path)
    if p.suffix == ".tflite":
        return TFLiteClassifier(p, num_threads=num_threads)
    return KerasClassifier(p)
