"""YAMNet (AudioSet, 521 classes) rebuilt for Keras 3, loading the original `yamnet.h5` weights.

Adapted from tensorflow/models research/audioset/yamnet (yamnet.py, params.py, features.py),
Copyright 2019 The TensorFlow Authors, licensed under the Apache License, Version 2.0
(http://www.apache.org/licenses/LICENSE-2.0). Changes: Keras 3 layer API, layer names without "/",
weights assigned by name from the HDF5 file with h5py, batched feature front-end (several 1 s windows
at once) instead of the single-waveform Keras graph.

Only used offline as the teacher (WP3). tfhub.dev is not reachable in the dev environment, so the
weights + class map are downloaded from storage.googleapis.com / raw.githubusercontent.com into
artifacts/cache/yamnet/.
"""

from __future__ import annotations

import csv
import math
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from soundwatch import paths

YAMNET_DIR = paths.CACHE / "yamnet"
WEIGHTS_URL = "https://storage.googleapis.com/audioset/yamnet.h5"
CLASS_MAP_URL = (
    "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"
)


@dataclass(frozen=True)
class YamnetParams:
    sample_rate: float = 16000.0
    stft_window_seconds: float = 0.025
    stft_hop_seconds: float = 0.010
    mel_bands: int = 64
    mel_min_hz: float = 125.0
    mel_max_hz: float = 7500.0
    log_offset: float = 0.001
    patch_window_seconds: float = 0.96
    patch_hop_seconds: float = 0.48
    num_classes: int = 521
    batchnorm_epsilon: float = 1e-4

    @property
    def patch_frames(self) -> int:
        return int(round(self.patch_window_seconds / self.stft_hop_seconds))


# (kind, kernel, stride, filters) — identical to _YAMNET_LAYER_DEFS upstream
_LAYER_DEFS = [
    ("conv", 3, 2, 32),
    ("sep", 3, 1, 64),
    ("sep", 3, 2, 128),
    ("sep", 3, 1, 128),
    ("sep", 3, 2, 256),
    ("sep", 3, 1, 256),
    ("sep", 3, 2, 512),
    ("sep", 3, 1, 512),
    ("sep", 3, 1, 512),
    ("sep", 3, 1, 512),
    ("sep", 3, 1, 512),
    ("sep", 3, 1, 512),
    ("sep", 3, 2, 1024),
    ("sep", 3, 1, 1024),
]


def ensure_files(directory: Path = YAMNET_DIR) -> Path:
    """Download yamnet.h5 + class map if missing. Returns the directory."""
    directory.mkdir(parents=True, exist_ok=True)
    for url in (WEIGHTS_URL, CLASS_MAP_URL):
        dst = directory / url.rsplit("/", 1)[1]
        if not dst.exists() or dst.stat().st_size == 0:
            print(f"[yamnet] downloading {url}")
            tmp = dst.with_suffix(dst.suffix + ".part")
            urllib.request.urlretrieve(url, tmp)
            tmp.rename(dst)
    return directory


def class_names(directory: Path = YAMNET_DIR) -> list[str]:
    with open(directory / "yamnet_class_map.csv", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        return [row[2] for row in reader]


def _h5_name(keras_name: str) -> str:
    return keras_name.replace("__", "/")


def build_core(params: YamnetParams | None = None):
    """Keras 3 model: (B, 96, 64) log-mel patches -> (scores (B,521), embeddings (B,1024))."""
    import keras
    from keras import layers

    params = params or YamnetParams()
    inp = keras.Input(shape=(params.patch_frames, params.mel_bands), name="patches")
    x = layers.Reshape((params.patch_frames, params.mel_bands, 1))(inp)

    def bn(name: str):
        return layers.BatchNormalization(name=name, center=True, scale=False,
                                         epsilon=params.batchnorm_epsilon)

    for i, (kind, k, s, f) in enumerate(_LAYER_DEFS):
        n = f"layer{i + 1}"
        if kind == "conv":
            x = layers.Conv2D(f, k, strides=s, padding="same", use_bias=False, name=f"{n}__conv")(x)
            x = bn(f"{n}__conv__bn")(x)
            x = layers.ReLU()(x)
        else:
            x = layers.DepthwiseConv2D(k, strides=s, padding="same", use_bias=False,
                                       name=f"{n}__depthwise_conv")(x)
            x = bn(f"{n}__depthwise_conv__bn")(x)
            x = layers.ReLU()(x)
            x = layers.Conv2D(f, 1, padding="same", use_bias=False, name=f"{n}__pointwise_conv")(x)
            x = bn(f"{n}__pointwise_conv__bn")(x)
            x = layers.ReLU()(x)
    emb = layers.GlobalAveragePooling2D(name="embeddings")(x)
    logits = layers.Dense(params.num_classes, name="logits")(emb)
    scores = layers.Activation("sigmoid", name="scores")(logits)
    return keras.Model(inp, [scores, emb], name="yamnet_core")


def load_h5_weights(model, h5_path: Path) -> None:
    """Assign weights from the original tf.keras H5 file by layer name (h5py ships with TF)."""
    import h5py

    with h5py.File(h5_path, "r") as f:
        for layer in model.layers:
            if not layer.weights:
                continue
            name = _h5_name(layer.name)
            g = f[name][name]
            # map variable short names (kernel, depthwise_kernel, bias, beta, moving_mean, ...)
            stored = {k.split(":")[0]: np.asarray(g[k]) for k in g.keys()}
            new = []
            for w in layer.weights:
                short = w.path.rsplit("/", 1)[-1]
                if short == "kernel" and "depthwise_kernel" in stored:  # Keras 3 renamed it
                    short = "depthwise_kernel"
                if short not in stored:
                    raise KeyError(f"{name}: weight {short} not in h5 (have {sorted(stored)})")
                arr = stored[short]
                if tuple(arr.shape) != tuple(w.shape):
                    raise ValueError(f"{name}/{short}: shape {arr.shape} != {w.shape}")
                new.append(arr)
            layer.set_weights(new)


class Yamnet:
    """Batched YAMNet: waveforms (B, n) float32 16 kHz -> per-window mean embedding / scores."""

    def __init__(self, directory: Path = YAMNET_DIR, params: YamnetParams | None = None):
        import tensorflow as tf

        params = params or YamnetParams()
        self.params = params
        directory = ensure_files(directory)
        self.core = build_core(params)
        load_h5_weights(self.core, directory / "yamnet.h5")
        self._names: list[str] | None = None
        self._dir = directory
        p = params
        self._win = int(round(p.sample_rate * p.stft_window_seconds))  # 400
        self._hop = int(round(p.sample_rate * p.stft_hop_seconds))  # 160
        self._fft = 2 ** int(math.ceil(math.log2(self._win)))  # 512
        self._mel = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=p.mel_bands, num_spectrogram_bins=self._fft // 2 + 1,
            sample_rate=p.sample_rate, lower_edge_hertz=p.mel_min_hz, upper_edge_hertz=p.mel_max_hz)
        self._run = tf.function(self._forward, reduce_retracing=True)

    @property
    def names(self) -> list[str]:
        if self._names is None:
            self._names = class_names(self._dir)
        return self._names

    def padded_length(self, n: int) -> int:
        """Same rule as upstream features.pad_waveform."""
        p = self.params
        min_n = int(p.sample_rate * (p.patch_window_seconds + p.stft_window_seconds - p.stft_hop_seconds))
        n = max(n, min_n)
        hop = int(p.patch_hop_seconds * p.sample_rate)
        return min_n + hop * math.ceil((n - min_n) / hop)

    def _forward(self, wave):
        import tensorflow as tf

        p = self.params
        spec = tf.abs(tf.signal.stft(wave, frame_length=self._win, frame_step=self._hop,
                                     fft_length=self._fft))
        logmel = tf.math.log(tf.matmul(spec, self._mel) + p.log_offset)  # (B, T, 64)
        patch_hop = int(round(p.patch_hop_seconds / p.stft_hop_seconds))
        patches = tf.signal.frame(logmel, p.patch_frames, patch_hop, axis=1)
        b = tf.shape(patches)[0]
        k = tf.shape(patches)[1]
        flat = tf.reshape(patches, (-1, p.patch_frames, p.mel_bands))
        scores, emb = self.core(flat, training=False)
        scores = tf.reshape(scores, (b, k, -1))
        emb = tf.reshape(emb, (b, k, -1))
        return tf.reduce_mean(scores, axis=1), tf.reduce_mean(emb, axis=1)

    def __call__(self, waves: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """waves (B, n) -> (mean scores (B, 521), mean embeddings (B, 1024)), float32."""
        waves = np.asarray(waves, dtype=np.float32)
        if waves.ndim == 1:
            waves = waves[None]
        target = self.padded_length(waves.shape[1])
        if target > waves.shape[1]:
            waves = np.pad(waves, ((0, 0), (0, target - waves.shape[1])))
        s, e = self._run(waves)
        return s.numpy(), e.numpy()

    def embed(self, waves: np.ndarray, batch_size: int = 64, progress: bool = False) -> np.ndarray:
        out = []
        n = len(waves)
        for i in range(0, n, batch_size):
            _, e = self(np.asarray(waves[i:i + batch_size], dtype=np.float32))
            out.append(e)
            if progress and (i // batch_size) % 50 == 0:
                print(f"[yamnet] {min(i + batch_size, n)}/{n}", flush=True)
        return np.concatenate(out, axis=0) if out else np.zeros((0, 1024), np.float32)

    def top_classes(self, wave: np.ndarray, k: int = 5) -> list[tuple[str, float]]:
        s, _ = self(wave[None] if wave.ndim == 1 else wave[:1])
        idx = np.argsort(-s[0])[:k]
        return [(self.names[i], float(s[0, i])) for i in idx]
