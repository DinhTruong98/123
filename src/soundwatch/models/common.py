"""Shared helpers for WP3/WP4: threading, dataset loading, feature caching, metrics, JSON IO."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from soundwatch import paths
from soundwatch.config import AudioConfig, Config

SPLITS = ("train", "val", "test")


def setup_tf(threads: int | None = None, seed: int | None = None) -> None:
    """Limit TF CPU threads (other jobs share the machine) and fix seeds. Call before building models."""
    import tensorflow as tf

    n = int(threads or os.environ.get("SOUNDWATCH_TF_THREADS", 0) or max(1, (os.cpu_count() or 4) - 1))
    try:
        tf.config.threading.set_intra_op_parallelism_threads(n)
        tf.config.threading.set_inter_op_parallelism_threads(2)
    except RuntimeError:  # already initialized (e.g. in tests); harmless
        pass
    if seed is not None:
        import keras

        keras.utils.set_random_seed(seed)


def load_split(split: str, windows_dir: Path | None = None) -> dict[str, np.ndarray]:
    """Load artifacts/data/windows/<split>.npz -> {"wave", "label", "source_clip"} (wave kept float16)."""
    p = (windows_dir or paths.WINDOWS) / f"{split}.npz"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found (run `python -m soundwatch build-dataset`)")
    with np.load(p, allow_pickle=False) as d:
        out = {k: d[k] for k in d.files}
    out["label"] = out["label"].astype(np.int64)
    return out


def logmel_batch(waves: np.ndarray, cfg: AudioConfig) -> np.ndarray:
    """(N, 16000) any float dtype -> (N, n_frames, n_mels) float32 using WP2's feature code."""
    from soundwatch.audio.features import compute_logmel_batch

    return compute_logmel_batch(waves, cfg)


def source_fingerprint(split: str, windows_dir: Path | None = None) -> dict:
    """Identity of a windows/<split>.npz (size + mtime) so caches are rebuilt when the dataset changes."""
    p = (windows_dir or paths.WINDOWS) / f"{split}.npz"
    st = p.stat()
    return {"split": split, "src_size": int(st.st_size), "src_mtime_ns": int(st.st_mtime_ns)}


def _meta_path(cache: Path) -> Path:
    return cache.with_name(cache.name + ".meta.json")


def cache_is_valid(cache: Path, split: str, n: int) -> bool:
    meta = _meta_path(cache)
    if not cache.exists() or not meta.exists():
        return False
    try:
        m = read_json(meta)
    except (OSError, ValueError):
        return False
    return m.get("n") == n and all(m.get(k) == v for k, v in source_fingerprint(split).items())


def mark_cache(cache: Path, split: str, n: int, **extra) -> None:
    write_json(source_fingerprint(split) | {"n": int(n)} | extra, _meta_path(cache))


def cached_logmel(split: str, cfg: AudioConfig, waves: np.ndarray, cache_dir: Path | None = None,
                  refresh: bool = False) -> np.ndarray:
    """Log-mel for every window of a split, cached as CACHE/logmel_<split>.npy (keyed by N)."""
    p = (cache_dir or paths.CACHE) / f"logmel_{split}.npy"
    if not refresh and cache_is_valid(p, split, len(waves)):
        return np.load(p)
    t0 = time.time()
    x = logmel_batch(waves, cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, x)
    mark_cache(p, split, len(x))
    print(f"[features] {split}: {x.shape} in {time.time() - t0:.1f}s -> {p}")
    return x


def class_weights(labels: np.ndarray, num_classes: int,
                  clip: tuple[float, float] = (0.25, 4.0), power: float = 1.0) -> np.ndarray:
    """Inverse-frequency weights (N / (C * n_c)) ** power, clipped, renormalized to mean 1 over samples.
    power < 1 softens the weighting (0.5 = inverse square root)."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    w = (len(labels) / (num_classes * np.maximum(counts, 1.0))) ** power
    w = np.clip(w, *clip)
    w = w / (w[labels].mean() if len(labels) else 1.0)
    return w.astype(np.float32)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, cfg: Config) -> dict:
    """Schema shared by teacher/student/quant metrics JSON (consumed by WP7):

    {"n", "accuracy", "macro_f1",
     "per_class": {key: {"precision", "recall", "f1", "support"}},
     "danger_recall": {key: recall}, "danger_recall_min",
     "confusion": [[...]]  (rows = true, cols = predicted, class order = config)}
    """
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

    labels = list(range(cfg.num_classes))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    present = s > 0
    per = {
        c.key: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
        for i, c in enumerate(cfg.classes)
    }
    danger = {c.key: float(r[i]) for i, c in enumerate(cfg.classes) if int(c.level) == 2}
    return {
        "n": int(len(y_true)),
        "accuracy": float(np.mean(y_true == y_pred)) if len(y_true) else 0.0,
        "macro_f1": float(np.mean(f[present])) if present.any() else 0.0,
        "per_class": per,
        "danger_recall": danger,
        "danger_recall_min": float(min(danger.values())) if danger else 0.0,
        "confusion": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    from sklearn.metrics import f1_score

    present = np.unique(y_true)
    return float(f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0))


def write_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def read_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def format_metrics_row(name: str, m: dict, cfg: Config) -> str:
    rec = " ".join(f"{m['per_class'][k]['recall']:.2f}" for k in cfg.class_keys)
    return f"{name:<22} acc={m['accuracy']:.3f} macroF1={m['macro_f1']:.3f} recall[{rec}]"
