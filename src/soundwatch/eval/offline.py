"""WP7: offline (window-level) metrics on the test split for every available model.

Models (skipped with a note when missing):
  teacher            artifacts/models/teacher_head.keras on cached YAMNet embeddings (cache/yamnet_test.npz)
  student_distill    artifacts/models/student_distill.keras      (float Keras, distilled)
  student_nodistill  artifacts/models/student_nodistill.keras    (float Keras, no distillation)
  student_float_tflite artifacts/models/student_float.tflite
  student_int8       artifacts/models/student_int8.tflite        (the deployed model)

Students get normalized log-mel (WP2 compute_logmel_batch + normalize with feature_stats.json).
Outputs: reports/offline_metrics.json and reports/offline_<model>.png (row-normalized confusion
matrix with Vietnamese labels).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from soundwatch import paths
from soundwatch.config import Config, load_config
from soundwatch.types import Level

_FONT_SET = False


def setup_font() -> str | None:
    """Register Be Vietnam Pro (diacritics) for matplotlib if present; else DejaVu Sans."""
    global _FONT_SET
    import matplotlib
    from matplotlib import font_manager

    name = None
    for f in ("BeVietnamPro-Regular.ttf", "BeVietnamPro-Bold.ttf"):
        p = paths.FONTS / f
        if p.exists():
            try:
                font_manager.fontManager.addfont(str(p))
                name = font_manager.FontProperties(fname=str(p)).get_name()
                break
            except Exception:  # pragma: no cover - broken font file
                continue
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = ([name] if name else []) + ["DejaVu Sans"]
    _FONT_SET = True
    return name


def metrics_from_predictions(y_true: np.ndarray, y_pred: np.ndarray, cfg: Config) -> dict:
    """Schema of models.common.classification_metrics + danger_recall_mean + normalized confusion."""
    from soundwatch.models.common import classification_metrics

    m = classification_metrics(np.asarray(y_true), np.asarray(y_pred), cfg)
    d = list(m["danger_recall"].values())
    m["danger_recall_mean"] = float(np.mean(d)) if d else 0.0
    cm = np.asarray(m["confusion"], dtype=np.float64)
    m["confusion_normalized"] = (cm / np.maximum(cm.sum(1, keepdims=True), 1)).round(4).tolist()
    return m


def plot_confusion(m: dict, cfg: Config, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    setup_font()
    cmn = np.asarray(m["confusion_normalized"])
    cm = np.asarray(m["confusion"])
    labels = [c.label_vi for c in cfg.classes]
    n = len(labels)
    fig, ax = plt.subplots(figsize=(1.0 * n + 3, 0.85 * n + 2.5))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            if cm[i, j] == 0:
                continue
            ax.text(
                j,
                i,
                f"{cmn[i, j]:.2f}\n({cm[i, j]})",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if cmn[i, j] > 0.5 else "black",
            )
    ax.set_xticks(range(n), labels, rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(n), labels, fontsize=9)
    for t, c in zip(ax.get_yticklabels(), cfg.classes, strict=True):
        if c.level == Level.DANGER:
            t.set_color("#c62828")
    ax.set_xlabel("Dự đoán")
    ax.set_ylabel("Thực tế")
    ax.set_title(
        f"{title}\nacc={m['accuracy']:.3f}  macro-F1={m['macro_f1']:.3f}  "
        f"recall DANGER min={m['danger_recall_min']:.2f}",
        fontsize=10,
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------- inference
def predict_keras(model_path: Path, x: np.ndarray, batch_size: int = 256) -> np.ndarray:
    import keras

    model = keras.models.load_model(str(model_path), compile=False)
    return np.asarray(model.predict(x, batch_size=batch_size, verbose=0), dtype=np.float32)


def predict_tflite(model_path: Path, x: np.ndarray) -> np.ndarray:
    from soundwatch.engine.classifier import TFLiteClassifier

    clf = TFLiteClassifier(model_path, num_threads=1)
    return np.stack([clf.predict(xi) for xi in x]).astype(np.float32)


def student_models(models_dir: Path) -> list[tuple[str, Path, str]]:
    return [
        ("student_distill", models_dir / "student_distill.keras", "keras"),
        ("student_nodistill", models_dir / "student_nodistill.keras", "keras"),
        ("student_float_tflite", models_dir / "student_float.tflite", "tflite"),
        ("student_int8", models_dir / "student_int8.tflite", "tflite"),
    ]


def load_test_features(cfg: Config, split: str, stats: dict) -> tuple[np.ndarray, np.ndarray]:
    """Normalized (N, 49, 40, 1) features + labels. Reuses the WP3 log-mel cache when valid
    (never writes it: other jobs share the cache)."""
    from soundwatch.audio.features import compute_logmel_batch, normalize
    from soundwatch.models import common

    d = common.load_split(split)
    cache = paths.CACHE / f"logmel_{split}.npy"
    lm = None
    try:
        if common.cache_is_valid(cache, split, len(d["label"])):
            lm = np.load(cache)
    except (OSError, ValueError):
        lm = None
    if lm is None:
        lm = compute_logmel_batch(d["wave"], cfg.audio)
    return normalize(lm, stats)[..., None], d["label"]


def evaluate(
    cfg: Config,
    split: str = "test",
    models_dir: Path | None = None,
    out_dir: Path | None = None,
    verbose: bool = True,
) -> dict:
    from soundwatch.audio.features import load_stats
    from soundwatch.models import common

    models_dir = Path(models_dir) if models_dir else paths.MODELS
    out_dir = Path(out_dir) if out_dir else paths.REPORTS
    res: dict = {"split": split, "models": {}, "missing": {}}
    try:
        y_all = common.load_split(split)["label"]
        res["n"] = int(len(y_all))
    except FileNotFoundError as e:
        res["error"] = str(e)
        y_all = None

    preds: dict[str, np.ndarray] = {}
    # teacher on cached YAMNet embeddings
    tpath = models_dir / "teacher_head.keras"
    epath = paths.CACHE / f"yamnet_{split}.npz"
    if y_all is None:
        pass
    elif not tpath.exists():
        res["missing"]["teacher"] = f"{tpath.name} not found"
    elif not epath.exists():
        res["missing"]["teacher"] = f"{epath.name} (YAMNet embeddings cache) not found"
    else:
        with np.load(epath) as d:
            emb = d["emb"]
        if len(emb) != len(y_all):
            res["missing"]["teacher"] = f"embedding cache has {len(emb)} rows, split has {len(y_all)}"
        else:
            preds["teacher"] = predict_keras(tpath, emb, 1024)

    stats = None
    spath = models_dir / "feature_stats.json"
    if spath.exists():
        stats = load_stats(spath)
    x = None
    for name, mpath, kind in student_models(models_dir):
        if y_all is None:
            break
        if not mpath.exists():
            res["missing"][name] = f"{mpath.name} not found"
            continue
        if stats is None:
            res["missing"][name] = "feature_stats.json not found"
            continue
        if x is None:
            x, _ = load_test_features(cfg, split, stats)
        try:
            preds[name] = predict_keras(mpath, x) if kind == "keras" else predict_tflite(mpath, x)
        except Exception as e:  # noqa: BLE001 - keep evaluating other models
            res["missing"][name] = f"failed: {e!r}"

    for name, p in preds.items():
        m = metrics_from_predictions(y_all, p.argmax(1), cfg)
        src = tpath if name == "teacher" else dict((n, mp) for n, mp, _ in student_models(models_dir))[name]
        m["path"] = str(src.relative_to(paths.ROOT)) if src.is_relative_to(paths.ROOT) else str(src)
        png = out_dir / f"offline_{name}.png"
        plot_confusion(m, cfg, png, name)
        m["confusion_png"] = png.name
        res["models"][name] = m
        if verbose:
            print(common.format_metrics_row(name, m, cfg))
    if "student_int8" in preds:
        q = res["models"]["student_int8"]
        for ref in ("student_float_tflite", "student_distill"):
            if ref in res["models"]:
                f = res["models"][ref]
                res["int8_drop"] = {
                    "reference": ref,
                    "accuracy_pts": round(100 * (f["accuracy"] - q["accuracy"]), 3),
                    "macro_f1_pts": round(100 * (f["macro_f1"] - q["macro_f1"]), 3),
                    "top1_agreement": float(np.mean(preds["student_int8"].argmax(1) == preds[ref].argmax(1))),
                }
                break
    for name, why in res["missing"].items():
        if verbose:
            print(f"[eval] skip {name}: {why}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "offline_metrics.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    return res


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--split", default="test")
    parser.add_argument("--threads", type=int, default=2, help="TF threads (machine is shared)")


def main(args: argparse.Namespace) -> int:
    from soundwatch.models.common import setup_tf

    cfg = load_config(args.config)
    try:
        setup_tf(args.threads)
    except Exception:  # noqa: BLE001 - TF missing is handled per model
        pass
    res = evaluate(cfg, args.split)
    print(f"[eval] {len(res['models'])} models evaluated -> {paths.REPORTS / 'offline_metrics.json'}")
    return 0
