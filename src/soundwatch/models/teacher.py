"""WP3 teacher: YAMNet (AudioSet) mean embeddings per 1 s window + small classifier head.

Outputs
  artifacts/cache/yamnet_{split}.npz        emb (N, 1024) float32, label (N,)
  artifacts/models/teacher_head.keras       Dense(256, relu) -> Dropout(0.3) -> Dense(C) (logits)
  artifacts/cache/teacher_logits_{split}.npy  teacher logits (train = clean windows, used for KD)
  artifacts/models/teacher_metrics.json     val/test metrics (schema: common.classification_metrics)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from soundwatch import paths
from soundwatch.config import load_config
from soundwatch.models import common


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--refresh", action="store_true", help="recompute cached YAMNet embeddings")
    parser.add_argument("--threads", type=int, default=None)
    common.add_data_dir_args(parser)
    parser.add_argument("--models-dir", default=None,
                        help="teacher_head.keras / teacher_metrics.json directory (default artifacts/models)")


def yamnet_embeddings(split: str, waves: np.ndarray, labels: np.ndarray, refresh: bool = False,
                      cache_dir: Path | None = None, model=None) -> np.ndarray:  # fmt: skip
    p = (cache_dir or paths.CACHE) / f"yamnet_{split}.npz"
    if not refresh and common.cache_is_valid(p, split, len(waves)):
        with np.load(p) as d:
            return d["emb"]
    if model is None:
        from soundwatch.models.yamnet_loader import Yamnet

        model = Yamnet()
    t0 = time.time()
    emb = model.embed(waves, batch_size=64, progress=True).astype(np.float32)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, emb=emb, label=labels)
    common.mark_cache(p, split, len(emb))
    print(f"[teacher] {split}: embeddings {emb.shape} in {time.time() - t0:.0f}s -> {p}")
    return emb


def build_head(num_classes: int, emb_dim: int = 1024, hidden: int = 256, dropout: float = 0.3):
    import keras
    from keras import layers

    inp = keras.Input(shape=(emb_dim,), name="yamnet_embedding")
    x = layers.Dense(hidden, activation="relu", name="fc1")(inp)
    x = layers.Dropout(dropout, name="dropout")(x)
    out = layers.Dense(num_classes, name="logits")(x)
    return keras.Model(inp, out, name="teacher_head")


def train_head(x_tr: np.ndarray, y_tr: np.ndarray, x_va: np.ndarray, y_va: np.ndarray, num_classes: int,
               epochs: int = 120, batch_size: int = 256, patience: int = 20,
               log_dir: Path | None = None):  # fmt: skip
    import keras

    from soundwatch.models.train import MacroF1Monitor

    model = build_head(num_classes, emb_dim=x_tr.shape[1])
    steps = max(1, int(np.ceil(len(x_tr) / batch_size))) * epochs
    sched = keras.optimizers.schedules.CosineDecay(1e-3, decay_steps=steps, alpha=0.01)
    model.compile(optimizer=keras.optimizers.Adam(sched),
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    cw = common.class_weights(y_tr, num_classes)
    mon = MacroF1Monitor(x_va, y_va, num_classes, patience=patience,
                         csv_path=(log_dir / "epochs.csv") if log_dir else None)
    model.fit(x_tr, y_tr, sample_weight=cw[y_tr], batch_size=batch_size, epochs=epochs, verbose=0,
              callbacks=[mon], shuffle=True)
    mon.restore_best(model)
    return model, mon


def main(args: argparse.Namespace) -> int:
    common.apply_data_dirs(args)
    if getattr(args, "models_dir", None):
        paths.MODELS = Path(args.models_dir).resolve()
    cfg = load_config(args.config)
    common.setup_tf(args.threads, seed=args.seed)
    t_start = time.time()
    data = {s: common.load_split(s) for s in common.SPLITS}
    emb = {}
    yam = None
    for s in common.SPLITS:
        cache = paths.CACHE / f"yamnet_{s}.npz"
        if yam is None and (args.refresh or not common.cache_is_valid(cache, s, len(data[s]["label"]))):
            from soundwatch.models.yamnet_loader import Yamnet

            yam = Yamnet()
        emb[s] = yamnet_embeddings(s, data[s]["wave"], data[s]["label"], args.refresh, model=yam)
    t_emb = time.time() - t_start

    log_dir = paths.LOGS / "teacher"
    t0 = time.time()
    model, mon = train_head(emb["train"], data["train"]["label"], emb["val"], data["val"]["label"],
                            cfg.num_classes, args.epochs, args.batch_size, args.patience, log_dir)
    t_train = time.time() - t0
    paths.MODELS.mkdir(parents=True, exist_ok=True)
    model.save(paths.MODELS / "teacher_head.keras")

    metrics: dict = {"model": "yamnet+head", "best_epoch": mon.best_epoch, "epochs_run": mon.epochs_run,
                     "embedding_time_s": round(t_emb, 1), "train_time_s": round(t_train, 1)}
    for s in common.SPLITS:
        logits = model.predict(emb[s], batch_size=1024, verbose=0).astype(np.float32)
        lp = paths.CACHE / f"teacher_logits_{s}.npy"
        np.save(lp, logits)
        common.mark_cache(lp, s, len(logits))
        if s != "train":
            metrics[s] = common.classification_metrics(data[s]["label"], logits.argmax(1), cfg)
            print(common.format_metrics_row(f"teacher {s}", metrics[s], cfg))
    common.write_json(metrics, paths.MODELS / "teacher_metrics.json")
    common.write_json({"epochs": args.epochs, "batch_size": args.batch_size, "patience": args.patience,
                       "seed": args.seed, "head": "Dense(256,relu)-Dropout(0.3)-Dense(C)"},
                      log_dir / "config.json")
    print(f"[teacher] saved {paths.MODELS / 'teacher_head.keras'}, logits -> {paths.CACHE}")
    return 0
