"""WP3 student training: DS-CNN on normalized log-mel, optional knowledge distillation from YAMNet.

Data flow
  windows/{train,val,test}.npz --(WP2 compute_logmel_batch)--> cache/logmel_{split}.npy (clean)
  train waves --(WP1 augment_wave: shift, noise SNR 0-20 dB, gain; K copies, offline once)-->
      cache/logmel_train_aug{k}.npy (float16)
  per epoch: every train window appears once, as a random version (clean or one of the K copies),
      normalized with feature_stats.json, SpecAugment applied online (cheap, numpy).

Outputs (tag defaults to "distill" / "nodistill")
  models/feature_stats.json, models/student_<tag>.keras (logits), models/student_<tag>_metrics.json,
  logs/<tag>/epochs.csv + config.json
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from soundwatch import paths
from soundwatch.config import Config, load_config
from soundwatch.models import common

try:  # keras is heavy; only needed when the training code actually runs
    import keras

    _CallbackBase = keras.callbacks.Callback
    _PyDatasetBase = keras.utils.PyDataset
except Exception:  # pragma: no cover
    _CallbackBase = object  # type: ignore[assignment,misc]
    _PyDatasetBase = object  # type: ignore[assignment,misc]


def add_arguments(parser: argparse.ArgumentParser) -> None:
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--distill", dest="distill", action="store_true", default=True,
                   help="knowledge distillation from cached teacher logits (default)")
    g.add_argument("--no-distill", dest="distill", action="store_false")
    parser.add_argument("--tag", default=None, help="run name (default: distill / nodistill)")
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--block-widths", default=None,
                        help="comma list, one per block (overrides --width/--blocks), e.g. 64,128,128,128")
    parser.add_argument("--block-strides", default=None, help="comma list, e.g. 1,2,1,1")
    parser.add_argument("--epochs", type=int, default=None, help="default 60 (1 with --quick)")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--aug-copies", type=int, default=2, help="offline augmented copies of train")
    parser.add_argument("--low-snr-copies", type=int, default=0,
                        help="extra offline copies with noise always mixed at SNR 0-10 dB")
    parser.add_argument("--p-specaug", type=float, default=0.8)
    parser.add_argument("--max-per-class", type=int, default=3000,
                        help="per-epoch cap of windows per class (random subset each epoch; 0 = no cap)")
    parser.add_argument("--max-background", type=int, default=None,
                        help="per-epoch cap for the background class (default: --max-per-class)")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--max-class-weight", type=float, default=4.0)
    parser.add_argument("--cw-power", type=float, default=0.5,
                        help="class weight = inverse frequency ** power (0 = off, 1 = full)")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--quick", action="store_true",
                        help="smoke test: <=2000 train windows, 1 epoch, no caches, outputs tagged *_quick")


# ---------------------------------------------------------------------------------------------------
class MacroF1Monitor(_CallbackBase):
    """Early stopping on validation macro-F1 (keeps best weights in memory) + per-epoch CSV log."""

    def __init__(self, x_val: np.ndarray, y_val: np.ndarray, num_classes: int, patience: int = 10,
                 csv_path: Path | None = None, verbose: bool = True):  # fmt: skip
        super().__init__()
        self.x_val, self.y_val, self.num_classes = x_val, y_val, num_classes
        self.patience = patience
        self.csv_path = csv_path
        self.verbose = verbose
        self.best_f1 = -1.0
        self.best_epoch = -1
        self.best_weights: list | None = None
        self.epochs_run = 0
        self.history: list[dict] = []
        self._t0 = time.time()

    def on_epoch_begin(self, epoch, logs=None):
        self._t0 = time.time()

    def on_epoch_end(self, epoch, logs=None):
        logits = self.model.predict(self.x_val, batch_size=512, verbose=0)
        pred = np.argmax(logits, axis=1)
        f1 = common.macro_f1(self.y_val, pred, self.num_classes)
        acc = float(np.mean(pred == self.y_val))
        lr = self.model.optimizer.learning_rate
        row = {"epoch": epoch + 1, "loss": float((logs or {}).get("loss", np.nan)),
               "val_macro_f1": round(f1, 5), "val_acc": round(acc, 5),
               "lr": float(np.asarray(lr)), "epoch_s": round(time.time() - self._t0, 1)}
        self.history.append(row)
        self.epochs_run = epoch + 1
        if self.csv_path is not None:
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            new = not self.csv_path.exists() or epoch == 0
            with open(self.csv_path, "w" if new else "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(row))
                if new:
                    w.writeheader()
                w.writerow(row)
        improved = f1 > self.best_f1
        if improved:
            self.best_f1, self.best_epoch = f1, epoch + 1
            self.best_weights = [np.array(v) for v in self.model.get_weights()]
        if self.verbose:
            print(f"  epoch {epoch + 1:3d} loss={row['loss']:.4f} val_f1={f1:.4f} val_acc={acc:.4f} "
                  f"({row['epoch_s']}s){' *' if improved else ''}", flush=True)
        if epoch + 1 - self.best_epoch >= self.patience:
            self.model.stop_training = True

    def restore_best(self, model) -> None:
        if self.best_weights is not None:
            model.set_weights(self.best_weights)


class StudentSequence(_PyDatasetBase):
    """Batches of (normalized + SpecAugmented log-mel (B,T,F,1), packed KD targets, sample weights)."""

    def __init__(self, versions: list[np.ndarray], targets: np.ndarray, weights: np.ndarray, stats: dict,
                 batch_size: int, seed: int, p_specaug: float = 0.8, labels: np.ndarray | None = None,
                 max_per_class: int = 0, caps: dict[int, int] | None = None):  # fmt: skip
        super().__init__(workers=1, use_multiprocessing=False)
        self.versions, self.targets, self.weights = versions, targets, weights
        self.stats, self.batch_size, self.seed, self.p_specaug = stats, batch_size, seed, p_specaug
        self.labels, self.max_per_class, self.caps = labels, max_per_class, caps or {}
        self.epoch = 0
        self._shuffle()

    def _shuffle(self) -> None:
        """New epoch: (optionally class-capped) random subset in random order, random version each."""
        rng = np.random.default_rng(self.seed + 7919 * self.epoch)
        n_all = len(self.targets)
        if self.labels is not None and self.max_per_class > 0:
            parts = []
            for c in np.unique(self.labels):
                idx = np.nonzero(self.labels == c)[0]
                cap = self.caps.get(int(c), self.max_per_class)
                parts.append(rng.choice(idx, cap, replace=False) if 0 < cap < len(idx) else idx)
            pool = np.concatenate(parts)
        else:
            pool = np.arange(n_all)
        self.order = rng.permutation(pool)
        self.n = len(self.order)
        self.version = np.zeros(n_all, np.int64)
        self.version[self.order] = rng.integers(0, len(self.versions), size=self.n)

    def __len__(self) -> int:
        return int(np.ceil(self.n / self.batch_size))

    def __getitem__(self, idx: int):
        from soundwatch.audio.features import normalize
        from soundwatch.data.augment import spec_augment

        sel = self.order[idx * self.batch_size:(idx + 1) * self.batch_size]
        ver = self.version[sel]
        x = np.empty((len(sel), *self.versions[0].shape[1:]), np.float32)
        for v in np.unique(ver):
            m = ver == v
            x[m] = self.versions[v][sel[m]]
        x = normalize(x, self.stats)
        rng = np.random.default_rng((self.seed, self.epoch, idx))
        for i in np.nonzero(rng.random(len(sel)) < self.p_specaug)[0]:
            x[i] = spec_augment(x[i], rng, fill=0.0)
        return x[..., None], self.targets[sel], self.weights[sel]

    def on_epoch_end(self) -> None:
        self.epoch += 1
        self._shuffle()


# ---------------------------------------------------------------------------------------------------
def augmented_copies(waves: np.ndarray, cfg: Config, k: int, seed: int, cache: bool = True,
                     snr_db: tuple[float, float] = (0.0, 20.0), p_noise: float = 0.8,
                     name: str = "aug") -> list[np.ndarray]:  # fmt: skip
    """K augmented log-mel copies of the train windows (WP1 augment_wave + WP2 features), cached as
    cache/logmel_train_<name><j>.npy. Background windows are augmented too (background + noise)."""
    from soundwatch.data import augment

    out: list[np.ndarray] = []
    bank = None
    for j in range(k):
        p = paths.CACHE / f"logmel_train_{name}{j}.npy"
        if cache and common.cache_is_valid(p, "train", len(waves)):
            out.append(np.load(p))
            continue
        if bank is None:
            try:
                bank = augment.NoiseBank(paths.NOISE / "train", cfg.audio.sample_rate, preload=True)
                _ = bank.files
            except FileNotFoundError as e:
                print(f"[train] WARNING: {e}; augmenting without noise")
                bank = False
        t0 = time.time()
        rng = np.random.default_rng(seed * 1000 + j + (0 if name == "aug" else 500))
        x = np.empty((len(waves), cfg.audio.n_frames, cfg.audio.n_mels), np.float16)
        chunk = 1024
        for i in range(0, len(waves), chunk):
            w = np.stack([augment.augment_wave(np.asarray(v, np.float32), rng, bank or None,
                                               p_noise=p_noise, p_gain=1.0, p_shift=0.5,
                                               snr_db_range=snr_db,
                                               sample_rate=cfg.audio.sample_rate)
                          for v in waves[i:i + chunk]])
            x[i:i + chunk] = common.logmel_batch(w, cfg.audio).astype(np.float16)
        if cache:
            p.parent.mkdir(parents=True, exist_ok=True)
            np.save(p, x)
            common.mark_cache(p, "train", len(x), seed=seed, copy=j, snr_db=list(snr_db), p_noise=p_noise)
        print(f"[train] augmented copy {j}: {x.shape} in {time.time() - t0:.0f}s", flush=True)
        out.append(x)
    return out


def _subsample(d: dict, n: int, seed: int) -> dict:
    if len(d["label"]) <= n:
        return d
    idx = np.sort(np.random.default_rng(seed).choice(len(d["label"]), n, replace=False))
    return {k: v[idx] for k, v in d.items()}


def _int_list(s: str | None) -> list[int] | None:
    return [int(v) for v in s.split(",")] if s else None


def run(args: argparse.Namespace) -> dict:
    from soundwatch.audio.features import compute_feature_stats, normalize, save_stats
    from soundwatch.models.distill import kd_loss, pack_targets
    from soundwatch.models.dscnn import build_dscnn, count_macs, count_params

    cfg = load_config(args.config)
    quick = bool(args.quick)
    tag = args.tag or ("distill" if args.distill else "nodistill")
    if quick and not tag.endswith("quick"):
        tag += "_quick"
    epochs = args.epochs or (1 if quick else 60)
    common.setup_tf(args.threads, seed=args.seed)
    t_start = time.time()

    arch = {"block_widths": _int_list(args.block_widths), "block_strides": _int_list(args.block_strides)}
    data = {s: common.load_split(s) for s in common.SPLITS}
    teacher = None
    if args.distill:
        tp = paths.CACHE / "teacher_logits_train.npy"
        if not common.cache_is_valid(tp, "train", len(data["train"]["label"])):
            raise FileNotFoundError(f"{tp} missing or stale: run `train-teacher` first or use --no-distill")
        teacher = np.load(tp).astype(np.float32)
        if len(teacher) != len(data["train"]["label"]):
            raise ValueError(f"teacher logits ({len(teacher)}) do not match train windows")
        data["train"]["teacher"] = teacher
    if quick:
        data = {s: _subsample(d, 2000 if s == "train" else 1000, args.seed) for s, d in data.items()}
        teacher = data["train"].get("teacher")

    feats = {s: (common.logmel_batch(d["wave"], cfg.audio) if quick
                 else common.cached_logmel(s, cfg.audio, d["wave"])) for s, d in data.items()}
    stats = compute_feature_stats(feats["train"], cfg.audio)
    stats_path = paths.MODELS / ("feature_stats_quick.json" if quick else "feature_stats.json")
    save_stats(stats, stats_path)
    aug = augmented_copies(data["train"]["wave"], cfg, 1 if quick else args.aug_copies, args.seed,
                           cache=not quick)
    if args.low_snr_copies and not quick:
        aug += augmented_copies(data["train"]["wave"], cfg, args.low_snr_copies, args.seed, cache=True,
                                snr_db=(0.0, 10.0), p_noise=1.0, name="lowsnr")
    t_feat = time.time() - t_start
    print(f"[train] features ready in {t_feat:.0f}s; train {feats['train'].shape} x{1 + len(aug)} versions; "
          f"stats mean={stats['mean']:.3f} std={stats['std']:.3f}", flush=True)

    y_tr = data["train"]["label"]
    C = cfg.num_classes
    targets = pack_targets(y_tr, C, teacher)
    # class weights are computed on the per-epoch (capped) class distribution
    caps = {}
    if args.max_background is not None:
        caps[cfg.background_id] = args.max_background
    cap_arr = np.full(C, args.max_per_class or len(y_tr))
    for c, v in caps.items():
        cap_arr[c] = v or len(y_tr)
    capped = np.minimum(np.bincount(y_tr, minlength=C), cap_arr)
    cw = common.class_weights(np.repeat(np.arange(C), capped), C, clip=(0.25, args.max_class_weight),
                              power=args.cw_power)
    seq = StudentSequence([feats["train"], *aug], targets, cw[y_tr], stats, args.batch_size, args.seed,
                          args.p_specaug, labels=y_tr, max_per_class=args.max_per_class, caps=caps)
    x_val = normalize(feats["val"], stats)[..., None]
    x_test = normalize(feats["test"], stats)[..., None]

    model = build_dscnn((cfg.audio.n_frames, cfg.audio.n_mels, 1), C, args.width, args.blocks, args.dropout,
                        **arch)
    sched = keras.optimizers.schedules.CosineDecay(args.lr, decay_steps=len(seq) * epochs, alpha=0.01)
    alpha = args.alpha if args.distill else 1.0
    model.compile(optimizer=keras.optimizers.Adam(sched), loss=kd_loss(C, alpha, args.temperature,
                                                                       args.label_smoothing))
    log_dir = paths.LOGS / tag
    run_cfg = {k: v for k, v in vars(args).items() if k != "config"} | {
        "tag": tag, "epochs": epochs, "alpha_effective": alpha, "class_weights": cw.round(3).tolist(),
        "n_train": int(len(y_tr)), "n_versions": 1 + len(aug)}
    common.write_json(run_cfg, log_dir / "config.json")
    mon = MacroF1Monitor(x_val, data["val"]["label"], C, args.patience, log_dir / "epochs.csv")
    print(f"[train] {tag}: {model.name} params={model.count_params()} macs={count_macs(model)} "
          f"epochs<={epochs} steps/epoch={len(seq)} alpha={alpha}", flush=True)
    t0 = time.time()
    model.fit(seq, epochs=epochs, callbacks=[mon], verbose=0)
    t_train = time.time() - t0
    mon.restore_best(model)

    # fresh, uncompiled copy (the compiled one references the custom KD loss)
    clean = build_dscnn((cfg.audio.n_frames, cfg.audio.n_mels, 1), C, args.width, args.blocks, args.dropout,
                        **arch)
    clean.set_weights(model.get_weights())
    paths.MODELS.mkdir(parents=True, exist_ok=True)
    model_path = paths.MODELS / f"student_{tag}.keras"
    clean.save(model_path)

    metrics: dict = {
        "tag": tag, "distill": bool(args.distill), "model_path": str(model_path.relative_to(paths.ROOT))
        if model_path.is_relative_to(paths.ROOT) else str(model_path),
        "width": args.width, "blocks": args.blocks, "arch": arch,
        "params": count_params(model), "macs": count_macs(model),
        "best_epoch": mon.best_epoch, "epochs_run": mon.epochs_run, "best_val_macro_f1": mon.best_f1,
        "feature_time_s": round(t_feat, 1), "train_time_s": round(t_train, 1),
        "stats": {"mean": stats["mean"], "std": stats["std"]},
    }
    for s, x in (("val", x_val), ("test", x_test)):
        pred = model.predict(x, batch_size=512, verbose=0).argmax(1)
        metrics[s] = common.classification_metrics(data[s]["label"], pred, cfg)
        print(common.format_metrics_row(f"student {tag} {s}", metrics[s], cfg), flush=True)
    metrics["history"] = mon.history
    common.write_json(metrics, paths.MODELS / f"student_{tag}_metrics.json")
    print(f"[train] saved {model_path} ({t_train:.0f}s training)")
    return metrics


def main(args: argparse.Namespace) -> int:
    run(args)
    return 0
