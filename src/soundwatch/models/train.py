"""WP3 student training: DS-CNN on normalized log-mel, optional knowledge distillation from YAMNet.

Data flow
  windows/{train,val,test}.npz --(WP2 compute_logmel_batch)--> cache/logmel_{split}.npy (clean)
  train waves --(WP1 augment_wave: shift, noise SNR 0-20 dB, gain; K copies, offline once)-->
      cache/logmel_train_aug{k}.npy (float16)
  per epoch: every train window appears once, as a random version (clean or one of the K copies),
      normalized with feature_stats.json, SpecAugment applied online (cheap, numpy).

Outputs (tag defaults to "distill" / "nodistill")
  models/feature_stats_<tag>.json (+ models/feature_stats.json for the default tags distill / nodistill only,
  see writes_shared_stats), models/student_<tag>.keras
  (logits), models/student_<tag>_metrics.json, logs/<tag>/epochs.csv + config.json

vn-v2 options (all off by default = the historical recipe):
  --frontend pcen   features = PCEN (audio.features) computed online from the cached log-mel versions;
                    stats and the model front-end metadata go to feature_stats.json `params`.
  --output sigmoid  multi-label head: BCE over the C outputs (background = "no alert sound"); with
                    --distill the target of un-mixed samples is alpha * hard + (1 - alpha) * teacher
                    softmax (T = 1), mixed samples use the hard union only.
  --p-mix P         mix augmentation (sigmoid only): with probability P a second window is added in
                    the mel-power domain (another event class at a relative level of +-mix_event_db,
                    or background / an event over background at 0..20 dB SNR); targets = union,
                    background = 1 only if no event is present.
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
    parser.add_argument("--width", type=int, default=96, help="DS-CNN stem width")
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--arch", choices=["dscnn", "mbv2"], default="dscnn",
                        help="dscnn (DS-CNN, optionally pyramid) or mbv2 (models/mbnet.py)")
    parser.add_argument("--width-mult", type=float, default=1.0, help="mbv2 width multiplier")
    parser.add_argument("--mixup-alpha", type=float, default=0.0, help="mixup Beta(a, a); 0 = off")
    parser.add_argument("--p-mixup", type=float, default=0.5, help="fraction of batches mixed")
    parser.add_argument("--block-widths", default="96,192,192,192,192",
                        help="comma list, one per block (overrides --width/--blocks), e.g. 64,128,128,128")
    parser.add_argument("--block-strides", default="1,2,1,1,1",
                        help="comma list; --block-widths '' --block-strides '' = plain plan DS-CNN")
    parser.add_argument("--epochs", type=int, default=None, help="default 45 (1 with --quick)")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--aug-copies", type=int, default=2, help="offline augmented copies of train")
    parser.add_argument("--low-snr-copies", type=int, default=0,
                        help="extra offline copies with noise always mixed at SNR 0-10 dB")
    parser.add_argument("--p-specaug", type=float, default=0.8)
    parser.add_argument("--max-per-class", type=int, default=3000,
                        help="per-epoch cap of windows per class (random subset each epoch; 0 = no cap)")
    parser.add_argument("--max-background", type=int, default=6000,
                        help="per-epoch cap for the background class (default: --max-per-class)")
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--max-class-weight", type=float, default=4.0)
    parser.add_argument("--cw-power", type=float, default=0.0,
                        help="class weight = inverse frequency ** power (0 = off, 1 = full)")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--quick", action="store_true",
                        help="smoke test: <=2000 train windows, 1 epoch, no caches, outputs tagged *_quick")
    common.add_data_dir_args(parser)
    parser.add_argument("--frontend", choices=["logmel", "pcen"], default="logmel",
                        help="feature front-end (pcen: fixed PCEN, see audio/features.py)")
    parser.add_argument("--output", choices=["softmax", "sigmoid"], default="softmax",
                        help="output activation: softmax (CE/KD) or sigmoid (multi-label BCE)")
    parser.add_argument("--p-mix", type=float, default=0.0,
                        help="sigmoid only: probability of mixing a second window into a sample")
    parser.add_argument("--mix-event-db", type=float, default=6.0,
                        help="event+event mixes: relative level uniform in +-this many dB")
    parser.add_argument("--mix-p-background", type=float, default=0.3,
                        help="event samples: probability that the mix partner is background")


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
                 max_per_class: int = 0, caps: dict[int, int] | None = None, mixup_alpha: float = 0.0,
                 p_mixup: float = 0.5, meta=None, audio_cfg=None,
                 multilabel: dict | None = None):  # fmt: skip
        """`meta` (audio.features.ModelMeta): front-end applied to the cached log-mel versions.
        `multilabel` (sigmoid head): {"background_id", "alpha", "label_smoothing", "teacher_probs"
        (N, C) or None, "p_mix", "mix_event_db", "mix_p_background"}; `targets` is then unused and
        the (B, C) BCE targets are built per batch (see module doc)."""
        super().__init__(workers=1, use_multiprocessing=False)
        self.versions, self.targets, self.weights = versions, targets, weights
        self.stats, self.batch_size, self.seed, self.p_specaug = stats, batch_size, seed, p_specaug
        self.labels, self.max_per_class, self.caps = labels, max_per_class, caps or {}
        self.mixup_alpha, self.p_mixup = mixup_alpha, p_mixup
        self.meta, self.audio_cfg, self.multilabel = meta, audio_cfg, multilabel
        if multilabel is not None and labels is None:
            raise ValueError("multilabel training needs labels")
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

    def _gather(self, sel: np.ndarray, ver: np.ndarray) -> np.ndarray:
        x = np.empty((len(sel), *self.versions[0].shape[1:]), np.float32)
        for v in np.unique(ver):
            m = ver == v
            x[m] = self.versions[v][sel[m]]
        return x

    def _multilabel_batch(self, sel: np.ndarray, x: np.ndarray, rng: np.random.Generator):
        """Mix augmentation on log-mel x (in place) + (B, C) BCE targets."""
        ml = self.multilabel
        bg = int(ml["background_id"])
        y = self.labels[sel]
        n_cls = int(ml["num_classes"])
        hard = np.zeros((len(sel), n_cls), np.float32)
        hard[np.arange(len(sel)), y] = 1.0
        mixed = np.zeros(len(sel), bool)
        p_mix = float(ml.get("p_mix", 0.0))
        if p_mix > 0:
            pool = self.order
            pool_lab = self.labels[pool]
            bg_pool = pool[pool_lab == bg]
            ev_pool = pool[pool_lab != bg]
            for i in np.nonzero(rng.random(len(sel)) < p_mix)[0]:
                yi = int(y[i])
                if yi == bg:
                    if not len(ev_pool):
                        continue
                    j = int(ev_pool[rng.integers(len(ev_pool))])
                    snr = rng.uniform(0.0, 20.0)  # event over this background
                    sign = +1.0
                elif len(bg_pool) and rng.random() < ml.get("mix_p_background", 0.3):
                    j = int(bg_pool[rng.integers(len(bg_pool))])
                    snr = rng.uniform(0.0, 20.0)  # this event over the background partner
                    sign = -1.0
                else:
                    other = ev_pool[self.labels[ev_pool] != yi]
                    if not len(other):
                        continue
                    j = int(other[rng.integers(len(other))])
                    e = float(ml.get("mix_event_db", 6.0))
                    snr = rng.uniform(-e, e)  # partner level relative to this event
                    sign = +1.0
                xj = self.versions[int(rng.integers(len(self.versions)))][j].astype(np.float32)
                # mel power energies (log-sum-exp); log g sets 10*log10(g*Ej / Ei) = sign * snr
                li = float(np.logaddexp.reduce(x[i].ravel()))
                lj = float(np.logaddexp.reduce(xj.ravel()))
                log_g = li - lj + sign * snr * np.log(10.0) / 10.0
                x[i] = np.logaddexp(x[i], xj + np.float32(log_g))
                yj = int(self.labels[j])
                hard[i, yj] = 1.0
                mixed[i] = True
        has_event = hard[:, :bg].max(axis=1) > 0 if bg > 0 else np.zeros(len(sel), bool)
        hard[:, bg] = np.where(has_event, 0.0, 1.0)
        ls = float(ml.get("label_smoothing", 0.0))
        t = hard * (1.0 - ls) + 0.5 * ls
        tp = ml.get("teacher_probs")
        a = float(ml.get("alpha", 1.0))
        if tp is not None and a < 1.0:
            um = ~mixed
            t[um] = a * t[um] + (1.0 - a) * tp[sel[um]]
        return t.astype(np.float32)

    def __getitem__(self, idx: int):
        from soundwatch.audio.features import logmel_to_features, normalize
        from soundwatch.data.augment import spec_augment

        sel = self.order[idx * self.batch_size:(idx + 1) * self.batch_size]
        ver = self.version[sel]
        x = self._gather(sel, ver)
        rng = np.random.default_rng((self.seed, self.epoch, idx))
        if self.multilabel is not None:
            t_ml = self._multilabel_batch(sel, x, rng)
        if self.meta is not None and self.meta.frontend != "logmel":
            x = logmel_to_features(x, self.meta, self.audio_cfg)
        x = normalize(x, self.stats)
        for i in np.nonzero(rng.random(len(sel)) < self.p_specaug)[0]:
            x[i] = spec_augment(x[i], rng, fill=0.0)
        t, w = (t_ml if self.multilabel is not None else self.targets[sel]), self.weights[sel]
        if self.mixup_alpha > 0 and len(sel) > 1 and rng.random() < self.p_mixup:
            # mixup: convex combination of examples, one-hot targets and teacher logits alike
            lam = rng.beta(self.mixup_alpha, self.mixup_alpha, size=len(sel)).astype(np.float32)
            perm = rng.permutation(len(sel))
            x = lam[:, None, None] * x + (1 - lam[:, None, None]) * x[perm]
            t = lam[:, None] * t + (1 - lam[:, None]) * t[perm]
            w = lam * w + (1 - lam) * w[perm]
        return x[..., None], t, w

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


DEFAULT_TAGS = ("distill", "nodistill")


def writes_shared_stats(tag: str, quick: bool) -> bool:
    """Whether a run also writes the shared models/feature_stats.json (the stats file of
    models/student_int8.tflite). Only the default recipe runs (`distill` / `nodistill`, the Makefile) and
    quick smoke tests do; experiment runs (`--tag <name>`) write feature_stats_<tag>.json only, so an
    experiment can no longer desync the shared stats from the deployed artifacts/models/student_int8.tflite
    (vn-v3; `quantize --tag <name>` reads the per-run file)."""
    return quick or tag in DEFAULT_TAGS


def _subsample(d: dict, n: int, seed: int) -> dict:
    if len(d["label"]) <= n:
        return d
    idx = np.sort(np.random.default_rng(seed).choice(len(d["label"]), n, replace=False))
    return {k: v[idx] for k, v in d.items()}


def build_student(args: argparse.Namespace, cfg: Config, arch: dict):
    from soundwatch.models.dscnn import build_dscnn
    from soundwatch.models.mbnet import build_mbv2

    shape = (cfg.audio.n_frames, cfg.audio.n_mels, 1)
    if getattr(args, "arch", "dscnn") == "mbv2":
        return build_mbv2(shape, cfg.num_classes, args.width_mult, dropout=args.dropout)
    return build_dscnn(shape, cfg.num_classes, args.width, args.blocks, args.dropout, **arch)


def _int_list(s: str | None) -> list[int] | None:
    return [int(v) for v in s.split(",")] if s else None


def run(args: argparse.Namespace) -> dict:
    from soundwatch.audio.features import (
        ModelMeta,
        compute_feature_stats,
        logmel_to_features,
        normalize,
        save_stats,
    )
    from soundwatch.models.distill import bce_loss, kd_loss, pack_targets
    from soundwatch.models.dscnn import count_macs, count_params

    common.apply_data_dirs(args)
    cfg = load_config(args.config)
    quick = bool(args.quick)
    tag = args.tag or ("distill" if args.distill else "nodistill")
    if quick and not tag.endswith("quick"):
        tag += "_quick"
    output = getattr(args, "output", "softmax")
    frontend = getattr(args, "frontend", "logmel")
    default_meta = output == "softmax" and frontend == "logmel" and abs(cfg.audio.window_s - 1.0) < 1e-9
    meta = None if default_meta else ModelMeta.make(frontend, output, cfg.audio.window_s)
    if getattr(args, "p_mix", 0.0) > 0 and output != "sigmoid":
        raise ValueError("--p-mix needs --output sigmoid (mixed samples have several labels)")
    epochs = args.epochs or (1 if quick else 45)
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
    lm_train = feats["train"]  # the sequence mixes / transforms log-mel versions itself
    if meta is not None and meta.frontend != "logmel":
        # model inputs = front-end of the cached log-mel (in chunks to bound memory)
        feats = {s: np.concatenate([logmel_to_features(x[i:i + 4096], meta, cfg.audio)
                                    for i in range(0, len(x), 4096)]) if len(x) else x
                 for s, x in feats.items()}
    stats = compute_feature_stats(feats["train"], cfg.audio, meta)
    if writes_shared_stats(tag, quick):
        save_stats(stats, paths.MODELS / ("feature_stats_quick.json" if quick else "feature_stats.json"))
    save_stats(stats, paths.MODELS / f"feature_stats_{tag}.json")
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
    alpha = args.alpha if args.distill else 1.0
    multilabel = None
    if output == "sigmoid":
        tprobs = None
        if teacher is not None:
            z = teacher - teacher.max(axis=1, keepdims=True)
            tprobs = (np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)).astype(np.float32)
        multilabel = {"num_classes": C, "background_id": cfg.background_id, "alpha": alpha,
                      "label_smoothing": args.label_smoothing, "teacher_probs": tprobs,
                      "p_mix": args.p_mix, "mix_event_db": args.mix_event_db,
                      "mix_p_background": args.mix_p_background}
    seq = StudentSequence([lm_train, *aug], targets, cw[y_tr], stats, args.batch_size, args.seed,
                          args.p_specaug, labels=y_tr, max_per_class=args.max_per_class, caps=caps,
                          mixup_alpha=args.mixup_alpha, p_mixup=args.p_mixup, meta=meta,
                          audio_cfg=cfg.audio, multilabel=multilabel)
    x_val = normalize(feats["val"], stats)[..., None]
    x_test = normalize(feats["test"], stats)[..., None]

    model = build_student(args, cfg, arch)
    sched = keras.optimizers.schedules.CosineDecay(args.lr, decay_steps=len(seq) * epochs, alpha=0.01)
    loss = bce_loss() if output == "sigmoid" else kd_loss(C, alpha, args.temperature, args.label_smoothing)
    model.compile(optimizer=keras.optimizers.Adam(sched), loss=loss)
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
    clean = build_student(args, cfg, arch)
    clean.set_weights(model.get_weights())
    paths.MODELS.mkdir(parents=True, exist_ok=True)
    model_path = paths.MODELS / f"student_{tag}.keras"
    clean.save(model_path)

    metrics: dict = {
        "tag": tag, "distill": bool(args.distill), "model_path": str(model_path.relative_to(paths.ROOT))
        if model_path.is_relative_to(paths.ROOT) else str(model_path),
        "width": args.width, "blocks": args.blocks,
        "arch": {"type": args.arch, "width_mult": args.width_mult} | arch,
        "params": count_params(model), "macs": count_macs(model),
        "best_epoch": mon.best_epoch, "epochs_run": mon.epochs_run, "best_val_macro_f1": mon.best_f1,
        "feature_time_s": round(t_feat, 1), "train_time_s": round(t_train, 1),
        "stats": {"mean": stats["mean"], "std": stats["std"]},
        "frontend": frontend, "output": output, "p_mix": float(getattr(args, "p_mix", 0.0)),
        "stats_path": str(paths.MODELS / f"feature_stats_{tag}.json"),
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
