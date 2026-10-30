"""WP1: build 1 s training windows, fold-based splits, noise bank and event clips.

Outputs (see docs/IMPLEMENTATION_PLAN.md WP1):
- ``paths.WINDOWS/{train,val,test}.npz``: ``wave (N,16000) float16``, ``label (N,) int16``,
  ``source_clip (N,) <U`` (e.g. ``esc50/1-100032-A-0.wav``, ``us8k/fold1/7061-6-0-0.wav``,
  ``fsd50k/dev/12345.wav`` with ``--with-fsd50k``)
- ``paths.WINDOWS/summary.json``: window counts per class per split (+ clip counts, drops)
- ``paths.NOISE/{train,test}/*.wav``: full background clips, 16 kHz mono PCM16
- ``paths.DATA/events/{val,test}/<class_key>/*.wav``: full alert-class clips, 16 kHz mono PCM16
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import shutil
from dataclasses import dataclass
from math import gcd
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly

from soundwatch import paths
from soundwatch.config import Config, load_config
from soundwatch.data import download, fsd50k, label_map

SPLITS = ("train", "val", "test")
ESC50_FOLDS = {1: "train", 2: "train", 3: "train", 4: "val", 5: "test"}
FSD50K_SPLITS = {"train": "train", "val": "val", "eval": "test"}  # FSD50K dev train/val, eval
CLOCK_ALARM_MIN_FSD_TRAIN = 60  # `--clock-alarm auto`: background once FSD50K has this many smoke clips
US8K_FOLDS = {**{f: "train" for f in range(1, 9)}, 9: "val", 10: "test"}
SILENCE_DB = 20.0
BG_CAP_FACTOR = 3.0
EVENTS = paths.DATA / "events"


@dataclass(frozen=True)
class ClipRecord:
    source_clip: str  # "<dataset>/<relative path>", unique id
    path: str  # absolute path of the source audio
    label: int
    split: str
    max_windows: int | None = None  # cap (evenly spaced) after silent-window dropping
    active_s: tuple[float, ...] | None = None  # FSD50K: start times of YAMNet patches with the event


# ---------------------------------------------------------------- collection


def collect_esc50(root: Path, cfg: Config, clock_alarm_background: bool = False) -> list[ClipRecord]:
    df = pd.read_csv(root / "meta" / "esc50.csv")
    out = []
    for r in df.itertuples(index=False):
        lab = label_map.map_esc50(r.category, cfg, clock_alarm_background)
        if lab is None:
            continue
        out.append(
            ClipRecord(f"esc50/{r.filename}", str(root / "audio" / r.filename), lab, ESC50_FOLDS[int(r.fold)])
        )
    return out


def collect_us8k(root: Path, cfg: Config) -> list[ClipRecord]:
    df = pd.read_csv(root / "metadata" / "UrbanSound8K.csv")
    out = []
    for r in df.to_dict("records"):
        lab = label_map.map_us8k(r["class"], cfg)
        if lab is None:
            continue
        rel = f"fold{int(r['fold'])}/{r['slice_file_name']}"
        out.append(ClipRecord(f"us8k/{rel}", str(root / "audio" / rel), lab, US8K_FOLDS[int(r["fold"])]))
    return out


def collect_fsd50k(root: Path, selection: pd.DataFrame, cfg: Config, max_windows: int) -> list[ClipRecord]:
    """Kept rows of ``raw/fsd50k/selection.csv`` (columns fname, split, fsd_split, target)."""
    out = []
    for r in selection.to_dict("records"):
        rel = f"{r['split']}/{r['fname']}.wav"
        out.append(
            ClipRecord(
                f"fsd50k/{rel}",
                str(root / rel),
                label_map.map_fsd50k(r["target"], cfg),
                FSD50K_SPLITS[r["fsd_split"]],
                max_windows,
                fsd50k.active_starts(r.get("active_patches")) if "active_patches" in r else None,
            )
        )
    return out


# windows kept per custom clip (evenly spaced): VN clips are 5-15 s, so this keeps VN data at roughly a
# quarter to a third of each alert class instead of swamping the original data (docs/vn_siren_finetune.md)
CUSTOM_MAX_WINDOWS = {"background": 4, "car_horn": 4}
CUSTOM_MAX_WINDOWS_DEFAULT = 6


def collect_custom(manifest: Path, cfg: Config, skip_low: bool = True) -> list[ClipRecord]:
    """Clips listed in a custom-data ``manifest.csv`` (e.g. project/vn-sound-data): columns ``file`` (relative
    to the manifest), ``class_key``, ``split`` (already grouped by channel), ``label_confidence``.
    ``source_clip`` is ``vn/<file>`` so the VN part of every split can be evaluated on its own."""
    df = pd.read_csv(manifest)
    out = []
    for r in df.to_dict("records"):
        if r["class_key"] not in cfg.class_keys:
            continue
        if skip_low and r.get("label_confidence") == "low":
            continue
        out.append(
            ClipRecord(
                f"vn/{r['file']}",
                str(manifest.parent / r["file"]),
                label_map.key_to_id(r["class_key"], cfg),
                r["split"],
                CUSTOM_MAX_WINDOWS.get(r["class_key"], CUSTOM_MAX_WINDOWS_DEFAULT),
            )
        )
    return out


# ---------------------------------------------------------------- audio


def load_audio(path: str | Path, sr: int) -> np.ndarray:
    """Load any audio file as mono float32 at ``sr`` Hz, clipped to [-1, 1]."""
    try:
        x, sr0 = sf.read(str(path), dtype="float32", always_2d=True)
        x = x.mean(axis=1)
    except (RuntimeError, sf.LibsndfileError):  # rare exotic encodings: fall back to librosa/audioread
        import librosa

        x, sr0 = librosa.load(str(path), sr=None, mono=True)
    if sr0 != sr:
        g = gcd(int(sr0), int(sr))
        x = resample_poly(x, sr // g, int(sr0) // g)
    return np.clip(np.asarray(x, dtype=np.float32), -1.0, 1.0)


def frame_windows(x: np.ndarray, win: int, hop: int) -> np.ndarray:
    """1 s windows with hop; clips shorter than ``win`` are zero-padded at the end."""
    if len(x) < win:
        x = np.pad(x, (0, win - len(x)))
    n = 1 + (len(x) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    return x[idx]


def window_rms_db(w: np.ndarray) -> np.ndarray:
    rms = np.sqrt(np.mean(np.square(w.astype(np.float64)), axis=1))
    return 20.0 * np.log10(rms + 1e-10)


def voiced_mask(w: np.ndarray, rel_db: float = SILENCE_DB) -> np.ndarray:
    """Windows whose RMS is within ``rel_db`` of the clip's loudest window."""
    db = window_rms_db(w)
    if db.max() <= -190.0:  # digital silence everywhere
        return np.zeros(len(w), bool)
    return db >= db.max() - rel_db


def drop_silent(w: np.ndarray, rel_db: float = SILENCE_DB) -> np.ndarray:
    """Keep windows whose RMS is within ``rel_db`` of the clip's loudest window."""
    return w[voiced_mask(w, rel_db)]


def active_window_mask(
    n: int, win_s: float, hop_s: float, starts: tuple[float, ...], patch_s: float = fsd50k.PATCH_S
) -> np.ndarray:
    """Windows whose centre lies inside one of the active patches [start, start + patch_s]."""
    centres = np.arange(n) * hop_s + win_s / 2
    if not starts:
        return np.zeros(n, bool)
    st = np.asarray(starts)[None, :]
    return ((centres[:, None] >= st) & (centres[:, None] <= st + patch_s)).any(axis=1)


def cap_windows(w: np.ndarray, n: int | None) -> np.ndarray:
    """At most ``n`` windows, evenly spaced over the clip (first and last included)."""
    if n is None or len(w) <= n:
        return w
    return w[np.round(np.linspace(0, len(w) - 1, n)).astype(int)]


@dataclass(frozen=True)
class _Task:
    rec: ClipRecord
    sr: int
    win: int
    hop: int
    is_alert: bool
    save_to: str | None  # write full 16 kHz clip here (noise bank / event clips)


def _process(t: _Task) -> tuple[ClipRecord, np.ndarray, int, int, str | None]:
    try:
        x = load_audio(t.rec.path, t.sr)
    except Exception as e:  # noqa: BLE001 - report and skip unreadable clips
        return t.rec, np.zeros((0, t.win), np.float16), 0, 0, f"{t.rec.source_clip}: {e!r}"
    w = frame_windows(x, t.win, t.hop)
    n_all = len(w)
    keep = voiced_mask(w) if t.is_alert else np.ones(n_all, bool)
    n_voiced = int(keep.sum())
    if t.rec.active_s is not None:  # FSD50K: only windows where YAMNet hears the event
        keep &= active_window_mask(n_all, t.win / t.sr, t.hop / t.sr, t.rec.active_s)
    w = cap_windows(w[keep], t.rec.max_windows)
    if t.save_to:
        Path(t.save_to).parent.mkdir(parents=True, exist_ok=True)
        sf.write(t.save_to, x, t.sr, subtype="PCM_16")
    return t.rec, w.astype(np.float16), n_all - n_voiced, n_voiced - len(w), None


# ---------------------------------------------------------------- build


def _flat_name(source_clip: str) -> str:
    return source_clip.replace("/", "_")


def _save_target(rec: ClipRecord, cfg: Config, noise_dir: Path, events_dir: Path) -> str | None:
    if rec.label == cfg.background_id:
        if rec.split in ("train", "test"):
            return str(noise_dir / rec.split / _flat_name(rec.source_clip))
        return None
    if rec.split in ("val", "test"):
        return str(events_dir / rec.split / cfg.class_keys[rec.label] / _flat_name(rec.source_clip))
    return None


def cap_background(labels: np.ndarray, bg_id: int, factor: float, rng: np.random.Generator) -> np.ndarray:
    """Indices to keep so that #background <= factor * largest non-background class."""
    others = np.bincount(labels[labels != bg_id], minlength=bg_id)
    cap = int(factor * others.max()) if others.size and others.max() > 0 else None
    bg = np.flatnonzero(labels == bg_id)
    if cap is None or len(bg) <= cap:
        return np.arange(len(labels))
    keep_bg = rng.choice(bg, size=cap, replace=False)
    return np.sort(np.concatenate([np.flatnonzero(labels != bg_id), keep_bg]))


def build(
    records: list[ClipRecord],
    cfg: Config,
    out_dir: Path = paths.WINDOWS,
    noise_dir: Path = paths.NOISE,
    events_dir: Path = EVENTS,
    workers: int = 4,
    seed: int = 1234,
    bg_cap: float = BG_CAP_FACTOR,
    verbose: bool = True,
) -> dict:
    ids = [r.source_clip for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate source_clip ids")
    for d in (noise_dir, events_dir):
        if d.exists():
            shutil.rmtree(d)
    out_dir.mkdir(parents=True, exist_ok=True)
    a = cfg.audio
    tasks = [
        _Task(
            r,
            a.sample_rate,
            a.window_samples,
            a.hop_samples,
            label_map.is_alert(r.label, cfg),
            _save_target(r, cfg, noise_dir, events_dir),
        )
        for r in records
    ]

    per_split: dict[str, tuple[list[np.ndarray], list[int], list[str]]] = {s: ([], [], []) for s in SPLITS}
    clips = {s: np.zeros(cfg.num_classes, int) for s in SPLITS}
    silent = {s: np.zeros(cfg.num_classes, int) for s in SPLITS}
    capped = {s: np.zeros(cfg.num_classes, int) for s in SPLITS}
    errors: list[str] = []

    def consume(it) -> None:
        for i, (rec, w, n_drop, n_cap, err) in enumerate(it, 1):
            if err:
                errors.append(err)
            else:
                waves, labs, srcs = per_split[rec.split]
                waves.append(w)
                labs.extend([rec.label] * len(w))
                srcs.extend([rec.source_clip] * len(w))
                clips[rec.split][rec.label] += 1
                silent[rec.split][rec.label] += n_drop
                capped[rec.split][rec.label] += n_cap
            if verbose and i % 1000 == 0:
                print(f"  processed {i}/{len(tasks)} clips", flush=True)

    if workers > 1:
        with mp.get_context("spawn").Pool(workers) as pool:
            consume(pool.imap(_process, tasks, chunksize=16))
    else:
        consume(map(_process, tasks))

    rng = np.random.default_rng(seed)
    summary: dict = {
        "classes": cfg.class_keys,
        "windows": {},
        "windows_before_bg_cap": {},
        "clips": {},
        "silent_windows_dropped": {},
        "windows_filtered": {},  # FSD50K: removed by the YAMNet-activity filter or the per-clip cap
        "errors": errors,
        "seed": seed,
    }
    for s in SPLITS:
        waves, labs, srcs = per_split[s]
        wave = np.concatenate(waves) if waves else np.zeros((0, a.window_samples), np.float16)
        label = np.asarray(labs, dtype=np.int16)
        src = np.asarray(srcs, dtype=str)
        per_split[s] = ([], [], [])  # free lists
        summary["windows_before_bg_cap"][s] = dict(
            zip(cfg.class_keys, np.bincount(label, minlength=cfg.num_classes).tolist(), strict=True)
        )
        keep = cap_background(label, cfg.background_id, bg_cap, rng)
        wave, label, src = wave[keep], label[keep], src[keep]
        np.savez(out_dir / f"{s}.npz", wave=wave, label=label, source_clip=src)
        del wave
        summary["windows"][s] = dict(
            zip(cfg.class_keys, np.bincount(label, minlength=cfg.num_classes).tolist(), strict=True)
        )
        summary["clips"][s] = dict(zip(cfg.class_keys, clips[s].tolist(), strict=True))
        summary["silent_windows_dropped"][s] = dict(zip(cfg.class_keys, silent[s].tolist(), strict=True))
        summary["windows_filtered"][s] = dict(zip(cfg.class_keys, capped[s].tolist(), strict=True))
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    if verbose:
        print(format_table(summary))
        if errors:
            print(f"{len(errors)} unreadable clips skipped, e.g. {errors[:3]}")
    return summary


def format_table(summary: dict) -> str:
    keys = summary["classes"]
    lines = [f"{'class':<15}" + "".join(f"{s:>9}" for s in SPLITS) + f"{'(clips tr/va/te)':>22}"]
    for k in keys:
        row = f"{k:<15}" + "".join(f"{summary['windows'][s][k]:>9}" for s in SPLITS)
        row += f"{'/'.join(str(summary['clips'][s][k]) for s in SPLITS):>22}"
        lines.append(row)
    lines.append(f"{'TOTAL':<15}" + "".join(f"{sum(summary['windows'][s].values()):>9}" for s in SPLITS))
    return "\n".join(lines)


# ---------------------------------------------------------------- CLI


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--quick", action="store_true", help="ESC-50 only")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--with-fsd50k",
        action="store_true",
        help="add the YAMNet-verified FSD50K baby_cry / smoke_alarm clips (run `download --with-fsd50k`)",
    )
    parser.add_argument(
        "--clock-alarm",
        choices=("auto", "smoke_alarm", "background"),
        default="auto",
        help="ESC-50 clock_alarm label: smoke_alarm proxy, background, or auto (background iff FSD50K "
        f"gives >= {CLOCK_ALARM_MIN_FSD_TRAIN} train smoke_alarm clips)",
    )
    parser.add_argument(
        "--fsd50k-max-windows", type=int, default=fsd50k.MAX_WINDOWS, help="window cap per FSD50K clip"
    )
    parser.add_argument(
        "--custom-manifest",
        default=None,
        help="also add clips from a custom manifest.csv (e.g. ../vn-sound-data/manifest.csv); "
        "its split column is used",
    )
    parser.add_argument(
        "--custom-keep-low", action="store_true", help="keep custom clips with label_confidence=low"
    )


def resolve_clock_alarm(mode: str, fsd_records: list[ClipRecord], cfg: Config) -> bool:
    """True -> map ESC-50 clock_alarm to background."""
    if mode != "auto":
        return mode == "background"
    smoke = label_map.key_to_id("smoke_alarm", cfg)
    n = sum(r.label == smoke and r.split == "train" for r in fsd_records)
    return n >= CLOCK_ALARM_MIN_FSD_TRAIN


def main(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    if not download.esc50_ready():
        print("ESC-50 missing: run `python -m soundwatch download` first")
        return 1
    fsd_records: list[ClipRecord] = []
    if args.with_fsd50k:
        try:
            kept = fsd50k.kept_clips()
        except FileNotFoundError as e:
            print(e)
            return 1
        fsd_records = collect_fsd50k(fsd50k.fsd_dir(), kept, cfg, args.fsd50k_max_windows)
    clock_bg = resolve_clock_alarm(args.clock_alarm, fsd_records, cfg)
    records = collect_esc50(download.esc50_dir(), cfg, clock_alarm_background=clock_bg)
    if not args.quick:
        if not download.us8k_ready():
            print("UrbanSound8K missing: run `python -m soundwatch download` (or use --quick)")
            return 1
        records += collect_us8k(download.us8k_dir(), cfg)
    records += fsd_records
    custom: list[ClipRecord] = []
    if args.custom_manifest:
        custom = collect_custom(Path(args.custom_manifest).resolve(), cfg, skip_low=not args.custom_keep_low)
        records += custom
    src = "quick: ESC-50 only" if args.quick else "ESC-50 + US8K"
    if fsd_records:
        src += f" + {len(fsd_records)} FSD50K clips"
    if custom:
        src += f" + {len(custom)} custom clips"
    print(
        f"building dataset from {len(records)} clips ({src}); "
        f"ESC-50 clock_alarm -> {'background' if clock_bg else 'smoke_alarm'}"
    )
    summary = build(records, cfg, workers=args.workers, seed=args.seed)
    summary["quick"] = bool(args.quick)
    summary["with_fsd50k"] = bool(args.with_fsd50k)
    summary["clock_alarm"] = "background" if clock_bg else "smoke_alarm"
    summary["custom_manifest"] = str(Path(args.custom_manifest).resolve()) if args.custom_manifest else None
    summary["custom_clips"] = len(custom)
    with open(paths.WINDOWS / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    missing = [k for k, v in summary["windows"]["test"].items() if v == 0]
    if missing:
        print(f"WARNING: classes with no test samples: {missing}")
        return 1
    return 0
