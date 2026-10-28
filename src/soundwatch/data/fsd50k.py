"""WP1: FSD50K extras for the two weakest classes, ``baby_cry`` and ``smoke_alarm``.

FSD50K (Zenodo record 4060432, CC licences per clip) has no "Baby cry" / "Smoke detector" class in
its 200-class vocabulary, so candidates are found in two steps and then verified with YAMNet:

1. **Candidates** from the ground truth + Freesound metadata (title / tags / description) and the
   finer "collection" labels shipped in ``FSD50K.metadata/collection`` (``Baby_cry_and_infant_cry``,
   ``Fire_alarm``, ``Alarm_clock``, ``Car_alarm`` ...):

   - baby_cry: GT ``Crying_and_sobbing`` and (collection ``Baby_cry_and_infant_cry`` or a
     baby/infant/newborn/toddler term).
   - smoke_alarm, tier ``keyword``: GT ``Alarm`` and (collection ``Fire_alarm`` or a smoke-detector /
     fire-alarm / CO-detector term), no excluded GT label (siren, telephone, horn, ...) and no
     clock/car/watch/phone/timer/... term in the title or tags.
   - smoke_alarm, tier ``acoustic``: the remaining generic ``Alarm`` clips (collection label just
     ``Alarm``, same exclusions). Smoke detectors are rarely *named* on Freesound, but many generic
     "alarm beep" clips are acoustically the same thing; this tier is only kept with a much stricter
     YAMNet threshold (see :data:`THRESHOLDS`).

2. **Selective download**: only candidate members are fetched from the split audio zips with HTTP
   Range requests (:mod:`soundwatch.data.remote_zip`), stored as ``raw/fsd50k/{dev,eval}/<fname>.wav``.

3. **YAMNet verification**: per 0.96 s patch (hop 0.48 s) scores; ``yamnet_score`` = max over patches of
   the target AudioSet classes ("Baby cry, infant cry" / "Smoke detector, smoke alarm" + "Fire alarm"),
   ``yamnet_top5`` = a target class is in some patch's top-5 (informative only), ``active_patches`` =
   patches where the target is present. Clip rule: :func:`keep_rule`. Results go to
   ``raw/fsd50k/selection.csv``; ``kept`` clips feed ``build-dataset --with-fsd50k``, which keeps only
   windows centred in an active patch, then drops silent windows and caps windows per clip.

Splits: FSD50K dev ``train`` -> train, dev ``val`` -> val, ``eval`` -> test.
"""

from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from soundwatch import paths
from soundwatch.data import remote_zip as rz

ZENODO = "https://zenodo.org/records/4060432/files/"
META_ZIPS = ("FSD50K.ground_truth.zip", "FSD50K.metadata.zip")
# split audio archives: (stem, number of disks, known segment sizes from the Zenodo API)
AUDIO = {
    "dev": ("FSD50K.dev_audio", (3221225472,) * 5 + (2306663327,)),
    "eval": ("FSD50K.eval_audio", (3221225472, 3037675767)),
}
TARGETS = ("baby_cry", "smoke_alarm")

BABY_TERMS = re.compile(r"\b(bab(y|ies|e)|infant|new[\s_-]?born|toddler|b[ée]b[ée]|neonat\w*)", re.I)
SMOKE_TERMS = re.compile(
    r"smoke[\s_-]*(detector|alarm)|fire[\s_-]*alarm|carbon[\s_-]*monoxide|\bco[\s_-]*(detector|alarm)"
    r"|smokealarm|smokedetector|firealarm|rauchmelder|feueralarm",
    re.I,
)
SMOKE_EXCLUDE_TERMS = re.compile(
    r"clock|\bcar\b|\bcars\b|vehicle|\bwatch|phone|\btimer|egg|kitchen[\s_-]*timer|microwave|truck|reverse"
    r"|backing|bicycle|\bbike|siren|horn|doorbell|door[\s_-]*bell|ring[\s_-]*tone|game|video|music|synth",
    re.I,
)
SMOKE_EXCLUDE_LABELS = frozenset({
    "Siren", "Telephone", "Ringtone", "Vehicle_horn_and_car_horn_and_honking", "Car", "Vehicle",
    "Motor_vehicle_(road)", "Doorbell", "Bicycle_bell", "Music", "Musical_instrument", "Speech",
})  # fmt: skip
SMOKE_EXCLUDE_COLLECTION = frozenset({"Alarm_clock", "Car_alarm"})

# AudioSet (YAMNet) class names per target
YAMNET_CLASSES = {
    "baby_cry": ("Baby cry, infant cry",),
    "smoke_alarm": ("Smoke detector, smoke alarm", "Fire alarm"),
}
# Clip-level rule: keep if the max-over-patches target score >= THRESHOLDS[(target, tier)] and, for
# smoke_alarm, the target beats YAMNet "Siren" (siren-like fire alarms would teach smoke_alarm == siren).
# Chosen from the observed score distribution (see docs / selection.csv): baby-cry scores are bimodal
# (true cries mostly >= 0.4, mislabelled / speech-dominated clips <= 0.12); named fire/smoke alarms are
# >= 0.35 while rejected keyword clips (sirens, music, bells) are <= 0.16; the acoustic tier needs 0.5.
# "Target in some patch's top-5" alone is NOT sufficient: it admits clips scoring 0.03-0.12.
THRESHOLDS = {
    ("baby_cry", "keyword"): 0.15,
    ("smoke_alarm", "keyword"): 0.30,
    ("smoke_alarm", "acoustic"): 0.50,
}
SIREN_CLASS = "Siren"
# Window-level activity: a YAMNet patch is "active" if its target score >= max(ACTIVE_MIN,
# ACTIVE_REL * clip max). build-dataset keeps only windows whose centre lies in an active patch, so
# long clips (ambiences with a cry / beep somewhere) only contribute the windows containing the event.
ACTIVE_MIN = 0.05
ACTIVE_REL = 0.25
PATCH_S = 0.96
PATCH_HOP_S = 0.48

MAX_WINDOWS = 10  # per FSD50K clip (clips are 0.3-30 s), evenly spaced


def fsd_dir(raw: Path | None = None) -> Path:
    return (raw or paths.RAW) / "fsd50k"


def selection_csv(raw: Path | None = None) -> Path:
    return fsd_dir(raw) / "selection.csv"


# ---------------------------------------------------------------- metadata


def ensure_metadata(d: Path) -> None:
    from soundwatch.data.download import fetch_resumable

    d.mkdir(parents=True, exist_ok=True)
    for name in META_ZIPS:
        z = d / name
        if not z.exists():
            print(f"[fsd50k] downloading {name}")
            fetch_resumable(ZENODO + name, z)
        target = d / name.removesuffix(".zip")
        if not target.exists():
            with zipfile.ZipFile(z) as zf:
                for m in zf.namelist():
                    if m.startswith("/") or ".." in Path(m).parts:
                        raise ValueError(f"unsafe member name {m!r} in {name}")
                zf.extractall(d)


def _labels(s: str) -> list[str]:
    return [x for x in str(s).split(",") if x]


def find_candidates(d: Path) -> pd.DataFrame:
    """One row per candidate clip: fname, split (dev/eval), fsd_split (train/val/eval), target, tier,
    labels, collection, title, tags."""
    rows = []
    for split in ("dev", "eval"):
        gt = pd.read_csv(d / "FSD50K.ground_truth" / f"{split}.csv", dtype={"fname": str})
        col = pd.read_csv(
            d / "FSD50K.metadata" / "collection" / f"collection_{split}.csv", dtype=str, keep_default_na=False
        )
        col_map = dict(zip(col["fname"], col["labels"], strict=True))
        with open(d / "FSD50K.metadata" / f"{split}_clips_info_FSD50K.json", encoding="utf-8") as f:
            info = json.load(f)
        for r in gt.to_dict("records"):
            fname = r["fname"]
            labels = set(_labels(r["labels"]))
            coll = set(_labels(col_map.get(fname, "")))
            meta = info.get(fname, {})
            title = str(meta.get("title", ""))
            tags = " ".join(meta.get("tags", []))
            desc = str(meta.get("description", ""))
            text = f"{title} {tags} {desc}"
            short = f"{title} {tags}"
            target = tier = None
            if "Crying_and_sobbing" in labels and (
                "Baby_cry_and_infant_cry" in coll or BABY_TERMS.search(text)
            ):
                target, tier = "baby_cry", "keyword"
            elif (
                "Alarm" in labels
                and not labels & SMOKE_EXCLUDE_LABELS
                and not coll & SMOKE_EXCLUDE_COLLECTION
                and not SMOKE_EXCLUDE_TERMS.search(short)
            ):
                if "Fire_alarm" in coll or SMOKE_TERMS.search(text):
                    target, tier = "smoke_alarm", "keyword"
                elif coll <= {"Alarm"} and not (labels - {"Alarm"}):
                    target, tier = "smoke_alarm", "acoustic"
            if target:
                rows.append(
                    dict(
                        fname=fname,
                        split=split,
                        fsd_split=r.get("split", "eval") if split == "dev" else "eval",
                        target=target,
                        tier=tier,
                        labels=r["labels"],
                        collection=col_map.get(fname, ""),
                        title=title,
                        tags=tags,
                        license=meta.get("license", ""),
                        uploader=meta.get("uploader", ""),
                    )
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- download


def clip_path(d: Path, split: str, fname: str) -> Path:
    return d / split / f"{fname}.wav"


def download_clips(d: Path, cand: pd.DataFrame) -> int:
    """Fetch candidate members from the split zips. Returns the number of bytes downloaded."""
    total = 0
    for split, (stem, sizes) in AUDIO.items():
        todo = [f for f in cand.loc[cand["split"] == split, "fname"] if not clip_path(d, split, f).exists()]
        if not todo:
            continue
        names = rz.split_segment_names(stem, len(sizes))
        src = rz.HttpSegments([ZENODO + n for n in names], sizes)
        z = rz.SplitZip(src)
        print(f"[fsd50k] {split}: {len(z.entries)} members in archive, fetching {len(todo)} clips")
        (d / split).mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(todo, 1):
            data = z.read(f"{stem}/{f}.wav")
            out = clip_path(d, split, f)
            tmp = out.with_suffix(".part")
            tmp.write_bytes(data)
            tmp.rename(out)
            if i % 50 == 0:
                print(f"  {i}/{len(todo)} ({src.bytes_fetched / 2**20:.0f} MiB)", flush=True)
        total += src.bytes_fetched
    return total


# ---------------------------------------------------------------- YAMNet verification


def patch_frames(x: np.ndarray, patch: int = 15600, hop: int = 7680) -> np.ndarray:
    """Waveform -> (K, patch) YAMNet input patches (0.96 s + STFT margin, hop 0.48 s)."""
    if len(x) < patch:
        x = np.pad(x, (0, patch - len(x)))
    n = 1 + (len(x) - patch) // hop
    idx = np.arange(patch)[None, :] + hop * np.arange(n)[:, None]
    return x[idx]


def yamnet_patch_scores(model, x: np.ndarray, batch: int = 64) -> np.ndarray:
    """(K, 521) per-patch scores of a 16 kHz waveform."""
    p = patch_frames(x)
    return np.concatenate([model(p[i : i + batch])[0] for i in range(0, len(p), batch)])


def score_clip(scores: np.ndarray, target_idx: list[int], names: list[str]) -> tuple[float, bool, str]:
    """(max target score over patches, target in some patch's top-5, clip-level top-3 names)."""
    s = float(scores[:, target_idx].max())
    top5 = np.argsort(-scores, axis=1)[:, :5]
    hit = bool(np.isin(top5, target_idx).any())
    top = np.argsort(-scores.max(axis=0))[:3]
    return s, hit, "; ".join(f"{names[i]}={scores[:, i].max():.2f}" for i in top)


def active_patches(target_scores: np.ndarray) -> list[int]:
    """Indices of patches whose target score is high enough (see ACTIVE_MIN / ACTIVE_REL)."""
    thr = max(ACTIVE_MIN, ACTIVE_REL * float(target_scores.max()))
    return np.flatnonzero(target_scores >= thr).tolist()


def active_starts(cell: object) -> tuple[float, ...]:
    """selection.csv ``active_patches`` cell ("0;1;5") -> patch start times in seconds."""
    if cell is None or (isinstance(cell, float) and np.isnan(cell)) or str(cell) == "":
        return ()
    return tuple(round(int(i) * PATCH_HOP_S, 3) for i in str(cell).split(";"))


def keep_rule(target: str, tier: str, score: float, siren: float = 0.0) -> bool:
    if score < THRESHOLDS[(target, tier)]:
        return False
    return target != "smoke_alarm" or score > siren


def verify(d: Path, cand: pd.DataFrame, sr: int = 16000) -> pd.DataFrame:
    from soundwatch.data.build_dataset import load_audio
    from soundwatch.models.yamnet_loader import Yamnet

    model = Yamnet()
    names = model.names
    idx = {t: [names.index(c) for c in cs] for t, cs in YAMNET_CLASSES.items()}
    siren = names.index(SIREN_CLASS)
    out = []
    for i, r in enumerate(cand.to_dict("records"), 1):
        x = load_audio(clip_path(d, r["split"], r["fname"]), sr)
        sc = yamnet_patch_scores(model, x)
        s, hit, top = score_clip(sc, idx[r["target"]], names)
        act = active_patches(sc[:, idx[r["target"]]].max(axis=1))
        out.append(
            dict(
                r,
                duration_s=round(len(x) / sr, 2),
                yamnet_score=round(s, 4),
                yamnet_top5=hit,
                yamnet_siren=round(float(sc[:, siren].max()), 4),
                n_patches=len(sc),
                active_patches=";".join(map(str, act)),
                yamnet_top=top,
            )
        )
        if i % 100 == 0:
            print(f"  [yamnet] {i}/{len(cand)}", flush=True)
    df = pd.DataFrame(out)
    df["kept"] = apply_rule(df)
    return df


def apply_rule(df: pd.DataFrame) -> list[bool]:
    return [
        keep_rule(t, k, s, sir)
        for t, k, s, sir in zip(df["target"], df["tier"], df["yamnet_score"], df["yamnet_siren"], strict=True)
    ]


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["target", "tier", "fsd_split"])
    return pd.DataFrame({"candidates": g.size(), "kept": g["kept"].sum()})


def run(raw: Path | None = None, rescore: bool = False) -> pd.DataFrame:
    """Metadata -> candidates -> selective download -> YAMNet selection -> selection.csv."""
    d = fsd_dir(raw)
    ensure_metadata(d)
    cand = find_candidates(d)
    print("[fsd50k] candidates:")
    print(cand.groupby(["target", "tier", "fsd_split"]).size().to_string())
    nbytes = download_clips(d, cand)
    print(f"[fsd50k] downloaded {nbytes / 2**20:.1f} MiB this run")
    csv = selection_csv(raw)
    if csv.exists() and not rescore:
        old = pd.read_csv(csv, dtype={"fname": str})
        same = set(zip(old["split"], old["fname"], strict=True)) == set(
            zip(cand["split"], cand["fname"], strict=True)
        )
        if same and {"yamnet_siren", "active_patches"} <= set(old.columns):
            print(f"[fsd50k] {csv}: YAMNet scores up to date (use --fsd50k-rescore to recompute)")
            old["kept"] = apply_rule(old)  # thresholds may have changed
            old.to_csv(csv, index=False)
            print(summarize(old).to_string())
            return old
    sel = verify(d, cand)
    sel.to_csv(csv, index=False)
    print(summarize(sel).to_string())
    return sel


def kept_clips(raw: Path | None = None) -> pd.DataFrame:
    """Rows of selection.csv with kept == True (raises if the selection was never run)."""
    csv = selection_csv(raw)
    if not csv.exists():
        raise FileNotFoundError(f"{csv} missing: run `python -m soundwatch download --with-fsd50k`")
    df = pd.read_csv(csv, dtype={"fname": str})
    return df[df["kept"].astype(bool)].reset_index(drop=True)
