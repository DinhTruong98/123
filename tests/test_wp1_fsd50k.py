"""Offline tests for the FSD50K extras (candidate selection, YAMNet rule, dataset integration)."""

import json

import numpy as np
import pandas as pd
import soundfile as sf

from soundwatch.config import load_config
from soundwatch.data import build_dataset as B
from soundwatch.data import fsd50k as F
from soundwatch.data import label_map as lm

CFG = load_config()


def _write_meta(d, split, rows):
    """rows: (fname, gt labels, collection labels, title, tags, description, dev split)."""
    (d / "FSD50K.ground_truth").mkdir(parents=True, exist_ok=True)
    (d / "FSD50K.metadata" / "collection").mkdir(parents=True, exist_ok=True)
    gt = pd.DataFrame(
        [dict(fname=r[0], labels=r[1], mids="x", **({"split": r[6]} if split == "dev" else {})) for r in rows]
    )
    gt.to_csv(d / "FSD50K.ground_truth" / f"{split}.csv", index=False)
    pd.DataFrame([dict(fname=r[0], labels=r[2], mids="x") for r in rows]).to_csv(
        d / "FSD50K.metadata" / "collection" / f"collection_{split}.csv", index=False
    )
    info = {
        r[0]: dict(title=r[3], tags=r[4].split(), description=r[5], license="cc0", uploader="u") for r in rows
    }
    (d / "FSD50K.metadata" / f"{split}_clips_info_FSD50K.json").write_text(json.dumps(info))


def _meta(tmp_path):
    d = tmp_path / "fsd50k"
    _write_meta(
        d,
        "dev",
        [
            ("1", "Crying_and_sobbing,Human_voice", "Baby_cry_and_infant_cry", "a.wav", "", "", "train"),
            ("2", "Crying_and_sobbing,Human_voice", "Crying_and_sobbing", "newborn", "", "", "val"),
            ("3", "Crying_and_sobbing,Human_voice", "Crying_and_sobbing", "man sobbing", "sad", "", "train"),
            ("4", "Alarm", "Fire_alarm", "beep.wav", "", "", "train"),
            ("5", "Alarm", "Alarm", "Smoke detector chirp", "", "", "train"),
            ("6", "Alarm", "Alarm", "Fire alarm clock", "", "", "train"),  # clock -> excluded
            ("7", "Alarm,Siren", "Fire_alarm", "fire alarm", "", "", "train"),  # Siren label -> excluded
            ("8", "Alarm", "Alarm", "beeps", "electronic", "", "val"),  # generic -> acoustic tier
            ("9", "Alarm,Mechanisms", "Alarm_clock", "alarm", "", "", "train"),  # alarm clock
            ("10", "Alarm,Telephone", "Alarm", "smoke alarm phone", "", "", "train"),
            ("11", "Dog,Animal", "Dog", "baby dog", "", "", "train"),  # not crying
        ],
    )
    _write_meta(d, "eval", [("20", "Crying_and_sobbing", "", "x", "infant", "", None)])
    return d


def test_find_candidates(tmp_path):
    c = F.find_candidates(_meta(tmp_path)).set_index("fname")
    assert c.loc["1", "target"] == "baby_cry" and c.loc["1", "fsd_split"] == "train"
    assert c.loc["2", "target"] == "baby_cry" and c.loc["2", "fsd_split"] == "val"
    assert c.loc["20", "target"] == "baby_cry" and c.loc["20", "fsd_split"] == "eval"
    assert "3" not in c.index and "11" not in c.index
    assert c.loc["4", "tier"] == "keyword" and c.loc["5", "tier"] == "keyword"
    assert c.loc["8", "tier"] == "acoustic" and c.loc["8", "target"] == "smoke_alarm"
    for f in ("6", "7", "9", "10"):
        assert f not in c.index


def test_keep_rule_and_scoring():
    names = [f"c{i}" for i in range(10)]
    sc = np.zeros((3, 10), np.float32)
    sc[:, :6] = 0.5  # target 7 never in top-5
    sc[1, 7] = 0.2
    s, hit, top = F.score_clip(sc, [7], names)
    assert s == np.float32(0.2) and not hit and top.startswith("c0=0.50")
    sc[2, 7] = 0.9
    s, hit, _ = F.score_clip(sc, [7, 8], names)
    assert hit and abs(s - 0.9) < 1e-6
    assert F.keep_rule("baby_cry", "keyword", 0.9)
    assert not F.keep_rule("baby_cry", "keyword", 0.01)
    assert F.keep_rule("baby_cry", "keyword", 0.9, siren=0.99)  # siren check only for smoke_alarm
    assert not F.keep_rule("smoke_alarm", "acoustic", 0.4)  # stricter than the keyword tier
    assert F.keep_rule("smoke_alarm", "keyword", 0.4)
    assert F.keep_rule("smoke_alarm", "acoustic", 0.9, siren=0.5)
    assert not F.keep_rule("smoke_alarm", "keyword", 0.9, siren=0.95)  # siren-like fire alarm
    assert F.active_patches(np.array([0.0, 0.9, 0.2, 0.3, 0.04])) == [1, 3]
    assert F.active_patches(np.array([0.06, 0.04])) == [0]
    assert F.active_starts("0;2") == (0.0, 0.96) and F.active_starts(float("nan")) == ()


def test_active_window_mask():
    # windows (1 s, hop 0.5 s) centred at 0.5, 1.0, 1.5, 2.0, 2.5; active patch [0.96, 1.92]
    m = B.active_window_mask(5, 1.0, 0.5, (0.96,))
    assert m.tolist() == [False, True, True, False, False]
    assert not B.active_window_mask(3, 1.0, 0.5, ()).any()


def test_patch_frames():
    assert F.patch_frames(np.zeros(1000)).shape == (1, 15600)
    assert F.patch_frames(np.zeros(15600 + 2 * 7680)).shape == (3, 15600)


def test_label_map_fsd50k_and_clock_alarm():
    smoke = CFG.class_keys.index("smoke_alarm")
    assert lm.map_fsd50k("smoke_alarm", CFG) == smoke
    assert lm.map_label("fsd50k", "baby_cry", CFG) == CFG.class_keys.index("baby_cry")
    assert lm.map_esc50("clock_alarm", CFG) == smoke
    assert lm.map_esc50("clock_alarm", CFG, clock_alarm_background=True) == CFG.background_id
    assert lm.map_esc50("siren", CFG, clock_alarm_background=True) == CFG.class_keys.index("siren")


def test_cap_windows():
    w = np.arange(25)[:, None] * np.ones((1, 4))
    c = B.cap_windows(w, 10)
    assert len(c) == 10 and c[0, 0] == 0 and c[-1, 0] == 24 and np.all(np.diff(c[:, 0]) > 0)
    assert B.cap_windows(w, None) is w and B.cap_windows(w[:3], 10).shape == (3, 4)


def test_build_with_fsd50k(tmp_path):
    root = tmp_path / "fsd"
    rng = np.random.default_rng(0)
    rows = []
    for i, (split, fsd_split, target, dur) in enumerate(
        [
            ("dev", "train", "smoke_alarm", 20.0),  # long -> capped
            ("dev", "val", "baby_cry", 2.0),
            ("eval", "eval", "smoke_alarm", 0.4),  # short -> padded
            ("eval", "eval", "baby_cry", 3.0),
        ]
    ):
        (root / split).mkdir(parents=True, exist_ok=True)
        t = np.arange(int(44100 * dur)) / 44100
        x = 0.3 * np.sin(2 * np.pi * 3000 * t) + 0.01 * rng.standard_normal(len(t))
        sf.write(root / split / f"{100 + i}.wav", x.astype(np.float32), 44100)
        act = "0" if i == 1 else ";".join(map(str, range(60)))  # clip 101: event in the first second only
        rows.append(
            dict(fname=str(100 + i), split=split, fsd_split=fsd_split, target=target, active_patches=act)
        )
    recs = B.collect_fsd50k(root, pd.DataFrame(rows), CFG, max_windows=10)
    assert [r.split for r in recs] == ["train", "val", "test", "test"]
    assert recs[0].source_clip == "fsd50k/dev/100.wav"
    s = B.build(recs, CFG, tmp_path / "w", tmp_path / "noise", tmp_path / "events", workers=1, verbose=False)
    smoke = CFG.class_keys.index("smoke_alarm")
    tr = np.load(tmp_path / "w" / "train.npz")
    assert (tr["label"] == smoke).sum() == 10 and s["windows_filtered"]["train"]["smoke_alarm"] == 39 - 10
    te = np.load(tmp_path / "w" / "test.npz")
    assert (te["source_clip"] == "fsd50k/eval/102.wav").sum() == 1
    assert len(list((tmp_path / "events" / "test" / "smoke_alarm").glob("fsd50k_eval_*.wav"))) == 1
    assert len(list((tmp_path / "events" / "val" / "baby_cry").glob("*.wav"))) == 1
    va = np.load(tmp_path / "w" / "val.npz")
    # 2 s clip -> 3 windows (centres 0.5/1.0/1.5 s); only the first is centred in active patch [0, 0.96]
    assert len(va["label"]) == 1 and s["windows_filtered"]["val"]["baby_cry"] == 2


def test_resolve_clock_alarm():
    smoke = CFG.class_keys.index("smoke_alarm")
    recs = [
        B.ClipRecord(f"fsd50k/dev/{i}.wav", "", smoke, "train") for i in range(B.CLOCK_ALARM_MIN_FSD_TRAIN)
    ]
    assert B.resolve_clock_alarm("auto", recs, CFG)
    assert not B.resolve_clock_alarm("auto", recs[:-1], CFG)
    assert not B.resolve_clock_alarm("smoke_alarm", recs, CFG)
    assert B.resolve_clock_alarm("background", [], CFG)
