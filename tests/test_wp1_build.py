import json

import numpy as np
import pandas as pd
import soundfile as sf

from soundwatch.config import load_config
from soundwatch.data import build_dataset as B

CFG = load_config()


def _tone(sr, dur, rng, amp=0.3):
    t = np.arange(int(sr * dur)) / sr
    return (amp * np.sin(2 * np.pi * 600 * t) + 0.01 * rng.standard_normal(len(t))).astype(np.float32)


def _make_esc50(root, rng):
    (root / "audio").mkdir(parents=True)
    rows = []
    cats = [
        "dog",
        "siren",
        "car_horn",
        "crying_baby",
        "door_wood_knock",
        "clock_alarm",
        "pouring_water",
        "rain",
        "cat",
    ]
    for fold in range(1, 6):
        for ti, cat in enumerate(cats):
            for take in "AB":
                name = f"{fold}-{100 + ti}-{take}-0.wav"
                x = _tone(44100, 3.0, rng)
                if cat == "dog":
                    x[44100:] = 0.0  # 2 s of digital silence -> windows must be dropped
                sf.write(root / "audio" / name, x, 44100)
                rows.append(
                    dict(
                        filename=name,
                        fold=fold,
                        target=0,
                        category=cat,
                        esc10=False,
                        src_file=100 + ti,
                        take=take,
                    )
                )
    (root / "meta").mkdir()
    pd.DataFrame(rows).to_csv(root / "meta" / "esc50.csv", index=False)


def _make_us8k(root, rng):
    rows = []
    for fold in range(1, 11):
        (root / "audio" / f"fold{fold}").mkdir(parents=True)
        for ci, cls in enumerate(["siren", "gun_shot", "jackhammer"]):
            name = f"{fold}{ci}-{ci}-0-0.wav"
            dur = 0.6 if cls == "siren" else 2.0  # short clip -> padded to 1 window
            sf.write(root / "audio" / f"fold{fold}" / name, np.stack([_tone(22050, dur, rng)] * 2, 1), 22050)
            rows.append(
                {
                    "slice_file_name": name,
                    "fsID": fold * 10 + ci,
                    "start": 0,
                    "end": dur,
                    "salience": 1,
                    "fold": fold,
                    "classID": ci,
                    "class": cls,
                }
            )
    (root / "metadata").mkdir()
    pd.DataFrame(rows).to_csv(root / "metadata" / "UrbanSound8K.csv", index=False)


def _build(tmp_path, workers=1, bg_cap=3.0):
    rng = np.random.default_rng(0)
    _make_esc50(tmp_path / "esc50", rng)
    _make_us8k(tmp_path / "us8k", rng)
    recs = B.collect_esc50(tmp_path / "esc50", CFG) + B.collect_us8k(tmp_path / "us8k", CFG)
    out = tmp_path / "windows"
    s = B.build(
        recs, CFG, out, tmp_path / "noise", tmp_path / "events", workers=workers, bg_cap=bg_cap, verbose=False
    )
    return recs, out, s


def test_build_splits_no_leakage(tmp_path):
    recs, out, summary = _build(tmp_path)
    assert not any(
        "gun_shot" in r.source_clip or "-1-0-0" in r.source_clip
        for r in recs
        if r.source_clip.startswith("us8k")
    )
    seen = {}
    for split in B.SPLITS:
        d = np.load(out / f"{split}.npz")
        assert d["wave"].dtype == np.float16 and d["wave"].shape[1] == 16000
        assert d["label"].dtype == np.int16 and len(d["label"]) == len(d["wave"]) == len(d["source_clip"])
        assert np.abs(d["wave"].astype(np.float32)).max() <= 1.0
        seen[split] = set(d["source_clip"].tolist())
        assert set(np.unique(d["label"])) == set(range(CFG.num_classes))
    assert (
        not (seen["train"] & seen["val"])
        and not (seen["train"] & seen["test"])
        and not (seen["val"] & seen["test"])
    )
    assert all(s.startswith("esc50/4-") or s.startswith("us8k/fold9/") for s in seen["val"])
    assert all(s.startswith("esc50/5-") or s.startswith("us8k/fold10/") for s in seen["test"])
    js = json.loads((out / "summary.json").read_text())
    assert js["windows"] == summary["windows"]
    # noise bank + event clips
    assert (
        len(list((tmp_path / "noise" / "train").glob("*.wav"))) == 3 * 2 * 2 + 8
    )  # rain,cat x3 folds x2 + jackhammer x8
    assert len(list((tmp_path / "noise" / "test").glob("*.wav"))) == 2 * 2 + 1
    x, sr = sf.read(next((tmp_path / "events" / "test" / "dog_bark").glob("*.wav")))
    assert sr == 16000 and len(x) == 48000
    assert (tmp_path / "events" / "val" / "siren").is_dir()


def test_silent_windows_dropped_and_padding(tmp_path):
    _, out, summary = _build(tmp_path)
    d = np.load(out / "test.npz")
    src = d["source_clip"]
    # 3 s clip -> 5 windows; dog clips have 2 s silence -> only windows overlapping the first second remain
    n_dog = (src == "esc50/5-100-A-0.wav").sum()
    n_siren = (src == "esc50/5-101-A-0.wav").sum()
    assert n_siren == 5 and n_dog == 2
    assert summary["silent_windows_dropped"]["test"]["dog_bark"] == 2 * 3
    assert (src == "us8k/fold10/100-0-0-0.wav").sum() == 1  # 0.6 s padded to one window


def test_silence_helper():
    w = np.zeros((4, 16000), np.float32)
    w[0] = 0.5
    w[1] = 0.05  # -20 dB -> kept (boundary)
    w[2] = 0.04  # < -20 dB -> dropped
    assert len(B.drop_silent(w)) == 2
    assert len(B.drop_silent(np.zeros((3, 16000), np.float32))) == 0


def test_background_cap(tmp_path):
    _, out, summary = _build(tmp_path, bg_cap=0.5)
    for s in B.SPLITS:
        w = summary["windows"][s]
        largest = max(v for k, v in w.items() if k != "background")
        assert w["background"] <= int(0.5 * largest)
        assert w["background"] < summary["windows_before_bg_cap"][s]["background"]


def test_multiprocessing_matches_serial(tmp_path):
    _, out1, s1 = _build(tmp_path / "a", workers=1)
    _, out2, s2 = _build(tmp_path / "b", workers=2)
    assert s1["windows"] == s2["windows"]
    a, b = np.load(out1 / "train.npz"), np.load(out2 / "train.npz")
    np.testing.assert_array_equal(a["wave"], b["wave"])
