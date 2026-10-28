import json

import numpy as np
import pytest
import soundfile as sf

from soundwatch.config import load_config
from soundwatch.data import scenarios as sc

SR = 16000


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def _write(path, x):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), x.astype(np.float32), SR, subtype="FLOAT")


@pytest.fixture
def lib_dirs(tmp_path, cfg):
    rng = np.random.default_rng(1)
    noise = tmp_path / "noise"
    for i in range(6):
        x = 0.1 * rng.standard_normal(int(SR * rng.uniform(2.5, 5.0)))
        x[: SR // 4] = 0.0  # leading silence (trimmed)
        _write(noise / f"n{i}.wav", x)
    events = tmp_path / "events"
    t = np.arange(SR) / SR
    for c, key in enumerate(cfg.class_keys[:-1]):
        for j in range(2):
            tone = 0.5 * np.sin(2 * np.pi * (400 + 150 * c + 37 * j) * t[: int(SR * 0.8)])
            x = np.concatenate(
                [np.zeros(SR // 2), tone, np.zeros(SR)]
            )  # 0.5 s silence, 0.8 s tone, 1 s silence
            _write(events / key / f"e{j}.wav", x)
    return noise, events


def test_active_region_trims_silence():
    x = np.concatenate([np.zeros(8000), 0.3 * np.ones(16000), np.zeros(4000)]).astype(np.float32)
    s, e = sc.active_region(x, SR)
    assert abs(s - 8000) <= 320 and abs(e - 24000) <= 320
    assert sc.active_region(np.zeros(SR, np.float32), SR) == (0, 0)


def test_place_events_no_overlap_min_gap():
    rng = np.random.default_rng(0)
    durs = [SR * 3, SR * 5, SR, SR * 2]
    for _ in range(50):
        on = sc.place_events(durs, SR * 60, rng, SR)
        for i in range(len(durs) - 1):
            assert on[i + 1] - (on[i] + durs[i]) >= 4 * SR
        assert on[0] >= 2 * SR and on[-1] + durs[-1] <= 57 * SR
    with pytest.raises(ValueError):
        sc.place_events([SR * 30, SR * 30], SR * 60, rng, SR)


def test_generate_scenarios(tmp_path, cfg, lib_dirs):
    noise, events = lib_dirs
    lib = sc.Library.from_dirs(cfg, noise, events)
    # trimmed to the 0.8 s tone (+ fades)
    for clips in lib.events.values():
        for _, w in clips:
            assert abs(len(w) / SR - 0.8) < 0.05
    out = tmp_path / "scen"
    idx = sc.generate(
        cfg,
        lib,
        out,
        n_events=4,
        n_background=2,
        seed=3,
        event_duration_s=40.0,
        bg_duration_s=20.0,
        verbose=False,
    )
    assert len(idx["event_scenarios"]) == 4 and len(idx["background_scenarios"]) == 2
    assert json.loads((out / "index.json").read_text())["seed"] == 3
    snr_seen = set()
    for s in idx["event_scenarios"]:
        rows = sc.read_csv(out / s["csv"])
        assert 3 <= len(rows) <= 6
        assert list(rows[0].keys())[:5] == ["class_key", "class_id", "onset_s", "offset_s", "snr_db"]
        rows.sort(key=lambda r: r["onset_s"])
        for a, b in zip(rows, rows[1:], strict=False):
            assert b["onset_s"] - a["offset_s"] >= 4.0 - 1e-3
        x, sr = sf.read(str(out / s["wav"]), dtype="float32")
        assert sr == SR and abs(len(x) / SR - 40.0) < 1e-3
        for r in rows:
            assert r["class_id"] == cfg.class_keys.index(r["class_key"])
            assert r["snr_db"] in sc.SNRS_DB
            snr_seen.add(r["snr_db"])
    assert len(snr_seen) >= 3
    for s in idx["background_scenarios"]:
        x, _ = sf.read(str(out / s["wav"]), dtype="float32")
        assert abs(len(x) / SR - 20.0) < 1e-3
        assert -45.5 <= sc.db(sc.rms(x)) <= -29.5


def test_snr_within_1db(cfg, lib_dirs):
    """Measured SNR (event energy vs the background underneath) is within +-1 dB of the label."""
    noise, events = lib_dirs
    lib = sc.Library.from_dirs(cfg, noise, events)
    rng = np.random.default_rng(5)
    for snr in sc.SNRS_DB:
        bg = sc.make_background(lib, SR * 10, rng, -35.0, SR)
        mix = bg.copy()
        _, ev = lib.events["siren"][0]
        on = 3 * SR
        sc.insert_event(mix, bg, ev, on, snr)
        seg = slice(on, on + len(ev))
        measured = sc.db(sc.rms(mix[seg] - bg[seg])) - sc.db(sc.rms(bg[seg]))
        assert abs(measured - snr) <= 1.0
        assert abs(sc.db(sc.rms(bg)) + 35.0) < 0.1


def test_scenario_snr_in_file(tmp_path, cfg, lib_dirs):
    """End to end: compare the written file with a re-synthesis of the background (same seed)."""
    noise, events = lib_dirs
    lib = sc.Library.from_dirs(cfg, noise, events)
    rng = np.random.default_rng(11)
    cyc = sc._Cycle(lib.class_keys, rng)
    snrs = sc._Cycle(sc.SNRS_DB, rng)
    clips = {k: sc._Cycle(v, rng) for k, v in lib.events.items()}
    mix, evs, meta = sc.synth_event_scenario(lib, 40.0, np.random.default_rng(2), cyc, snrs, clips)
    rng2 = np.random.default_rng(2)
    level = float(rng2.uniform(*sc.BG_LEVEL_DBFS))
    bg = sc.make_background(lib, len(mix), rng2, level, SR) * meta["peak_gain"]
    for e in evs:
        s, t = int(round(e["onset_s"] * SR)), int(round(e["offset_s"] * SR))
        measured = sc.db(sc.rms(mix[s:t] - bg[s:t])) - sc.db(max(sc.rms(bg[s:t]), 1e-7))
        assert abs(measured - e["snr_db"]) <= 1.0


def test_make_demo(tmp_path, cfg, lib_dirs):
    noise, events = lib_dirs
    lib = sc.Library.from_dirs(cfg, noise, events)
    wav = tmp_path / "demo" / "demo_scenario.wav"
    evs = sc.make_demo(cfg, lib, wav, seed=0, duration_s=60.0)
    assert sorted(e["class_key"] for e in evs) == sorted(cfg.class_keys[:-1])
    assert all(e["snr_db"] == 10.0 for e in evs)
    info = sf.info(str(wav))
    assert abs(info.frames / SR - 60.0) < 1.0 and info.subtype == "PCM_16"
    assert len(sc.read_csv(wav.with_suffix(".csv"))) == 7


def test_fallback_from_windows(tmp_path, cfg):
    rng = np.random.default_rng(0)
    waves, labels, srcs = [], [], []
    for c in range(cfg.num_classes):
        for clip in range(3):
            for _ in range(3):
                waves.append((0.2 * rng.standard_normal(SR)).astype(np.float16))
                labels.append(c)
                srcs.append(f"c{c}_{clip}")
    npz = tmp_path / "test.npz"
    np.savez(npz, wave=np.stack(waves), label=np.array(labels, np.int16), source_clip=np.array(srcs))
    lib = sc.load_library(
        cfg,
        "test",
        noise_dir=tmp_path / "nope",
        events_dir=tmp_path / "nope",
        windows_dir=tmp_path,
        verbose=False,
    )
    assert set(lib.class_keys) == set(cfg.class_keys[:-1])
    idx = sc.generate(cfg, lib, tmp_path / "s", 1, 1, 0, 30.0, 10.0, verbose=False)
    assert idx["event_scenarios"][0]["n_events"] >= 3
    with pytest.raises(FileNotFoundError):
        sc.load_library(
            cfg,
            "test",
            noise_dir=tmp_path / "x",
            events_dir=tmp_path / "x",
            windows_dir=tmp_path / "x",
            verbose=False,
        )
