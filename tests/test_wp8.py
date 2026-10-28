"""WP8: post-processing tuning, bg_margin option, FA analysis helpers, record, UI stats / GIF."""

from __future__ import annotations

import dataclasses
import json
import os
import queue
import sys
import time
from argparse import Namespace

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import soundfile as sf  # noqa: E402

from soundwatch.config import load_config  # noqa: E402
from soundwatch.engine.pipeline import StatusSummary  # noqa: E402
from soundwatch.engine.postprocess import PostProcessor  # noqa: E402
from soundwatch.eval import fa_analysis as fa  # noqa: E402
from soundwatch.eval import tune  # noqa: E402
from soundwatch.eval.streaming import GTEvent, ScenarioRun  # noqa: E402
from soundwatch.types import AlertEvent, EngineStopped  # noqa: E402

SR = 16000


@pytest.fixture(scope="module")
def cfg():
    # these tests exercise smoothing / k-of-n tuning: keep the DANGER fast path (rule 4b) off
    c = load_config()
    return dataclasses.replace(c, postprocess=dataclasses.replace(c.postprocess, danger_fast_threshold=None))


def _probs(cfg, cid, p):
    v = np.full(cfg.num_classes, (1 - p) / (cfg.num_classes - 1), np.float32)
    v[cid] = p
    return v


# ---------------------------------------------------------------- bg_margin
def _pp_cfg(cfg, margin, thr=0.3):
    pp = dataclasses.replace(cfg.postprocess, smooth_n=1, k_of_n=(1, 1), bg_margin=margin)
    classes = tuple(
        dataclasses.replace(c, threshold=thr) if c.id != cfg.background_id else c for c in cfg.classes
    )
    return classes, pp


def test_bg_margin_default_off(cfg):
    assert load_config().postprocess.bg_margin is None  # shipped config: rule off
    classes, pp = _pp_cfg(cfg, None)
    dog = cfg.class_by_key("dog_bark").id
    p = np.zeros(cfg.num_classes, np.float32)
    p[dog], p[cfg.background_id] = 0.40, 0.60  # above threshold 0.3 but below background
    ev = PostProcessor(classes, pp).update(p, 1.0, t_wall=0.0)
    assert [e.class_id for e in ev] == [dog]


def test_bg_margin_blocks_when_background_close(cfg):
    dog = cfg.class_by_key("dog_bark").id
    p = np.zeros(cfg.num_classes, np.float32)
    p[dog], p[cfg.background_id] = 0.40, 0.60
    classes, pp = _pp_cfg(cfg, 0.1)
    assert PostProcessor(classes, pp).update(p, 1.0, t_wall=0.0) == []
    p[dog], p[cfg.background_id] = 0.55, 0.40  # margin 0.15 >= 0.1
    assert [e.class_id for e in PostProcessor(classes, pp).update(p, 1.0, t_wall=0.0)] == [dog]


def test_bg_margin_from_yaml(tmp_path):
    from soundwatch import paths

    txt = paths.CONFIG.read_text(encoding="utf-8").replace(
        "danger_clear_s: 10.0", "danger_clear_s: 10.0\n  bg_margin: 0.25"
    )
    p = tmp_path / "c.yaml"
    p.write_text(txt, encoding="utf-8")
    assert load_config(p).postprocess.bg_margin == pytest.approx(0.25)


# ---------------------------------------------------------------- tune
def _summary(cfg, probs_by_t: dict[float, np.ndarray], duration: float) -> StatusSummary:
    ts = np.arange(1.0, duration + 1e-9, 0.5)
    bg = _probs(cfg, cfg.background_id, 1.0)
    P = np.stack([probs_by_t.get(round(float(t), 1), bg) for t in ts]).astype(np.float32)
    n = len(ts)
    return StatusSummary(ts, np.full(n, -30.0), np.zeros(n, bool), np.zeros(n), P)


def _synthetic_runs(cfg):
    siren = cfg.class_by_key("siren").id
    dog = cfg.class_by_key("dog_bark").id
    # event run: siren 10-13 s detected confidently from t=11.0 on
    ev = {round(t, 1): _probs(cfg, siren, 1.0) for t in np.arange(11.0, 13.6, 0.5)}
    run_ev = ScenarioRun("scenario_00", 30.0, [GTEvent(siren, 10.0, 13.0, 10.0)], [], _summary(cfg, ev, 30.0))
    # background run: isolated single-hop dog spikes (never two hops in a row)
    bgp = {20.0: _probs(cfg, dog, 1.0), 120.0: _probs(cfg, dog, 1.0), 300.0: _probs(cfg, dog, 1.0)}
    run_bg = ScenarioRun("bg_00", 3600.0, [], [], _summary(cfg, bgp, 3600.0))
    return [run_ev, run_bg]


def test_class_tables_counts(cfg):
    runs = _synthetic_runs(cfg)
    t11 = tune.class_tables(cfg, runs, tune.PPSetting(1, 1, 1), grid=(0.5, 0.99))
    assert [r["tp"] for r in t11["siren"]] == [1, 1]
    assert [r["fa_bg"] for r in t11["dog_bark"]] == [3, 3]  # every spike alerts with k/n = 1/1
    assert t11["siren"][0]["lat"] == [pytest.approx(1.0)]
    t22 = tune.class_tables(cfg, runs, tune.PPSetting(1, 2, 2), grid=(0.5, 0.99))
    assert [r["fa_bg"] for r in t22["dog_bark"]] == [0, 0]  # single-hop spikes are rejected
    assert t22["siren"][0]["lat"] == [pytest.approx(1.5)]


def test_greedy_respects_budget_and_weights():
    # rows ordered by increasing threshold (grid order); lower threshold = more TP and more FA
    def tab(tps, fas):
        return [
            {"thr": t, "tp": tp, "n_gt": 4, "fa_bg": f, "fa_ev": 0, "lat": []}
            for t, tp, f in zip((0.3, 0.6, 0.9), tps, fas, strict=True)
        ]

    tabs = {"a": tab([4, 3, 1], [4, 1, 0]), "b": tab([4, 2, 1], [3, 1, 0])}
    idx, ok = tune.greedy_thresholds(tabs, {"a": 1.0, "b": 2.0}, budget=2)
    assert ok and sum(tabs[k][i]["fa_bg"] for k, i in idx.items()) <= 2
    assert idx == {"a": 1, "b": 1}
    idx, ok = tune.greedy_thresholds(tabs, {"a": 1.0, "b": 2.0}, budget=100)
    assert ok and idx == {"a": 0, "b": 0}
    infeasible = {"a": tab([4, 3, 1], [4, 2, 1])}
    idx, ok = tune.greedy_thresholds(infeasible, {"a": 1.0}, budget=0)
    assert not ok and idx == {"a": 2}


def test_pareto_front():
    pts = [
        {"weighted_recall": 0.5, "fa_per_hour_background": 2.0, "latency_median_s": 1.0},
        {"weighted_recall": 0.4, "fa_per_hour_background": 3.0, "latency_median_s": 1.2},  # dominated
        {"weighted_recall": 0.6, "fa_per_hour_background": 5.0, "latency_median_s": 1.0},
    ]
    assert tune.pareto_front(pts) == [0, 2]


def test_tune_picks_kofn_2_2_under_zero_budget(cfg):
    runs = _synthetic_runs(cfg)
    settings = [tune.PPSetting(1, 1, 1), tune.PPSetting(1, 2, 2), tune.PPSetting(3, 2, 3)]
    res = tune.tune(cfg, runs, settings, budgets=(0.0, 5.0), max_latency=1.5, jobs=1, verbose=False)
    best0 = res["best"]["0"]
    assert best0["smooth_n"] == 1 and best0["k_of_n"] == [2, 2] and best0["fa_bg"] == 0
    assert best0["per_class_recall"]["siren"] == 1.0
    # smooth 3: latency 2.0 s > 1.5 s -> never chosen
    assert all(p["smooth_n"] == 1 for p in res["best"].values() if p)
    p11 = [p for p in res["points"] if p["setting"] == settings[0].label and p["budget_per_h"] == 0.0][0]
    assert not p11["feasible"]
    md = tune.render_md(res, "m.tflite", "val")
    assert "Pareto" in md and "smooth=1 k/n=2/2" in md
    assert set(res["tables"][settings[1].label]) == set(cfg.class_keys[:-1])


# ---------------------------------------------------------------- FA analysis helpers
def test_parse_clip_and_segments():
    u = fa.parse_clip("us8k_fold10_115241-9-0-9.wav")
    assert u["dataset"] == "UrbanSound8K" and u["fsid"] == 115241 and u["label"] == "street_music"
    e = fa.parse_clip("esc50_5-216213-A-13.wav")
    assert e["dataset"] == "ESC-50" and e["class_id"] == 13
    segs = [(0.0, 4.0, "a.wav"), (3.5, 7.5, "b.wav"), (7.0, 9.0, "c.wav")]
    assert sorted(fa.segments_at(segs, 3.0, 4.5)) == [("a.wav", 1.0), ("b.wav", 1.0)]
    assert [n for n, _ in fa.segments_at(segs, 6.0, 7.5)] == ["b.wav", "c.wav"]
    assert fa.group_indices(["Speech", "Siren", "Bark"], "siren") == [1]


def test_fa_analyse_without_yamnet(cfg, tmp_path):
    st = {
        "model": "m",
        "postprocess": {},
        "scenarios": [
            {
                "name": "bg_00",
                "alerts": [{"class_key": "siren", "t": 5.0, "conf": 0.99, "repeat": False}],
                "fa_alerts": [{"class_key": "siren", "t": 5.0, "kind": "outside"}],
            },
            {"name": "scenario_00", "alerts": [], "fa_alerts": [{"class_key": "dog_bark", "t": 1.0}]},
        ],
    }
    index = {
        "background_scenarios": [
            {"wav": "bg_00.wav", "bg_segments": [[0.0, 10.0, "us8k_fold10_1-9-0-0.wav"]]}
        ]
    }
    res = fa.analyse(cfg, st, index, tmp_path, tmp_path, top=5, use_yamnet=False)
    assert res["n_analysed"] == 1  # only background scenarios
    r = res["rows"][0]
    assert r["clip"] == "us8k_fold10_1-9-0-0.wav" and "street_music" in r["clip_label"]
    assert not r["label_noise_likely"]
    md = fa.render_md(res, st)
    assert "bg_00" in md and "Tóm tắt" in md


# ---------------------------------------------------------------- scenarios: segments
def test_make_background_segments_do_not_change_audio(cfg):
    from soundwatch.data import scenarios as sc

    rng = np.random.default_rng(0)
    noise = [(f"n{i}.wav", 0.1 * rng.standard_normal(SR * 3).astype(np.float32)) for i in range(4)]
    lib = sc.Library(cfg, noise, {"siren": [("s.wav", np.ones(SR, np.float32))]})
    segs: list = []
    a = sc.make_background(lib, SR * 10, np.random.default_rng(5), -35.0, SR, segments=segs)
    b = sc.make_background(lib, SR * 10, np.random.default_rng(5), -35.0, SR)
    np.testing.assert_array_equal(a, b)
    assert segs[0][0] == 0.0 and segs[-1][1] == pytest.approx(10.0)
    assert all(s[2].startswith("n") for s in segs)
    assert all(segs[i + 1][0] < segs[i][1] for i in range(len(segs) - 1))  # crossfaded overlap


def test_load_library_val_uses_windows_noise_and_event_dirs(cfg, tmp_path):
    from soundwatch.data import scenarios as sc

    ev = tmp_path / "events" / "val" / "siren"
    ev.mkdir(parents=True)
    sf.write(str(ev / "e.wav"), 0.5 * np.sin(np.arange(SR * 3) / 5.0).astype(np.float32), SR)
    rng = np.random.default_rng(0)
    n = 6
    np.savez(
        tmp_path / "val.npz",
        wave=(0.1 * rng.standard_normal((n, SR))).astype(np.float16),
        label=np.full(n, cfg.background_id),
        source_clip=np.array(["bgclip"] * n),
    )
    lib = sc.load_library(
        cfg,
        "val",
        noise_dir=tmp_path / "noise" / "val",
        events_dir=tmp_path / "events" / "val",
        windows_dir=tmp_path,
        verbose=False,
    )
    assert [k for k in lib.events] == ["siren"]
    assert len(lib.events["siren"][0][1]) > SR  # full clip, not a 1 s window
    assert lib.noise[0][0] == "bgclip"


# ---------------------------------------------------------------- record
class _FakeSD:
    def __init__(self):
        self.calls = []

    def rec(self, n, samplerate, channels, dtype, device=None):
        self.calls.append((n, samplerate, channels, device))
        return (0.25 * np.ones((n, channels))).astype(np.float32)

    def wait(self):
        pass


def test_record_writes_16k_wav(cfg, tmp_path, monkeypatch):
    from soundwatch.tools import record

    fake = _FakeSD()
    monkeypatch.setitem(sys.modules, "sounddevice", fake)
    args = Namespace(
        config=None,
        class_key="door_knock",
        seconds=0.5,
        count=2,
        device="3",
        out_dir=str(tmp_path),
        countdown=0,
    )
    assert record.main(args) == 0
    files = sorted((tmp_path / "door_knock").glob("*.wav"))
    assert len(files) == 2
    x, sr = sf.read(str(files[0]))
    assert sr == 16000 and len(x) == 8000 and abs(x.max() - 0.25) < 1e-3
    assert fake.calls[0] == (8000, 16000, 1, 3)


def test_record_errors(cfg, tmp_path, monkeypatch, capsys):
    from soundwatch.tools import record

    monkeypatch.setitem(sys.modules, "sounddevice", None)  # import -> ImportError
    args = Namespace(
        config=None,
        class_key="door_knock",
        seconds=1.0,
        count=1,
        device=None,
        out_dir=str(tmp_path),
        countdown=0,
    )
    assert record.main(args) == 1
    assert "PortAudio" in capsys.readouterr().err
    assert record.main(Namespace(**{**vars(args), "class_key": "cat"})) == 2  # unknown class


# ---------------------------------------------------------------- UI stats + GIF
def test_run_app_writes_stats_and_gif(cfg, tmp_path):
    from soundwatch.ui.app import run_app

    out_q: queue.Queue = queue.Queue()
    in_q: queue.Queue = queue.Queue()
    out_q.put(AlertEvent(cfg.class_by_key("siren").id, 0.99, 1.0, time.time()))
    gif, stats = tmp_path / "d.gif", tmp_path / "ui_stats.json"
    st = run_app(cfg, out_q, in_q, show_debug=True, max_seconds=1.0, record_gif=gif, stats_path=stats)
    assert gif.exists() and gif.stat().st_size > 0 and st["gif_frames"] >= 2
    d = json.loads(stats.read_text())
    assert d["fps"] > 0 and d["fps_capacity"] > 0 and d["target_fps"] == cfg.ui.fps
    out_q.put(EngineStopped("eof"))


def test_report_ui_fps_and_new_sections(cfg, tmp_path):
    from soundwatch.eval import report

    rows = {
        r[0]: r for r in report.targets_checklist(None, None, None, None, {"fps": 29.8, "fps_capacity": 150})
    }
    assert rows["UI FPS (khung hình thực / khả năng không giới hạn)"][3] == "✅"
    rows = {r[0]: r for r in report.targets_checklist(None, None, None, None, {"fps": 20.0})}
    assert rows["UI FPS (khung hình thực / khả năng không giới hạn)"][3] == "❌"
    assert "tuning.md" in report.section_tuning(cfg, {"best": {"2": None}, "scenarios": "val"})
    res = {
        "n_analysed": 1,
        "n_label_noise_likely": 0,
        "yamnet_hit": 0.2,
        "rows": [
            {
                "class_key": "siren",
                "clip_label": "UrbanSound8K: street_music",
                "label_noise_likely": False,
                "recording_has_target_label": False,
            }
        ],
    }
    txt = report.section_fa_analysis(res, None)
    assert "Kết luận" in txt and "lỗi của model" in txt
