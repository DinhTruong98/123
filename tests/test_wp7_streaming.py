import json
from argparse import Namespace

import numpy as np
import pytest
import soundfile as sf

from soundwatch import paths
from soundwatch.config import load_config
from soundwatch.data import scenarios as sc
from soundwatch.eval import streaming as ss
from soundwatch.types import AlertEvent

SR = 16000


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def A(c, t, repeat=False):
    return AlertEvent(class_id=c, confidence=0.9, t_audio=t, t_wall=0.0, repeat=repeat)


GT = [ss.GTEvent(0, 10.0, 13.0, 10.0), ss.GTEvent(3, 30.0, 32.0, 0.0), ss.GTEvent(0, 50.0, 52.0, 20.0)]


def test_tp_window_bounds_and_latency():
    m = ss.match_events([A(0, 10.0), A(3, 34.0), A(0, 54.01)], GT)
    assert [(i, round(lat, 3)) for i, _, lat in m.tp] == [(0, 0.0), (1, 4.0)]
    assert m.missed == [2]
    assert [k for _, k in m.fa] == ["outside"]  # 54.01 > offset + 2


def test_before_onset_is_fa():
    m = ss.match_events([A(0, 9.99)], GT)
    assert not m.tp and m.n_fa == 1 and m.fa_kind("outside") == 1


def test_duplicates_and_confusions_are_fa():
    m = ss.match_events([A(0, 11.0), A(0, 12.0), A(5, 31.0)], GT)
    assert len(m.tp) == 1 and m.tp[0][0] == 0
    assert m.fa_kind("duplicate") == 1 and m.fa_kind("confusion") == 1


def test_repeats_ignored():
    m = ss.match_events([A(0, 10.5), A(0, 15.5, repeat=True), A(0, 80.0, repeat=True)], GT)
    assert len(m.tp) == 1 and m.n_fa == 0 and m.n_repeats == 2
    # a repeat cannot create a TP either
    m = ss.match_events([A(3, 31.0, repeat=True)], GT)
    assert not m.tp and m.missed == [0, 1, 2]


def test_each_gt_matched_once_earliest_onset():
    gt = [ss.GTEvent(1, 10.0, 12.0), ss.GTEvent(1, 12.5, 14.0)]
    m = ss.match_events([A(1, 13.0), A(1, 13.5)], gt)
    assert [i for i, _, _ in m.tp] == [0, 1] and m.n_fa == 0


def test_gt_covered_seconds():
    gt = [ss.GTEvent(0, 1.0, 2.0), ss.GTEvent(0, 3.0, 4.0), ss.GTEvent(0, 20.0, 29.0)]
    assert ss.gt_covered_seconds(gt, 30.0, 2.0) == pytest.approx((6.0 - 1.0) + (30.0 - 20.0))


def test_aggregate(cfg):
    runs = [
        ss.ScenarioRun("scenario_00", 120.0, GT, [A(0, 11.0), A(3, 31.0), A(4, 90.0)]),
        ss.ScenarioRun("bg_00", 1800.0, [], [A(5, 100.0), A(1, 200.0, repeat=True)]),
    ]
    r = ss.aggregate(runs, cfg)
    assert r["n_gt_events"] == 3 and r["n_tp"] == 2
    assert r["event_recall"] == pytest.approx(2 / 3)
    assert r["per_class"]["smoke_alarm"]["recall"] == 0.5
    assert r["per_snr"]["10"]["recall"] == 1.0 and r["per_snr"]["20"]["recall"] == 0.0
    assert r["fa_per_hour_background"] == pytest.approx(2.0)
    assert r["background_false_alarms"] == 1 and r["repeat_alerts_ignored"] == 1
    non_gt_h = (120.0 - 5.0 - 4.0 - 4.0) / 3600
    assert r["fa_per_hour_event_outside_gt"] == pytest.approx(1 / non_gt_h)
    assert r["latency_median_s"] == pytest.approx(1.0)
    assert r["per_class"]["dog_bark"]["fa_background"] == 1


def _write_scenarios(d, cfg):
    """2 event scenarios with loud noise bursts as 'events' + 1 background file."""
    rng = np.random.default_rng(0)
    d.mkdir(parents=True)
    for i, evs in enumerate([[(0, 5.0, 7.0), (3, 15.0, 17.0)], [(5, 6.0, 8.0)]]):
        x = (0.01 * rng.standard_normal(SR * 25)).astype(np.float32)
        rows = []
        for c, on, off in evs:
            x[int(on * SR) : int(off * SR)] += 0.1 * rng.standard_normal(int((off - on) * SR)).astype(
                np.float32
            )
            rows.append(
                {
                    "class_key": cfg.class_keys[c],
                    "class_id": c,
                    "onset_s": on,
                    "offset_s": off,
                    "snr_db": 20.0,
                    "source_clip": "x",
                }
            )
        sf.write(str(d / f"scenario_{i:02d}.wav"), x, SR, subtype="PCM_16")
        sc.write_csv(d / f"scenario_{i:02d}.csv", rows)
    x = (0.003 * rng.standard_normal(SR * 30)).astype(np.float32)
    x[: SR * 10] = 0.0  # gated part
    sf.write(str(d / "bg_00.wav"), x, SR, subtype="PCM_16")


def test_end_to_end_dummy_with_sweep(tmp_path, cfg, monkeypatch):
    scen = tmp_path / "scen"
    _write_scenarios(scen, cfg)
    rep = tmp_path / "reports"
    monkeypatch.setattr(paths, "SCENARIOS", scen)
    monkeypatch.setattr(paths, "REPORTS", rep)
    args = Namespace(
        config=None, model=None, stats=None, dummy=True, sweep=True, scenarios=None, limit=None, out_dir=None
    )
    assert ss.main(args) == 0
    res = json.loads((rep / "streaming_metrics_dummy.json").read_text())
    assert res["n_gt_events"] == 3 and res["event_recall"] == 1.0
    assert res["background_false_alarms"] == 0
    assert 0.2 < res["engine"]["gated_fraction_background"] < 0.5
    assert res["latency_median_s"] <= 2.0
    assert (rep / "streaming_sweep_dummy.png").exists()
    import yaml

    y = yaml.safe_load((rep / "suggested_thresholds_dummy.yaml").read_text())
    assert set(y["thresholds"]) == set(cfg.class_keys[:-1])
    # oracle confidence 1.0 (like the saturating int8 model) -> recall 1 over the whole grid
    g = {r["threshold"]: r["recall"] for r in res["sweep"]["global"]}
    assert g[0.3] == 1.0 and g[0.95] == 1.0


def test_suggest_thresholds_respects_budget(cfg):
    grid = [0.3, 0.5, 0.7, 0.9]
    keys = cfg.class_keys[:-1]
    per_class = {}
    for k in keys:
        # lower threshold -> more TP but also more FA
        per_class[k] = [
            {"threshold": t, "tp": tp, "n_gt": 4, "recall": tp / 4, "fa_background": fa}
            for t, tp, fa in zip(grid, [4, 3, 2, 0], [3, 1, 0, 0], strict=True)
        ]
    sw = {"grid": grid, "per_class": per_class}
    s = ss.suggest_thresholds(cfg, sw, bg_hours=1.0, budget_per_h=2.0)
    total_fa = sum(v["fa_background"] for v in s.values())
    assert total_fa <= 2
    assert all(v["threshold"] <= 0.7 for v in s.values())  # 0.7 gains recall at zero FA
    # DANGER classes are preferred for the FA budget
    danger = [c.key for c in cfg.classes[:-1] if int(c.level) == 2]
    assert all(s[k]["threshold"] == 0.5 for k in danger[:2])
