import json
from argparse import Namespace

import pytest
import yaml

from soundwatch import paths
from soundwatch.config import load_config
from soundwatch.eval import report


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def test_report_all_missing(tmp_path, cfg, monkeypatch):
    for attr in ("REPORTS", "MODELS", "WINDOWS"):
        monkeypatch.setattr(paths, attr, tmp_path / attr.lower())
    assert report.main(Namespace(config=None, out=None)) == 0
    md = (tmp_path / "reports" / "report.md").read_text(encoding="utf-8")
    assert "chưa có" in md and "Đối chiếu mục tiêu" in md
    assert "✅" not in md.split("## 1.")[0].split("| Đạt? |")[1]


def _fake(cfg, tmp_path):
    rep, mod, win = tmp_path / "reports", tmp_path / "models", tmp_path / "windows"
    for d in (rep, mod, win):
        d.mkdir()
    keys = cfg.class_keys
    (win / "summary.json").write_text(
        json.dumps(
            {
                "windows": {s: {k: 10 for k in keys} for s in ("train", "test")},
                "clips": {"train": {k: 2 for k in keys}},
            }
        )
    )
    per = {k: {"precision": 0.9, "recall": 0.9, "f1": 0.9, "support": 10} for k in keys}
    m = {
        "accuracy": 0.9,
        "macro_f1": 0.85,
        "danger_recall_mean": 0.9,
        "danger_recall_min": 0.88,
        "per_class": per,
        "confusion_png": "offline_student_int8.png",
    }
    (rep / "offline_metrics.json").write_text(
        json.dumps(
            {
                "models": {"student_int8": m, "teacher": dict(m, confusion_png="offline_teacher.png")},
                "missing": {"student_nodistill": "not found"},
                "int8_drop": {
                    "reference": "student_float_tflite",
                    "accuracy_pts": 0.5,
                    "macro_f1_pts": 0.3,
                    "top1_agreement": 0.99,
                },
            }
        )
    )
    pc = {
        k: {
            "n_gt": 5,
            "tp": 4,
            "recall": 0.8,
            "latency_median_s": 1.0,
            "latency_p90_s": 1.5,
            "fa_background": 0,
            "fa_event_scenarios": 1,
        }
        for k in keys[:-1]
    }
    (rep / "streaming_metrics.json").write_text(
        json.dumps(
            {
                "model": "x.tflite",
                "n_event_scenarios": 20,
                "n_background_scenarios": 10,
                "n_gt_events": 35,
                "event_recall": 0.8,
                "danger_event_recall_mean": 0.8,
                "danger_event_recall_min": 0.7,
                "latency_median_s": 1.2,
                "latency_p90_s": 2.0,
                "background_hours": 1.0,
                "background_false_alarms": 3,
                "fa_per_hour_background": 3.0,
                "fa_per_hour_event_outside_gt": 1.0,
                "event_scenarios_false_alarms": {"duplicate": 1, "confusion": 2, "outside": 3},
                "engine": {"gated_fraction_background": 0.1},
                "per_class": pc,
                "per_snr": {"0": {"n_gt": 9, "tp": 5, "recall": 0.55}},
            }
        )
    )
    (rep / "streaming_sweep.png").write_bytes(b"png")
    with open(rep / "suggested_thresholds.yaml", "w") as f:
        yaml.safe_dump(
            {
                "thresholds": {k: 0.5 for k in keys[:-1]},
                "current": {k: 0.6 for k in keys[:-1]},
                "expected": {"event_recall": 0.9, "fa_per_hour_background": 2.0},
            },
            f,
        )
    est = [
        {
            "target": "M4",
            "mac_per_cycle": [0.25, 0.5],
            "latency_ms_best": 100.0,
            "latency_ms_worst": 200.0,
            "duty_cycle_worst": 0.4,
            "note": "n",
        }
    ]
    (rep / "benchmark.json").write_text(
        json.dumps(
            {
                "runs": 1000,
                "feature_ms": {"median": 0.5, "p95": 0.7},
                "models": {
                    "student_int8": {
                        "format": "tflite",
                        "size_bytes": 60000,
                        "params": 25000,
                        "macs": 3_000_000,
                        "arena_estimate_bytes": 200_000,
                        "infer_ms": {"median": 0.4, "p95": 0.6},
                        "total_ms_median": 0.9,
                        "mcu_estimates": est,
                    }
                },
                "gating": {"gated_fraction": 0.1, "source": "test"},
            }
        )
    )
    (rep / "ui_idle.png").write_bytes(b"png")
    return rep, mod, win


def test_report_with_fake_inputs(tmp_path, cfg):
    rep, mod, win = _fake(cfg, tmp_path)
    md = report.build_report(cfg, rep, mod, win)
    rows = {
        r[0]: r
        for r in report.targets_checklist(
            json.loads((rep / "offline_metrics.json").read_text()),
            json.loads((rep / "streaming_metrics.json").read_text()),
            json.loads((rep / "benchmark.json").read_text()),
            None,
        )
    }
    assert rows["Macro-F1 test (student int8)"][3] == "✅"
    assert rows["Tensor arena ước tính (KB)"][3] == "❌"
    assert rows["Báo nhầm / giờ (chỉ nền)"][3] == "❌"
    assert rows["Độ trễ median onset → cảnh báo (s)"][3] == "✅"
    assert rows["UI FPS"][3] == "⬜"
    assert "![sweep](streaming_sweep.png)" in md and "offline_teacher.png" in md
    assert "ui_idle.png" in md and "ƯỚC TÍNH" in md and "Báo cháy" in md
    assert "7/9 đạt" in md or "đạt" in md


def test_report_quant_fallback(cfg):
    quant = {
        "test": {"tflite_int8": {"macro_f1": 0.7, "danger_recall_min": 0.9}, "int8_drop_accuracy_pts": 1.0},
        "size_bytes": {"tflite_int8": 50_000},
        "macs": 2_000_000,
        "int8": {"arena_estimate_bytes": 100_000},
    }
    rows = {r[0]: r for r in report.targets_checklist(None, None, None, quant)}
    assert rows["Macro-F1 test (student int8)"][3] == "❌"
    assert rows["Recall lớp DANGER (min, student int8)"][3] == "✅"
    assert rows["Kích thước .tflite int8 (KB)"][3] == "✅"
    assert rows["MACs / lần suy luận (M)"][2] == "2.00"
