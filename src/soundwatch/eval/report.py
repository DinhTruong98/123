"""WP7/WP8: assemble artifacts/reports/report.md from the JSON / PNG outputs of the other commands.

Inputs (all optional - missing ones become "chưa có" rows, the command never fails on them):
  data/windows/summary.json, reports/offline_metrics.json, reports/streaming_metrics.json,
  reports/benchmark.json, reports/suggested_thresholds.yaml, models/quant_metrics.json,
  models/student_<tag>_metrics.json, models/teacher_metrics.json, reports/ui_*.png,
  reports/tuning.json (`tune`), reports/fa_analysis.json (`fa-analysis`), reports/ui_stats.json
  (`run --ui-stats` / `report --measure-ui-fps`), reports/demo.gif (`run --record-gif`)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from soundwatch import paths
from soundwatch.config import Config, load_config
from soundwatch.types import Level

NA = "chưa có"
TARGETS = {  # plan section 2 (hardware target ESP32-S3 + PSRAM)
    "macro_f1": 0.80,
    "danger_recall": 0.85,
    "int8_drop_pts": 2.0,
    "size_kb": 500.0,
    "arena_kb": 512.0,
    "macs_m": 50.0,
    "fa_per_h": 2.0,
    "latency_s": 1.5,
    "pc_ms": 20.0,
    "ui_fps": 30.0,
}
TARGETS_OLD = {"size_kb": 100.0, "arena_kb": 150.0, "macs_m": 10.0}  # nRF52840 (cũ)
MODEL_TITLES = {
    "teacher": "Teacher (YAMNet + head)",
    "student_distill": "Student float (distill, Keras)",
    "student_nodistill": "Student float (không distill, Keras)",
    "student_float_tflite": "Student float TFLite",
    "student_int8": "Student int8 TFLite",
}


def _load_json(p: Path) -> dict | None:
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _load_yaml(p: Path) -> dict | None:
    try:
        import yaml

        with open(p, encoding="utf-8") as f:
            return yaml.safe_load(f)
    except (OSError, ValueError, ImportError):
        return None
    except Exception:  # noqa: BLE001 - yaml.YAMLError
        return None


def _f(v, fmt: str = "{:.3f}") -> str:
    if v is None:
        return NA
    try:
        return fmt.format(v)
    except (TypeError, ValueError):
        return str(v)


def _table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _get(d: dict | None, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


# ---------------------------------------------------------------- sections
def section_dataset(cfg: Config, summary: dict | None) -> str:
    s = ["## 1. Dữ liệu"]
    if not summary:
        return "\n\n".join(s + [f"`summary.json`: {NA} (chạy `build-dataset`)."])
    splits = [k for k in ("train", "val", "test") if k in summary.get("windows", {})]
    rows = []
    for c in cfg.classes:
        rows.append(
            [f"{c.key} ({c.label_vi})"]
            + [str(_get(summary, "windows", sp, c.key, default=NA)) for sp in splits]
        )
    rows.append(["**tổng**"] + [str(sum(summary["windows"][sp].values())) for sp in splits])
    s.append("Số cửa sổ 1 s (hop 0.5 s) sau khi bỏ cửa sổ im lặng và giới hạn background:")
    s.append(_table(["Lớp"] + splits, rows))
    if summary.get("clips"):
        s.append("Số clip gốc: " + ", ".join(f"{sp}={sum(v.values())}" for sp, v in summary["clips"].items()))
    return "\n\n".join(s)


def section_offline(cfg: Config, off: dict | None) -> str:
    s = ["## 2. Đánh giá offline (test split, mức cửa sổ)"]
    models = (off or {}).get("models", {})
    rows = []
    for name, title in MODEL_TITLES.items():
        m = models.get(name)
        if m:
            rows.append(
                [
                    title,
                    _f(m["accuracy"]),
                    _f(m["macro_f1"]),
                    _f(m.get("danger_recall_mean")),
                    _f(m.get("danger_recall_min")),
                    f"[png]({m['confusion_png']})" if m.get("confusion_png") else "",
                ]
            )
        else:
            why = _get(off, "missing", name, default="")
            rows.append([title, NA, NA, NA, NA, why])
    s.append(
        _table(
            ["Model", "Accuracy", "Macro-F1", "Recall DANGER (TB)", "Recall DANGER (min)", "Confusion"], rows
        )
    )
    drop = (off or {}).get("int8_drop")
    if drop:
        s.append(
            f"Sụt giảm float → int8 (so với `{drop['reference']}`): accuracy {drop['accuracy_pts']:+.2f} "
            f"điểm %, macro-F1 {drop['macro_f1_pts']:+.2f} điểm %, trùng top-1 {drop['top1_agreement']:.4f}."
        )
    q = models.get("student_int8")
    s.append("### Từng lớp – student int8")
    if q:
        rows = []
        for c in cfg.classes:
            pc = q["per_class"].get(c.key, {})
            rows.append(
                [
                    c.key,
                    c.label_vi,
                    Level(c.level).name,
                    _f(pc.get("precision")),
                    _f(pc.get("recall")),
                    _f(pc.get("f1")),
                    str(pc.get("support", NA)),
                ]
            )
        s.append(_table(["Lớp", "Nhãn", "Mức", "Precision", "Recall", "F1", "Support"], rows))
    else:
        s.append(f"Student int8: {NA}.")
    imgs = [f"![{n}]({m['confusion_png']})" for n, m in models.items() if m.get("confusion_png")]
    if imgs:
        s.append("### Confusion matrix (chuẩn hoá theo hàng)")
        s.append("\n\n".join(imgs))
    return "\n\n".join(s)


def section_streaming(cfg: Config, st: dict | None, sugg: dict | None, reports_dir: Path) -> str:
    s = ["## 3. Đánh giá streaming (kịch bản dài)"]
    if not st:
        return "\n\n".join(
            s + [f"`streaming_metrics.json`: {NA} (chạy `make-scenarios` rồi `eval-streaming`)."]
        )
    s.append(
        f"Model `{st.get('model')}`; {st.get('n_event_scenarios')} kịch bản có sự kiện "
        f"({st.get('n_gt_events')} sự kiện), {st.get('n_background_scenarios')} kịch bản chỉ có nền "
        f"({_f(st.get('background_hours'), '{:.2f}')} giờ). Quy tắc khớp: cùng lớp và "
        f"`onset ≤ t_alert ≤ offset + {st.get('tol_after_s', 2)} s`; mỗi sự kiện khớp tối đa 1 lần; "
        f"cảnh báo lặp lại (`repeat`) của lớp DANGER bị bỏ qua; cảnh báo không khớp = báo nhầm."
    )
    s.append(
        _table(
            ["Chỉ số", "Giá trị"],
            [
                ["Event recall (tổng)", _f(st.get("event_recall"))],
                [
                    "Recall DANGER (TB / min)",
                    f"{_f(st.get('danger_event_recall_mean'))} / {_f(st.get('danger_event_recall_min'))}",
                ],
                [
                    "Độ trễ median / p90 (s)",
                    f"{_f(st.get('latency_median_s'), '{:.2f}')} / {_f(st.get('latency_p90_s'), '{:.2f}')}",
                ],
                [
                    "Báo nhầm / giờ (chỉ nền)",
                    f"{_f(st.get('fa_per_hour_background'), '{:.2f}')} "
                    f"({st.get('background_false_alarms', NA)} lần)",
                ],
                [
                    "Báo nhầm / giờ (kịch bản có sự kiện, ngoài cửa sổ GT)",
                    _f(st.get("fa_per_hour_event_outside_gt"), "{:.2f}"),
                ],
                [
                    "Báo nhầm trong kịch bản sự kiện (trùng / nhầm lớp / ngoài GT)",
                    " / ".join(
                        str(_get(st, "event_scenarios_false_alarms", k, default=NA))
                        for k in ("duplicate", "confusion", "outside")
                    ),
                ],
                [
                    "Tỉ lệ cửa sổ bị energy gate (nền / sự kiện)",
                    f"{_f(_get(st, 'engine', 'gated_fraction_background'))} / "
                    f"{_f(_get(st, 'engine', 'gated_fraction_events'))}",
                ],
            ],
        )
    )
    rows = []
    for k, v in st.get("per_class", {}).items():
        rows.append(
            [
                k,
                str(v.get("n_gt")),
                str(v.get("tp")),
                _f(v.get("recall")),
                _f(v.get("latency_median_s"), "{:.2f}"),
                _f(v.get("latency_p90_s"), "{:.2f}"),
                str(v.get("fa_background")),
                str(v.get("fa_event_scenarios")),
            ]
        )
    s.append("### Theo lớp")
    s.append(
        _table(
            [
                "Lớp",
                "Số sự kiện",
                "TP",
                "Recall",
                "Trễ median (s)",
                "Trễ p90 (s)",
                "FA (nền)",
                "FA (kịch bản sự kiện)",
            ],
            rows,
        )
    )
    rows = [
        [f"{k} dB", str(v["n_gt"]), str(v["tp"]), _f(v["recall"])] for k, v in st.get("per_snr", {}).items()
    ]
    if rows:
        s.append("### Theo SNR")
        s.append(_table(["SNR", "Số sự kiện", "TP", "Recall"], rows))
    s.append("### Quét ngưỡng")
    if (reports_dir / "streaming_sweep.png").exists():
        s.append("![sweep](streaming_sweep.png)")
    else:
        s.append(f"Hình quét ngưỡng: {NA}.")
    if sugg and sugg.get("thresholds"):
        rows = [
            [k, _f(_get(sugg, "current", k), "{:.2f}"), _f(v, "{:.2f}")]
            for k, v in sugg["thresholds"].items()
        ]
        s.append(
            "Ngưỡng đề xuất (`suggested_thresholds.yaml`, chỉ để tham khảo: chọn trên chính kịch bản "
            f"test nên KHÔNG áp dụng; tối đa hoá recall với báo nhầm nền ≤ {TARGETS['fa_per_h']:g}/giờ). "
            "Config dùng kết quả `tune` trên kịch bản **val** (mục 3b):"
        )
        s.append(_table(["Lớp", "Hiện tại", "Đề xuất"], rows))
        if sugg.get("infeasible"):
            s.append(
                "⚠️ Không đạt được ngân sách báo nhầm chỉ bằng ngưỡng cho: "
                + ", ".join(f"`{k}`" for k in sugg["infeasible"])
                + " (ngưỡng cao nhất của lưới vẫn báo nhầm quá nhiều → cần cải thiện model/dữ liệu)."
            )
        exp = sugg.get("expected") or {}
        if exp:
            s.append(
                f"Kỳ vọng với ngưỡng đề xuất (trên chính các kịch bản này, lạc quan): recall "
                f"{_f(exp.get('event_recall'))}, báo nhầm nền "
                f"{_f(exp.get('fa_per_hour_background'), '{:.2f}')}/giờ, "
                f"trễ median {_f(exp.get('latency_median_s'), '{:.2f}')} s."
            )
    else:
        s.append(f"Ngưỡng đề xuất: {NA}.")
    return "\n\n".join(s)


def section_benchmark(bench: dict | None) -> str:
    s = ["## 4. Benchmark"]
    if not bench:
        return "\n\n".join(s + [f"`benchmark.json`: {NA} (chạy `benchmark`)."])
    rows = []
    for name, m in bench.get("models", {}).items():
        rows.append(
            [
                name,
                m.get("format", ""),
                _f(m.get("size_bytes", 0) / 1024, "{:.1f}"),
                str(m.get("params")),
                _f(m.get("macs", 0) / 1e6, "{:.2f}"),
                _f(m["arena_estimate_bytes"] / 1024, "{:.1f}") if m.get("arena_estimate_bytes") else "–",
                _f(_get(m, "infer_ms", "median")),
                _f(_get(m, "infer_ms", "p95")),
            ]
        )
    fm = bench.get("feature_ms", {})
    s.append(
        f"PC, 1 thread, {bench.get('runs')} lần. Feature (log-mel + chuẩn hoá): median "
        f"{_f(fm.get('median'))} ms, p95 {_f(fm.get('p95'))} ms."
    )
    s.append(
        _table(
            [
                "Model",
                "Định dạng",
                "Kích thước (KB)",
                "Tham số",
                "MACs (M)",
                "Arena ước tính (KB)",
                "Suy luận median (ms)",
                "p95 (ms)",
            ],
            rows,
        )
    )
    g = bench.get("gating", {})
    s.append(
        f"Tỉ lệ cửa sổ bị energy gate bỏ qua trên kịch bản chỉ có nền: {_f(g.get('gated_fraction'))} "
        f"(nguồn: {g.get('source', NA)})."
    )
    s.append("### Ước tính độ trễ trên MCU (ƯỚC TÍNH, không phải đo thật)")
    q = bench.get("models", {}).get("student_int8") or next(iter(bench.get("models", {}).values()), None)
    if q and q.get("mcu_estimates"):
        rows = [
            [
                e["target"],
                f"{e['mac_per_cycle'][0]}–{e['mac_per_cycle'][1]}",
                f"{e['latency_ms_best']:.1f}–{e['latency_ms_worst']:.1f}",
                f"{100 * e['duty_cycle_worst']:.1f}%",
                e.get("note", ""),
            ]
            for e in q["mcu_estimates"]
        ]
        s.append(
            _table(
                ["MCU", "MAC/cycle giả định", "Suy luận (ms)", "CPU/hop 0.5 s (xấu nhất)", "Giả định"], rows
            )
        )
        s.append(
            "Công thức: `latency = MACs / (MAC_per_cycle × f_clk)`; chưa tính feature, overhead "
            "framework, truy cập bộ nhớ flash/PSRAM."
        )
    else:
        s.append(f"Ước tính MCU: {NA}.")
    return "\n\n".join(s)


def section_ui(reports_dir: Path) -> str:
    s = ["## 5. Giao diện mô phỏng"]
    imgs = sorted(reports_dir.glob("ui_*.png"))
    if imgs:
        s.append("\n\n".join(f"![{p.stem}]({p.name})" for p in imgs))
    else:
        s.append(f"Ảnh chụp UI (`ui_*.png`): {NA}.")
    return "\n\n".join(s)


def _check(ok: bool | None) -> str:
    return "⬜" if ok is None else ("✅" if ok else "❌")


def targets_checklist(
    off: dict | None, st: dict | None, bench: dict | None, quant: dict | None, ui_fps: float | None = None
) -> list[list[str]]:
    q = _get(off, "models", "student_int8")
    bq = _get(bench, "models", "student_int8")
    rows = []

    def add(name: str, target: str, value, ok, fmt="{:.3f}"):
        rows.append([name, target, _f(value, fmt), _check(None if value is None else ok)])

    v = q.get("macro_f1") if q else _get(quant, "test", "tflite_int8", "macro_f1")
    add(
        "Macro-F1 test (student int8)",
        f"≥ {TARGETS['macro_f1']}",
        v,
        v is not None and v >= TARGETS["macro_f1"],
    )
    v = q.get("danger_recall_min") if q else _get(quant, "test", "tflite_int8", "danger_recall_min")
    add(
        "Recall lớp DANGER (min, student int8)",
        f"≥ {TARGETS['danger_recall']}",
        v,
        v is not None and v >= TARGETS["danger_recall"],
    )
    v = _get(off, "int8_drop", "accuracy_pts")
    if v is None:
        v = _get(quant, "test", "int8_drop_accuracy_pts")
    add(
        "Sụt accuracy float → int8 (điểm %)",
        f"≤ {TARGETS['int8_drop_pts']}",
        v,
        v is not None and v <= TARGETS["int8_drop_pts"],
        "{:+.2f}",
    )
    v = bq["size_bytes"] / 1024 if bq else (_get(quant, "size_bytes", "tflite_int8") or 0) / 1024 or None
    add(
        "Kích thước .tflite int8 (KB)",
        f"≤ {TARGETS['size_kb']:g} (nRF52840 cũ: ≤ {TARGETS_OLD['size_kb']:g})",
        v,
        v is not None and v <= TARGETS["size_kb"],
        "{:.1f}",
    )
    v = bq.get("arena_estimate_bytes") if bq else _get(quant, "int8", "arena_estimate_bytes")
    v = v / 1024 if v else None
    add(
        "Tensor arena ước tính (KB)",
        f"≤ {TARGETS['arena_kb']:g} (nRF52840 cũ: ≤ {TARGETS_OLD['arena_kb']:g})",
        v,
        v is not None and v <= TARGETS["arena_kb"],
        "{:.1f}",
    )
    v = bq.get("macs") if bq else _get(quant, "macs")
    v = v / 1e6 if v else None
    add(
        "MACs / lần suy luận (M)",
        f"≤ {TARGETS['macs_m']:g} (nRF52840 cũ: ≤ {TARGETS_OLD['macs_m']:g})",
        v,
        v is not None and v <= TARGETS["macs_m"],
        "{:.2f}",
    )
    v = _get(st, "fa_per_hour_background")
    add(
        "Báo nhầm / giờ (chỉ nền)",
        f"≤ {TARGETS['fa_per_h']:g}",
        v,
        v is not None and v <= TARGETS["fa_per_h"],
        "{:.2f}",
    )
    v = _get(st, "latency_median_s")
    add(
        "Độ trễ median onset → cảnh báo (s)",
        f"≤ {TARGETS['latency_s']}",
        v,
        v is not None and v <= TARGETS["latency_s"],
        "{:.2f}",
    )
    v = bq.get("total_ms_median") if bq else None
    add(
        "Feature + suy luận trên PC (ms, median)",
        f"≤ {TARGETS['pc_ms']:g}",
        v,
        v is not None and v <= TARGETS["pc_ms"],
        "{:.2f}",
    )
    ui = ui_fps if isinstance(ui_fps, dict) else ({"fps": ui_fps} if ui_fps is not None else None)
    v = ui.get("fps") if ui else None
    cap = ui.get("fps_capacity") if ui else None
    # the loop is capped at cfg.ui.fps (= target) by clock.tick, so allow 1 fps of timer jitter; when
    # the uncapped render capacity is known it must reach the target too
    ok = v is not None and v >= TARGETS["ui_fps"] - 1.0 and (cap is None or cap >= TARGETS["ui_fps"])
    rows.append(
        [
            "UI FPS (khung hình thực / khả năng không giới hạn)",
            f"≥ {TARGETS['ui_fps']:g}",
            NA if v is None else (f"{v:.1f}" + (f" / {cap:.0f}" if cap is not None else "")),
            _check(None if v is None else ok),
        ]
    )
    return rows


def _ui_fps(reports_dir: Path) -> dict | None:
    for name in ("ui_stats.json", "ui_metrics.json"):
        d = _load_json(reports_dir / name)
        if d and isinstance(d.get("fps"), (int, float)):
            return d
    return None


def measure_ui_fps(cfg: Config, reports_dir: Path, seconds: float = 3.0) -> dict:
    """Run the UI demo (fake events, full debug panel) headless for `seconds` -> ui_stats.json."""
    import os

    if not os.environ.get("SDL_VIDEODRIVER"):  # unset or empty (the Makefile exports it empty)
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    from soundwatch.ui.app import run_demo, write_ui_stats

    st = run_demo(cfg, seconds=seconds, show_debug=True)
    write_ui_stats(
        reports_dir / "ui_stats.json", st, cfg.ui.fps, source=f"report --measure-ui-fps ({seconds:g} s)"
    )
    return st


def section_tuning(cfg: Config, tun: dict | None) -> str:
    s = ["## 3b. Tinh chỉnh hậu xử lý trên kịch bản val (WP8)"]
    if not tun:
        return "\n\n".join(s + [f"`tuning.json`: {NA} (chạy `make-scenarios --split val` rồi `tune`)."])
    s.append(
        f"`python -m soundwatch tune` trên `{tun.get('scenarios')}` ({tun.get('n_event_scenarios')} kịch bản "
        f"sự kiện, {_f(tun.get('background_hours'), '{:.1f}')} giờ chỉ có nền; "
        "**không** dùng kịch bản test). "
        "Lưới smooth_n {1,2,3} × k_of_n {1/1, 2/2, 2/3, 3/4} × bg_margin {tắt, 0.2, 0.4} × ngưỡng từng lớp "
        "(tham lam, DANGER ×2); ràng buộc trễ median ≤ "
        f"{tun.get('max_latency_s')} s. Bảng đầy đủ + Pareto: [tuning.md](tuning.md)."
    )
    rows = []
    for b, p in (tun.get("best") or {}).items():
        if not p:
            rows.append([f"≤ {b}", "không có điểm khả thi", "", "", "", ""])
            continue
        rows.append(
            [
                f"≤ {b}",
                p["setting"],
                _f(p.get("weighted_recall")),
                _f(p.get("recall")),
                _f(p.get("fa_per_hour_background"), "{:.1f}"),
                f"{_f(p.get('latency_median_s'), '{:.2f}')} / {_f(p.get('latency_p90_s'), '{:.2f}')}",
            ]
        )
    s.append(
        _table(
            ["Ngân sách FA/h", "Hậu xử lý tốt nhất", "Recall (trọng số)", "Recall", "FA/h (val)", "Trễ (s)"],
            rows,
        )
    )
    pp = cfg.postprocess
    s.append(
        f"**Đang dùng trong config:** smooth_n={pp.smooth_n}, k_of_n={pp.k_of_n[0]}/{pp.k_of_n[1]}, "
        f"bg_margin={'tắt' if pp.bg_margin is None else pp.bg_margin}; ngưỡng: "
        + ", ".join(f"{c.key}={c.threshold:.2f}" for c in cfg.classes[:-1])
        + ". Kết quả trên test ở mục 3."
    )
    return "\n\n".join(s)


def section_fa_analysis(fa: dict | None, fa_old: dict | None) -> str:
    s = ["## 3c. Phân tích báo nhầm (kịch bản chỉ có nền, test)"]
    if not fa:
        return "\n\n".join(s + [f"`fa_analysis.json`: {NA} (chạy `fa-analysis`)."])
    from soundwatch.eval.fa_analysis import summarize, summary_lines

    s.append("Chi tiết từng báo nhầm: [fa_analysis.md](fa_analysis.md).")
    s.append("\n".join(summary_lines(fa)))
    n_tot, k_tot = fa["n_analysed"], fa["n_label_noise_likely"]
    if fa_old:
        s.append(
            f"Với hậu xử lý cũ của WP7 (nhiều báo nhầm hơn): {fa_old['n_label_noise_likely']}/"
            f"{fa_old['n_analysed']} báo nhầm nghi do nhiễu nhãn (`fa_analysis_wp7config.json`)."
        )
        n_tot += fa_old["n_analysed"]
        k_tot += fa_old["n_label_noise_likely"]
    frac = k_tot / n_tot if n_tot else 0.0
    top = ", ".join(src for src, _ in summarize(fa)["top_sources"][:4])
    if frac < 0.3:
        s.append(
            f'**Kết luận:** giả thuyết "nền UrbanSound8K/ESC-50 thật ra chứa còi/chó" chỉ giải thích '
            f"~{100 * frac:.0f}% báo nhầm (YAMNet hầu như không nghe thấy lớp bị báo, "
            "và không bản ghi nền nào "
            "có slice cùng lớp). Phần lớn là **lỗi của model**: nhầm âm có cao độ ổn định / hài âm "
            "(nhạc đường "
            "phố – violin, máy khoan, điều hoà, động cơ) thành `siren`, tiếng dế (bíp cao tần lặp) thành "
            "`smoke_alarm` (proxy `clock_alarm`), tiếng cười / ếch / chim thành `dog_bark`. Nguồn chính: "
            f"{top}. Các loại nền này đã có trong background của train nhưng model vẫn nhầm → hướng xử lý: "
            "hard-negative mining (oversample / tăng trọng số các cửa sổ nền bị nhầm), mixup nền–sự kiện, "
            "hiệu chỉnh xác suất (temperature) vì model int8 bão hoà ở 0.996."
        )
    else:
        s.append(
            f"**Kết luận:** ~{100 * frac:.0f}% báo nhầm có dấu hiệu nhiễu nhãn trong dữ liệu nền "
            "(YAMNet cũng "
            "nghe thấy lớp bị báo) → cần lọc dữ liệu nền; phần còn lại là lỗi model."
        )
    return "\n\n".join(s)


def section_demo(reports_dir: Path) -> str:
    if not (reports_dir / "demo.gif").exists():
        return ""
    return (
        "## 5b. Demo (kịch bản `assets/demo/demo_scenario.wav`)\n\n"
        "`python -m soundwatch run --source assets/demo/demo_scenario.wav --record-gif "
        "artifacts/reports/demo.gif` (5 fps, ½ kích thước):\n\n![demo](demo.gif)"
    )


def build_report(
    cfg: Config,
    reports_dir: Path | None = None,
    models_dir: Path | None = None,
    windows_dir: Path | None = None,
) -> str:
    reports_dir = Path(reports_dir) if reports_dir else paths.REPORTS
    models_dir = Path(models_dir) if models_dir else paths.MODELS
    windows_dir = Path(windows_dir) if windows_dir else paths.WINDOWS
    summary = _load_json(windows_dir / "summary.json")
    off = _load_json(reports_dir / "offline_metrics.json")
    st = _load_json(reports_dir / "streaming_metrics.json")
    bench = _load_json(reports_dir / "benchmark.json")
    sugg = _load_yaml(reports_dir / "suggested_thresholds.yaml")
    quant = _load_json(models_dir / "quant_metrics.json")
    tun = _load_json(reports_dir / "tuning.json")
    fa = _load_json(reports_dir / "fa_analysis.json")
    fa_old = _load_json(reports_dir / "fa_analysis_wp7config.json")
    checklist = targets_checklist(off, st, bench, quant, _ui_fps(reports_dir))
    n_ok = sum(1 for r in checklist if r[3] == "✅")
    n_known = sum(1 for r in checklist if r[3] != "⬜")
    parts = [
        "# SoundWatch PoC – Báo cáo đánh giá (WP7–WP8)",
        "Sinh tự động bởi `python -m soundwatch report`. ✅ đạt, ❌ chưa đạt, ⬜ chưa có số liệu.",
        f"## 0. Đối chiếu mục tiêu (mục 2 kế hoạch, phần cứng ESP32-S3 + PSRAM): {n_ok}/{n_known} đạt",
        _table(["Chỉ số", "Mục tiêu", "Đo được", "Đạt?"], checklist),
        section_dataset(cfg, summary),
        section_offline(cfg, off),
        section_streaming(cfg, st, sugg, reports_dir),
        section_tuning(cfg, tun),
        section_fa_analysis(fa, fa_old),
        section_benchmark(bench),
        section_ui(reports_dir),
        section_demo(reports_dir),
        "## 6. Ghi chú\n\n"
        "- `smoke_alarm` dùng `clock_alarm` (ESC-50) làm proxy khi chưa có dữ liệu báo cháy thật (mục 10).\n"
        "- Kịch bản streaming được tổng hợp từ test split (nền + sự kiện trộn theo SNR), "
        "không phải ghi âm thật; "
        "số báo nhầm/giờ dựa trên ~1 giờ âm nền nên sai số thống kê lớn.\n"
        "- Hậu xử lý + ngưỡng trong config được chọn trên kịch bản **val** (`tune`); số liệu mục 3 là "
        "trên test. Ngưỡng đề xuất của `eval-streaming` được chọn trên chính test → chỉ tham khảo, "
        "lạc quan.\n"
        "- Lệch miền sang micro thật chưa được đo (container không có micro).\n"
        "- Số liệu MCU là ước tính từ MACs, không phải đo trên phần cứng.",
    ]
    return "\n\n".join(parts) + "\n"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out", default=None, help="output path (default artifacts/reports/report.md)")
    parser.add_argument(
        "--measure-ui-fps",
        action="store_true",
        help="run the UI demo headless for 3 s first and write reports/ui_stats.json",
    )


def main(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    if getattr(args, "measure_ui_fps", False):
        st = measure_ui_fps(cfg, paths.REPORTS)
        print(f"[report] UI: {st['fps']:.1f} fps (capacity {st['fps_capacity']:.0f})")
    md = build_report(cfg)
    out = Path(args.out) if args.out else paths.REPORTS / "report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    print(f"[report] -> {out}")
    return 0
