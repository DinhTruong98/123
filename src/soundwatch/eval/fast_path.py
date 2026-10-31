"""Before/after evaluation of the DANGER fast path (postprocess.danger_fast_threshold).

Runs inference once per scenario (int8 model from config), then replays only the PostProcessor
with danger_fast_threshold = None (rule off = current behaviour) and with each candidate value,
so every variant sees exactly the same probabilities. Besides the usual streaming metrics it
reports recall / latency of *short* events (GT duration <= SHORT_S) and, for short-event
scenario sets (make-scenarios --event-lengths), a breakdown by excerpt length.

    python -m soundwatch eval-fast-path [--sets test,val,short_test,short_val] [--values 0.9921875,0.99609375]

Writes artifacts/reports/fast_path.json.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
from pathlib import Path

import numpy as np

from soundwatch import paths
from soundwatch.config import Config, load_config
from soundwatch.eval.streaming import (
    ScenarioRun,
    aggregate,
    list_scenarios,
    match_events,
    replay_all,
    run_scenarios,
)
from soundwatch.types import Level

SHORT_S = 1.5
SETS = {
    "test": paths.DATA / "scenarios",
    "val": paths.DATA / "scenarios_val",
    "short_test": paths.DATA / "scenarios_short_test",
    "short_val": paths.DATA / "scenarios_short_val",
    "vn_test": paths.DATA / "scenarios_vn_test",
    "vn_val": paths.DATA / "scenarios_vn_val",
}
_LEN_RE = re.compile(r"@([0-9.]+)s\+")


def with_fast(cfg: Config, value: float | None) -> Config:
    post = dataclasses.replace(cfg.postprocess, danger_fast_threshold=value)
    return dataclasses.replace(cfg, postprocess=post)


def _pct(v: list[float], q: float) -> float | None:
    return float(np.percentile(np.asarray(v, float), q)) if v else None


def short_metrics(cfg: Config, runs: list[ScenarioRun], short_s: float = SHORT_S) -> dict:
    """Recall / latency of GT events with offset - onset <= short_s: overall, DANGER, per class."""
    keys = cfg.class_keys
    danger = {c.id for c in cfg.classes[:-1] if c.level == Level.DANGER}
    groups: dict[str, list[int]] = {}  # name -> [n_gt, tp]
    lats: dict[str, list[float]] = {}

    def add(name: str, hit: bool, lat: float | None) -> None:
        g = groups.setdefault(name, [0, 0])
        g[0] += 1
        if hit:
            g[1] += 1
            lats.setdefault(name, []).append(lat)

    for r in runs:
        if r.is_background:
            continue
        m = match_events(r.alerts, r.gt)
        hit = {i: lat for i, _, lat in m.tp}
        for i, e in enumerate(r.gt):
            if e.offset - e.onset > short_s + 1e-6:
                continue
            h, lat = i in hit, hit.get(i)
            add("all", h, lat)
            if e.class_id in danger:
                add("danger", h, lat)
            add(keys[e.class_id], h, lat)
    return {
        k: {
            "n_gt": n,
            "tp": tp,
            "recall": tp / n if n else None,
            "latency_median_s": _pct(lats.get(k, []), 50),
            "latency_p90_s": _pct(lats.get(k, []), 90),
        }
        for k, (n, tp) in groups.items()
    }


def by_length(cfg: Config, runs: list[ScenarioRun], scen_dir: Path) -> dict:
    """Recall / latency per excerpt length for short-event scenario sets."""
    import csv

    out: dict[str, list] = {}
    for r in runs:
        if r.is_background:
            continue
        csvp = scen_dir / f"{r.name}.csv"
        rows = list(csv.DictReader(open(csvp, encoding="utf-8")))
        m = match_events(r.alerts, r.gt)
        hit = {i: lat for i, _, lat in m.tp}
        for i, row in enumerate(rows):
            mm = _LEN_RE.search(row.get("source_clip", ""))
            if not mm:
                continue
            for k in (f"{float(mm.group(1)):g}s", f"{float(mm.group(1)):g}s/{row['class_key']}"):
                g = out.setdefault(k, [0, 0, []])
                g[0] += 1
                if i in hit:
                    g[1] += 1
                    g[2].append(hit[i])
    return {
        k: {"n_gt": n, "tp": tp, "recall": tp / n if n else None, "latency_median_s": _pct(lat, 50)}
        for k, (n, tp, lat) in sorted(out.items())
    }


def summarize(cfg: Config, runs: list[ScenarioRun], scen_dir: Path) -> dict:
    agg = aggregate(runs, cfg, with_details=False)
    danger = [c.key for c in cfg.classes[:-1] if c.level == Level.DANGER]
    pc = agg["per_class"]
    return {
        "event_recall": agg["event_recall"],
        "danger_recall_mean": agg["danger_event_recall_mean"],
        "recall": {k: pc[k]["recall"] for k in danger},
        "latency_median_s": agg["latency_median_s"],
        "latency_p90_s": agg["latency_p90_s"],
        "latency_median_by_class": {k: pc[k]["latency_median_s"] for k in danger},
        "latency_p90_by_class": {k: pc[k]["latency_p90_s"] for k in danger},
        "fa_per_hour_background": agg["fa_per_hour_background"],
        "fa_background_by_class": {k: pc[k]["fa_background"] for k in pc},
        "background_hours": agg["background_hours"],
        "fa_per_hour_event_outside_gt": agg["fa_per_hour_event_outside_gt"],
        "event_fa_kinds": agg["event_scenarios_false_alarms"],
        "n_gt_events": agg["n_gt_events"],
        "short": short_metrics(cfg, runs),
        "by_length": by_length(cfg, runs, scen_dir),
    }


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sets", default=",".join(SETS), help=f"scenario sets among {list(SETS)}")
    parser.add_argument(
        "--values", default="0.9921875,0.99609375", help="danger_fast_threshold candidates (comma-separated)"
    )
    parser.add_argument("--out-dir", default=None, help="report dir (default artifacts/reports)")
    parser.add_argument("--model", default=None, help="int8 .tflite to evaluate (default: config model)")
    parser.add_argument("--stats", default=None, help="feature_stats.json for --model (default: config)")


def main(args: argparse.Namespace) -> int:
    from soundwatch.audio.features import load_stats
    from soundwatch.engine.classifier import load_classifier

    cfg = load_config(args.config)
    base = with_fast(cfg, None)
    values = [float(v) for v in args.values.split(",") if v.strip()]
    model = Path(args.model) if args.model else cfg.model.resolved_tflite()
    clf = load_classifier(model)
    stats = load_stats(Path(args.stats) if args.stats else cfg.model.resolved_stats())
    out_dir = Path(args.out_dir) if args.out_dir else paths.REPORTS
    out_dir.mkdir(parents=True, exist_ok=True)
    result: dict = {"model": str(model), "values": values, "sets": {}}
    for name in args.sets.split(","):
        scen_dir = SETS[name]
        scen = list_scenarios(scen_dir)
        if not scen:
            print(f"[fast-path] {name}: no scenarios in {scen_dir}, skipped")
            continue
        runs = run_scenarios(base, scen, lambda gt: clf, stats, verbose=False)
        res = {"dir": str(scen_dir), "off": summarize(base, replay_all(base, runs, base.classes), scen_dir)}
        for v in values:
            c = with_fast(cfg, v)
            res[f"{v:.8f}"] = summarize(c, replay_all(c, runs, c.classes), scen_dir)
        result["sets"][name] = res
        print(f"[fast-path] {name}: " + "; ".join(
            f"{k}: recall {r['event_recall']:.3f} siren {r['recall'].get('siren')} "
            f"FA/h {r['fa_per_hour_background']} lat {r['latency_median_s']}"
            for k, r in res.items() if k != "dir"
        ), flush=True)
    text = json.dumps(result, indent=2, ensure_ascii=False)
    (out_dir / "fast_path.json").write_text(text, encoding="utf-8")
    print(f"[fast-path] -> {out_dir / 'fast_path.json'}")
    return 0
