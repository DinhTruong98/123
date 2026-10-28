"""WP7: event-level streaming evaluation on the synthesized scenarios (plan section 6 / WP7).

For every scenario the real engine (`run_offline`, `FileSource(realtime=False)`) is run once and
the alerts are matched against the ground-truth CSV.

Matching rules (`match_events`)
  * Only fresh alerts count (`repeat=True` DANGER re-emissions are ignored for both TP and FA).
  * A fresh alert of class c at `t = AlertEvent.t_audio` (end of the analysed window) can match a GT
    event e if `e.class == c` and `e.onset <= t <= e.offset + TOL_AFTER_S` (2 s).
  * Alerts are processed in time order; each GT event is matched at most once (the earliest
    eligible alert); among several eligible GT events the one with the earliest onset wins.
  * Every unmatched fresh alert is a false alarm. FAs are broken down into `duplicate` (inside the
    window of an already matched GT event of the same class), `confusion` (inside the window of a
    GT event of another class) and `outside` (outside every GT window).
  * Latency of a TP = t_alert - onset.

Metrics: event recall per class / overall / per SNR, latency median & p90 per class / overall,
false alarms per hour on background-only scenarios (all FAs / background hours) and on event
scenarios outside GT windows (outside-FAs / non-GT hours).

Threshold sweep (default on; `--no-sweep` to skip): per-hop probabilities are recorded once, then
only the PostProcessor is replayed (`replay_postprocess`) for a global threshold grid (all alert
classes), per-level grids (only classes of one level varied) and per-class grids. Suggested
thresholds: greedy - start every class at the highest grid value and repeatedly lower the threshold
of the class with the best (weighted recall gain) / (FA increase) while the background-only FA rate
stays <= FA_BUDGET_PER_H (2 / h); DANGER events weigh 2x. Ties (same TP, no more FA) resolve to the
grid value closest to the current config threshold. Written to suggested_thresholds.yaml
(config is NOT edited).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from soundwatch import paths
from soundwatch.config import Config, load_config
from soundwatch.types import AlertEvent, Level

TOL_AFTER_S = 2.0
FA_BUDGET_PER_H = 2.0
SWEEP_GRID = tuple(round(float(v), 2) for v in np.arange(0.30, 0.951, 0.05))
DANGER_WEIGHT = 2.0


@dataclass(frozen=True)
class GTEvent:
    class_id: int
    onset: float
    offset: float
    snr_db: float = float("nan")
    class_key: str = ""


@dataclass
class MatchResult:
    tp: list[tuple[int, AlertEvent, float]]  # (gt index, alert, latency)
    fa: list[tuple[AlertEvent, str]]  # (alert, kind: duplicate|confusion|outside)
    missed: list[int]  # gt indices
    n_repeats: int

    @property
    def n_fa(self) -> int:
        return len(self.fa)

    def fa_kind(self, kind: str) -> int:
        return sum(1 for _, k in self.fa if k == kind)


def load_gt(csv_path: Path) -> list[GTEvent]:
    from soundwatch.data.scenarios import read_csv

    return [
        GTEvent(r["class_id"], r["onset_s"], r["offset_s"], r["snr_db"], r["class_key"])
        for r in read_csv(Path(csv_path))
    ]


def match_events(alerts: list[AlertEvent], gt: list[GTEvent], tol_after: float = TOL_AFTER_S) -> MatchResult:
    fresh = sorted((a for a in alerts if not a.repeat), key=lambda a: a.t_audio)
    n_rep = sum(1 for a in alerts if a.repeat)
    matched: set[int] = set()
    tp: list[tuple[int, AlertEvent, float]] = []
    fa: list[tuple[AlertEvent, str]] = []
    for a in fresh:
        t = a.t_audio
        inside = [i for i, e in enumerate(gt) if e.onset <= t <= e.offset + tol_after]
        cand = [i for i in inside if gt[i].class_id == a.class_id and i not in matched]
        if cand:
            i = min(cand, key=lambda j: (gt[j].onset, j))
            matched.add(i)
            tp.append((i, a, t - gt[i].onset))
        elif any(gt[i].class_id == a.class_id for i in inside):
            fa.append((a, "duplicate"))
        elif inside:
            fa.append((a, "confusion"))
        else:
            fa.append((a, "outside"))
    missed = [i for i in range(len(gt)) if i not in matched]
    return MatchResult(tp, fa, missed, n_rep)


def gt_covered_seconds(gt: list[GTEvent], duration: float, tol_after: float = TOL_AFTER_S) -> float:
    """Total length of the union of GT windows [onset, offset + tol] clipped to [0, duration]."""
    iv = sorted((max(0.0, e.onset), min(duration, e.offset + tol_after)) for e in gt)
    tot, cur_s, cur_e = 0.0, None, None
    for s, e in iv:
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                tot += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        tot += cur_e - cur_s
    return tot


def _pct(v: list[float], q: float) -> float | None:
    return float(np.percentile(np.asarray(v, float), q)) if v else None


# ---------------------------------------------------------------- aggregation
@dataclass
class ScenarioRun:
    name: str
    duration_s: float
    gt: list[GTEvent]
    alerts: list[AlertEvent]
    summary: object | None = None  # StatusSummary
    engine_stats: dict | None = None

    @property
    def is_background(self) -> bool:
        return not self.gt and self.name.startswith("bg_")


def aggregate(
    runs: list[ScenarioRun], cfg: Config, tol_after: float = TOL_AFTER_S, with_details: bool = True
) -> dict:
    keys = cfg.class_keys
    alert_keys = keys[:-1]
    n_gt = {k: 0 for k in alert_keys}
    n_tp = {k: 0 for k in alert_keys}
    lat: dict[str, list[float]] = {k: [] for k in alert_keys}
    snr_gt: dict[str, int] = {}
    snr_tp: dict[str, int] = {}
    bg_s = bg_fa = 0.0
    bg_fa_cls = {k: 0 for k in alert_keys}
    ev_out_s = ev_out_fa = 0.0
    ev_fa_kinds = {"duplicate": 0, "confusion": 0, "outside": 0}
    ev_fa_cls = {k: 0 for k in alert_keys}
    repeats = 0
    details = []
    for r in runs:
        m = match_events(r.alerts, r.gt, tol_after)
        repeats += m.n_repeats
        if r.is_background:
            bg_s += r.duration_s
            bg_fa += m.n_fa
            for a, _ in m.fa:
                bg_fa_cls[keys[a.class_id]] += 1
        else:
            for e in r.gt:
                k = keys[e.class_id]
                n_gt[k] += 1
                sk = f"{e.snr_db:g}"
                snr_gt[sk] = snr_gt.get(sk, 0) + 1
            for i, a, latency in m.tp:
                k = keys[a.class_id]
                n_tp[k] += 1
                lat[k].append(latency)
                sk = f"{r.gt[i].snr_db:g}"
                snr_tp[sk] = snr_tp.get(sk, 0) + 1
            ev_out_s += r.duration_s - gt_covered_seconds(r.gt, r.duration_s, tol_after)
            for a, kind in m.fa:
                ev_fa_kinds[kind] += 1
                ev_fa_cls[keys[a.class_id]] += 1
            ev_out_fa += m.fa_kind("outside")
        if with_details:
            d = {
                "name": r.name,
                "duration_s": r.duration_s,
                "n_gt": len(r.gt),
                "tp": len(m.tp),
                "fa": m.n_fa,
                "repeats": m.n_repeats,
                "alerts": [
                    {
                        "class_key": keys[a.class_id],
                        "t": round(a.t_audio, 3),
                        "conf": round(a.confidence, 3),
                        "repeat": a.repeat,
                    }
                    for a in r.alerts
                ],
                "fa_alerts": [
                    {"class_key": keys[a.class_id], "t": round(a.t_audio, 3), "kind": k} for a, k in m.fa
                ],
                "missed": [
                    {"class_key": keys[r.gt[i].class_id], "onset_s": r.gt[i].onset, "snr_db": r.gt[i].snr_db}
                    for i in m.missed
                ],
            }
            if r.engine_stats:
                d["gated_fraction"] = r.engine_stats.get("gated_fraction")
            details.append(d)
    tot_gt, tot_tp = sum(n_gt.values()), sum(n_tp.values())
    all_lat = [v for k in alert_keys for v in lat[k]]
    danger = [c.key for c in cfg.classes[:-1] if c.level == Level.DANGER]
    per_class = {
        k: {
            "n_gt": n_gt[k],
            "tp": n_tp[k],
            "recall": (n_tp[k] / n_gt[k]) if n_gt[k] else None,
            "latency_median_s": _pct(lat[k], 50),
            "latency_p90_s": _pct(lat[k], 90),
            "fa_background": bg_fa_cls[k],
            "fa_event_scenarios": ev_fa_cls[k],
        }
        for k in alert_keys
    }
    dr = [per_class[k]["recall"] for k in danger if per_class[k]["recall"] is not None]
    out = {
        "n_event_scenarios": sum(1 for r in runs if not r.is_background),
        "n_background_scenarios": sum(1 for r in runs if r.is_background),
        "n_gt_events": tot_gt,
        "n_tp": tot_tp,
        "event_recall": tot_tp / tot_gt if tot_gt else None,
        "danger_event_recall_mean": float(np.mean(dr)) if dr else None,
        "danger_event_recall_min": float(np.min(dr)) if dr else None,
        "per_class": per_class,
        "per_snr": {
            s: {"n_gt": snr_gt[s], "tp": snr_tp.get(s, 0), "recall": snr_tp.get(s, 0) / snr_gt[s]}
            for s in sorted(snr_gt, key=float)
        },
        "latency_median_s": _pct(all_lat, 50),
        "latency_p90_s": _pct(all_lat, 90),
        "background_hours": bg_s / 3600.0,
        "background_false_alarms": int(bg_fa),
        "fa_per_hour_background": (bg_fa / (bg_s / 3600.0)) if bg_s > 0 else None,
        "event_scenarios_non_gt_hours": ev_out_s / 3600.0,
        "event_scenarios_false_alarms": ev_fa_kinds,
        "fa_per_hour_event_outside_gt": (ev_out_fa / (ev_out_s / 3600.0)) if ev_out_s > 0 else None,
        "repeat_alerts_ignored": repeats,
    }
    if with_details:
        out["scenarios"] = details
    return out


# ---------------------------------------------------------------- sweep
def with_thresholds(cfg: Config, thr: dict[int, float]) -> tuple:
    return tuple(
        dataclasses.replace(c, threshold=float(thr[c.id])) if c.id in thr else c for c in cfg.classes
    )


def replay_all(cfg: Config, runs: list[ScenarioRun], classes) -> list[ScenarioRun]:
    from soundwatch.engine.pipeline import replay_postprocess

    return [
        ScenarioRun(r.name, r.duration_s, r.gt, replay_postprocess(cfg, r.summary, classes)) for r in runs
    ]


def sweep(cfg: Config, runs: list[ScenarioRun], grid=SWEEP_GRID) -> dict:
    """Recall vs FA/h for global, per-level and per-class threshold grids."""
    alert = list(cfg.classes[:-1])

    def point(thr: dict[int, float]) -> dict:
        a = aggregate(replay_all(cfg, runs, with_thresholds(cfg, thr)), cfg, with_details=False)
        return {
            "recall": a["event_recall"],
            "danger_recall_min": a["danger_event_recall_min"],
            "fa_per_hour_background": a["fa_per_hour_background"],
            "fa_per_hour_event_outside_gt": a["fa_per_hour_event_outside_gt"],
            "latency_median_s": a["latency_median_s"],
            "per_class": {
                k: {
                    "tp": v["tp"],
                    "n_gt": v["n_gt"],
                    "recall": v["recall"],
                    "fa_background": v["fa_background"],
                }
                for k, v in a["per_class"].items()
            },
        }

    res: dict = {"grid": list(grid), "global": [], "per_level": {}, "per_class": {}}
    for t in grid:
        res["global"].append({"threshold": t, **point({c.id: t for c in alert})})
    for lvl in sorted({int(c.level) for c in alert}):
        name = Level(lvl).name.lower()
        res["per_level"][name] = [
            {"threshold": t, **point({c.id: t for c in alert if int(c.level) == lvl})} for t in grid
        ]
    # Classes are independent in the PostProcessor (per-class smoothing / k-of-n / cooldown) and
    # TP / background-FA counts of class c depend only on alerts of class c, so the per-class
    # tables (only c varied) can be read off the global sweep.
    for c in alert:
        res["per_class"][c.key] = [
            {"threshold": g["threshold"], **g["per_class"][c.key]} for g in res["global"]
        ]
    return res


def suggest_thresholds(
    cfg: Config,
    sw: dict,
    bg_hours: float,
    budget_per_h: float = FA_BUDGET_PER_H,
    danger_weight: float = DANGER_WEIGHT,
) -> dict:
    """Greedy per-class threshold choice under a total background FA budget (see module doc)."""
    grid = sw["grid"]
    budget = budget_per_h * bg_hours
    keys = [c.key for c in cfg.classes[:-1]]
    w = {c.key: (danger_weight if c.level == Level.DANGER else 1.0) for c in cfg.classes[:-1]}
    tab = sw["per_class"]
    idx = {k: len(grid) - 1 for k in keys}  # start at highest threshold

    # classes whose highest threshold alone exceeds the budget stay at the top (flagged infeasible)
    # and are left out of the budget so the remaining classes can still be tuned
    infeasible = {k for k in keys if tab[k][-1]["fa_background"] > budget + 1e-9}

    def fa_tot() -> float:
        return sum(tab[k][idx[k]]["fa_background"] for k in keys if k not in infeasible)

    while True:
        best, best_score = None, None
        for k in keys:
            if idx[k] == 0 or k in infeasible:
                continue
            cur = tab[k][idx[k]]
            # look ahead to the next grid point that actually gains recall
            j = idx[k] - 1
            while j > 0 and tab[k][j]["tp"] <= cur["tp"]:
                j -= 1
            nxt = tab[k][j]
            gain = (nxt["tp"] - cur["tp"]) * w[k]
            dfa = nxt["fa_background"] - cur["fa_background"]
            if gain <= 0 or fa_tot() + dfa > budget + 1e-9:
                continue
            score = gain / (dfa + 0.5)
            if best_score is None or score > best_score:
                best, best_score = (k, j), score
        if best is None:
            break
        idx[best[0]] = best[1]
    out = {}
    for c in cfg.classes[:-1]:
        # tie-break: among grid points with the same TP and no more FA, stay closest to the config
        ch = tab[c.key][idx[c.key]]
        same = [
            j
            for j, r in enumerate(tab[c.key])
            if r["tp"] == ch["tp"] and r["fa_background"] <= ch["fa_background"]
        ]
        idx[c.key] = min(same, key=lambda j: (abs(grid[j] - c.threshold), -grid[j]))
        row = tab[c.key][idx[c.key]]
        out[c.key] = {
            "threshold": float(grid[idx[c.key]]),
            "current": float(c.threshold),
            "recall": row["recall"],
            "fa_background": row["fa_background"],
            # even the highest grid threshold exceeds the whole FA budget for this class
            "feasible": c.key not in infeasible,
        }
    return out


def write_suggested_yaml(path: Path, sugg: dict, verify: dict | None, model: str, budget: float) -> None:
    import yaml

    doc = {
        "note": "Suggested per-class thresholds from eval-streaming --sweep (greedy: maximize event "
        f"recall s.t. background FA <= {budget:g}/h; DANGER weighted x{DANGER_WEIGHT:g}). "
        "NOT applied automatically - copy into config/soundwatch.yaml if desired.",
        "model": model,
        "thresholds": {k: round(v["threshold"], 2) for k, v in sugg.items()},
        "current": {k: v["current"] for k, v in sugg.items()},
    }
    bad = [k for k, v in sugg.items() if not v.get("feasible", True)]
    if bad:
        doc["infeasible"] = bad
        doc["note"] += (
            f" INFEASIBLE for {', '.join(bad)}: even threshold {max(SWEEP_GRID):g} gives more "
            "background false alarms than the budget - the model (not the threshold) must improve."
        )
    if verify:
        doc["expected"] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in verify.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, allow_unicode=True, sort_keys=False)


def plot_sweep(
    sw: dict, path: Path, sugg_point: dict | None = None, current_point: dict | None = None
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from soundwatch.eval.offline import setup_font

    setup_font()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ax = axes[0]
    curves = [("tất cả lớp", sw["global"])] + [(f"chỉ {k.upper()}", v) for k, v in sw["per_level"].items()]
    for label, rows in curves:
        xs = [r["fa_per_hour_background"] or 0.0 for r in rows]
        ys = [r["recall"] or 0.0 for r in rows]
        ax.plot(xs, ys, marker="o", ms=3, label=label)
        if label == "tất cả lớp":
            for r, x, y in zip(rows, xs, ys, strict=True):
                if int(round(r["threshold"] * 100)) % 10 == 0:
                    ax.annotate(
                        f"{r['threshold']:.1f}", (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points"
                    )
    if current_point:
        ax.plot([current_point["fa"]], [current_point["recall"]], "k*", ms=12, label="config hiện tại")
    if sugg_point:
        ax.plot([sugg_point["fa"]], [sugg_point["recall"]], "r*", ms=12, label="ngưỡng đề xuất")
    ax.axvline(FA_BUDGET_PER_H, color="gray", ls="--", lw=1)
    ax.set_xlabel("Báo nhầm / giờ (kịch bản chỉ có nền)")
    ax.set_ylabel("Event recall (tổng)")
    ax.set_title("Quét ngưỡng: recall vs báo nhầm")
    ax.set_xscale("symlog", linthresh=1.0)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax = axes[1]
    for k, rows in sw["per_class"].items():
        ax.plot([r["threshold"] for r in rows], [r["recall"] or 0.0 for r in rows], marker=".", label=k)
    ax.set_xlabel("Ngưỡng (chỉ lớp đó thay đổi)")
    ax.set_ylabel("Event recall của lớp")
    ax.set_title("Recall từng lớp theo ngưỡng")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------- runner
def list_scenarios(scen_dir: Path) -> list[tuple[str, Path, Path | None]]:
    out: list[tuple[str, Path, Path | None]] = []
    for w in sorted(scen_dir.glob("scenario_*.wav")):
        c = w.with_suffix(".csv")
        out.append((w.stem, w, c if c.exists() else None))
    for w in sorted(scen_dir.glob("bg_*.wav")):
        out.append((w.stem, w, None))
    return out


def run_scenarios(
    cfg: Config,
    scen: list[tuple[str, Path, Path | None]],
    make_classifier,
    stats: dict,
    classes=None,
    verbose: bool = True,
) -> list[ScenarioRun]:
    """make_classifier(gt) -> classifier (called per scenario; lets a Dummy oracle see the GT)."""
    from soundwatch.audio.sources import FileSource
    from soundwatch.engine.pipeline import run_offline

    runs = []
    for name, wav, csvp in scen:
        gt = load_gt(csvp) if csvp else []
        src = FileSource(wav, sample_rate=cfg.audio.sample_rate, realtime=False)
        dur = src.duration_s
        t0 = time.perf_counter()
        events, summary, st = run_offline(cfg, src, make_classifier(gt), stats, postprocessor_classes=classes)
        runs.append(ScenarioRun(name, dur, gt, events, summary, st))
        if verbose:
            print(
                f"[streaming] {name}: {len(events)} alerts, gated {st['gated_fraction']:.2f}, "
                f"{time.perf_counter() - t0:.1f}s",
                flush=True,
            )
    return runs


def oracle_dummy(cfg: Config, confidence: float = 0.9):
    """DummyClassifier factory predicting the GT class while the window overlaps an event by >= 0.5 s."""
    from soundwatch.engine.classifier import DummyClassifier

    def make(gt: list[GTEvent]):
        sched = [(e.onset + 0.5, e.offset + 1.0, e.class_id) for e in gt]
        return DummyClassifier(cfg.num_classes, sched, confidence)

    return make


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=None, help="model (.tflite / .keras); default config model")
    parser.add_argument("--stats", default=None, help="feature_stats.json (default from config)")
    parser.add_argument(
        "--dummy",
        action="store_true",
        help="use an oracle DummyClassifier (pipeline check); writes *_dummy outputs",
    )
    parser.add_argument(
        "--sweep", dest="sweep", action="store_true", default=True, help="threshold sweep (default on)"
    )
    parser.add_argument("--no-sweep", dest="sweep", action="store_false")
    parser.add_argument("--scenarios", default=None, help="scenario dir (default artifacts/data/scenarios)")
    parser.add_argument("--limit", type=int, default=None, help="only the first N event + N background files")
    parser.add_argument("--out-dir", default=None, help="report dir (default artifacts/reports)")


def main(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    scen_dir = Path(args.scenarios) if args.scenarios else paths.SCENARIOS
    scen = list_scenarios(scen_dir)
    if not scen:
        print(f"[streaming] no scenarios in {scen_dir}: run `python -m soundwatch make-scenarios` first")
        return 1
    if args.limit:
        ev = [s for s in scen if s[2] is not None][: args.limit]
        bg = [s for s in scen if s[2] is None][: args.limit]
        scen = ev + bg
    out_dir = Path(args.out_dir) if args.out_dir else paths.REPORTS
    suffix = ""
    if args.dummy:
        make, stats, model_name, suffix = (
            oracle_dummy(cfg),
            {"mean": 0.0, "std": 1.0},
            "dummy-oracle",
            "_dummy",
        )
    else:
        from soundwatch.audio.features import load_stats
        from soundwatch.engine.classifier import load_classifier

        mpath = Path(args.model) if args.model else cfg.model.resolved_tflite()
        spath = Path(args.stats) if args.stats else cfg.model.resolved_stats()
        if not mpath.exists() or not spath.exists():
            print(
                f"[streaming] model {mpath} or stats {spath} missing (train/quantize first, or use --dummy)"
            )
            return 1
        stats = load_stats(spath)
        clf = load_classifier(mpath, num_threads=1)
        make = lambda gt: clf  # noqa: E731 - engine resets nothing in the classifier
        model_name = str(mpath.relative_to(paths.ROOT)) if mpath.is_relative_to(paths.ROOT) else str(mpath)
    t0 = time.perf_counter()
    runs = run_scenarios(cfg, scen, make, stats)
    res = aggregate(runs, cfg)
    bg_stats = [r.engine_stats for r in runs if r.is_background and r.engine_stats]
    ev_stats = [r.engine_stats for r in runs if not r.is_background and r.engine_stats]

    def mean_of(lst, key):
        v = [s[key] for s in lst if s.get(key) is not None]
        return float(np.mean(v)) if v else None

    res = {
        "model": model_name,
        "tol_after_s": TOL_AFTER_S,
        "thresholds": {c.key: c.threshold for c in cfg.classes[:-1]},
        "postprocess": dataclasses.asdict(cfg.postprocess),
        "gate_dbfs": cfg.audio.gate_dbfs,
        "engine": {
            "gated_fraction_background": mean_of(bg_stats, "gated_fraction"),
            "gated_fraction_events": mean_of(ev_stats, "gated_fraction"),
            "infer_ms_mean": mean_of(bg_stats + ev_stats, "infer_ms_mean"),
            "feature_ms_mean": mean_of(bg_stats + ev_stats, "feature_ms_mean"),
            "realtime_factor_mean": mean_of(bg_stats + ev_stats, "realtime_factor"),
        },
        **res,
    }
    print(
        f"[streaming] recall {res['event_recall']}, FA/h bg {res['fa_per_hour_background']}, "
        f"latency median {res['latency_median_s']} ({time.perf_counter() - t0:.0f}s)"
    )
    if args.sweep and runs:
        sw = sweep(cfg, runs)
        bg_h = res["background_hours"]
        sugg = suggest_thresholds(cfg, sw, bg_h) if bg_h > 0 else None
        verify = None
        if sugg:
            thr = {cfg.class_keys.index(k): v["threshold"] for k, v in sugg.items()}
            v = aggregate(replay_all(cfg, runs, with_thresholds(cfg, thr)), cfg, with_details=False)
            verify = {
                "event_recall": v["event_recall"],
                "danger_event_recall_min": v["danger_event_recall_min"],
                "fa_per_hour_background": v["fa_per_hour_background"],
                "fa_per_hour_event_outside_gt": v["fa_per_hour_event_outside_gt"],
                "latency_median_s": v["latency_median_s"],
            }
            write_suggested_yaml(
                out_dir / f"suggested_thresholds{suffix}.yaml", sugg, verify, model_name, FA_BUDGET_PER_H
            )
        res["sweep"] = sw
        res["suggested_thresholds"] = sugg
        res["suggested_expected"] = verify
        cur = {"fa": res["fa_per_hour_background"] or 0.0, "recall": res["event_recall"] or 0.0}
        sp = (
            {"fa": verify["fa_per_hour_background"] or 0.0, "recall": verify["event_recall"] or 0.0}
            if verify
            else None
        )
        plot_sweep(sw, out_dir / f"streaming_sweep{suffix}.png", sp, cur)
        print(
            "[streaming] suggested thresholds: "
            + (", ".join(f"{k}={v['threshold']:.2f}" for k, v in sugg.items()) if sugg else "n/a")
        )
        if verify:
            print(f"[streaming] with suggested thresholds: {verify}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"streaming_metrics{suffix}.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"[streaming] -> {out_dir / f'streaming_metrics{suffix}.json'}")
    return 0
