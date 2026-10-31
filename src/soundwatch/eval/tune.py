"""WP8: offline tuning of the post-processing (smoothing, k-of-n, per-class thresholds, bg margin).

    python -m soundwatch make-scenarios --split val --out artifacts/data/scenarios_val --no-demo
    python -m soundwatch tune                      # -> artifacts/reports/tuning.{json,md}

Tuning runs on **val**-split scenarios (never on the test scenarios used for the final numbers).

1. The engine (`run_offline`, real int8 model) runs once per scenario; the per-hop raw probabilities
   (StatusSummary) are kept.
2. For every post-processing setting  smooth_n x k_of_n x bg_margin  (`SMOOTH_GRID`, `KOFN_GRID`,
   `MARGIN_GRID`) and every threshold of `THR_GRID`, only the PostProcessor is replayed
   (`replay_postprocess`) with that threshold for all alert classes. Classes are independent in the
   PostProcessor and in `match_events` (TP / FA of class c depend only on alerts of class c), so this
   gives per-class tables  threshold -> (TP, background FA, latencies)  from which any per-class
   threshold vector can be scored exactly without another replay.
3. Per setting and FA budget (`BUDGETS`, false alarms / hour on the background-only streams):
   greedy per-class thresholds (`greedy_thresholds`): start every class at the highest grid value,
   repeatedly lower the class with the best (weighted TP gain) / (FA increase + 0.5) while the
   *total* background FA count stays within budget. Objective = weighted event recall, DANGER
   events weigh `DANGER_WEIGHT` (2x). If even the highest thresholds exceed the budget, the setting
   is infeasible for that budget.
4. A (setting, budget) point is admissible if it is feasible and the median latency is
   <= `MAX_LATENCY_S`. The best admissible point per budget (max weighted recall; ties: lower
   latency, then fewer FA) is reported; the Pareto front over (weighted recall up, FA/h down,
   median latency down) of all points is listed as well.

The config is NOT edited; copy the chosen values into config/soundwatch.yaml.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from soundwatch import paths
from soundwatch.config import Config, load_config
from soundwatch.eval.streaming import (
    DANGER_WEIGHT,
    ScenarioRun,
    list_scenarios,
    match_events,
    run_scenarios,
    with_thresholds,
)
from soundwatch.types import Level

SMOOTH_GRID = (1, 2, 3)
KOFN_GRID = ((1, 1), (2, 2), (2, 3), (3, 4))
MARGIN_GRID: tuple[float | None, ...] = (None, 0.2, 0.4)
THR_GRID = tuple(round(float(v), 2) for v in np.arange(0.30, 0.951, 0.05)) + (0.97, 0.99)
BUDGETS = (2.0, 5.0, 10.0)
MAX_LATENCY_S = 1.5
DEFAULT_VAL_DIR = paths.DATA / "scenarios_val"


@dataclass(frozen=True)
class PPSetting:
    smooth_n: int
    k: int
    n: int
    bg_margin: float | None = None

    @property
    def label(self) -> str:
        m = "–" if self.bg_margin is None else f"{self.bg_margin:g}"
        return f"smooth={self.smooth_n} k/n={self.k}/{self.n} margin={m}"

    def apply(self, cfg: Config) -> Config:
        pp = dataclasses.replace(
            cfg.postprocess, smooth_n=self.smooth_n, k_of_n=(self.k, self.n), bg_margin=self.bg_margin
        )
        return dataclasses.replace(cfg, postprocess=pp)


def all_settings(smooth=SMOOTH_GRID, kofn=KOFN_GRID, margins=MARGIN_GRID) -> list[PPSetting]:
    return [PPSetting(s, k, n, m) for s in smooth for (k, n) in kofn for m in margins]


# ---------------------------------------------------------------- per-class tables
def class_tables(cfg: Config, runs: list[ScenarioRun], setting: PPSetting, grid=THR_GRID) -> dict:
    """{class_key: [row per grid threshold]}, row = {thr, tp, n_gt, fa_bg, fa_ev, lat: [s, ...]}.

    fa_bg: false alarms on background-only runs; fa_ev: 'outside' FAs in event runs."""
    from soundwatch.engine.pipeline import replay_postprocess

    c2 = setting.apply(cfg)
    alert = list(cfg.classes[:-1])
    keys = cfg.class_keys
    n_gt = {c.key: sum(1 for r in runs for e in r.gt if e.class_id == c.id) for c in alert}
    tabs: dict[str, list[dict]] = {c.key: [] for c in alert}
    for thr in grid:
        classes = with_thresholds(c2, {c.id: thr for c in alert})
        rows = {
            c.key: {"thr": thr, "tp": 0, "n_gt": n_gt[c.key], "fa_bg": 0, "fa_ev": 0, "lat": []}
            for c in alert
        }
        for r in runs:
            m = match_events(replay_postprocess(c2, r.summary, classes), r.gt)
            for _, a, lat in m.tp:
                row = rows[keys[a.class_id]]
                row["tp"] += 1
                row["lat"].append(round(float(lat), 4))
            for a, kind in m.fa:
                if r.is_background:
                    rows[keys[a.class_id]]["fa_bg"] += 1
                elif kind == "outside":
                    rows[keys[a.class_id]]["fa_ev"] += 1
        for k, row in rows.items():
            tabs[k].append(row)
    return tabs


def greedy_thresholds(tabs: dict, weights: dict, budget: float) -> tuple[dict[str, int], bool]:
    """Per-class grid indices maximizing sum(w * TP) s.t. sum(fa_bg) <= budget (counts).

    Returns (index per class, feasible). Infeasible = even the highest thresholds exceed the budget
    (then the highest thresholds are returned)."""
    keys = list(tabs)
    idx = {k: len(tabs[k]) - 1 for k in keys}

    def fa_tot() -> float:
        return float(sum(tabs[k][idx[k]]["fa_bg"] for k in keys))

    if fa_tot() > budget + 1e-9:
        return idx, False
    while True:
        best, best_score = None, -1.0
        cur_fa = fa_tot()
        for k in keys:
            cur = tabs[k][idx[k]]
            # next lower grid point that actually gains TP (look-ahead over flat stretches)
            j = idx[k] - 1
            while j >= 0 and tabs[k][j]["tp"] <= cur["tp"]:
                j -= 1
            if j < 0:
                continue
            nxt = tabs[k][j]
            gain = (nxt["tp"] - cur["tp"]) * weights[k]
            dfa = nxt["fa_bg"] - cur["fa_bg"]
            if cur_fa + dfa > budget + 1e-9:
                continue
            score = gain / (max(dfa, 0) + 0.5)
            if score > best_score:
                best, best_score = (k, j), score
        if best is None:
            return idx, True
        idx[best[0]] = best[1]


def score_choice(tabs: dict, idx: dict[str, int], weights: dict, bg_hours: float, ev_hours: float) -> dict:
    rows = {k: tabs[k][i] for k, i in idx.items()}
    tp = sum(r["tp"] for r in rows.values())
    ngt = sum(r["n_gt"] for r in rows.values())
    wtp = sum(r["tp"] * weights[k] for k, r in rows.items())
    wgt = sum(r["n_gt"] * weights[k] for k, r in rows.items())
    lat = [v for r in rows.values() for v in r["lat"]]
    fa = sum(r["fa_bg"] for r in rows.values())
    fa_ev = sum(r["fa_ev"] for r in rows.values())
    danger = [k for k, w in weights.items() if w > 1.0]
    dr = [rows[k]["tp"] / rows[k]["n_gt"] for k in danger if rows[k]["n_gt"]]
    return {
        "weighted_recall": wtp / wgt if wgt else None,
        "recall": tp / ngt if ngt else None,
        "danger_recall_mean": float(np.mean(dr)) if dr else None,
        "danger_recall_min": float(np.min(dr)) if dr else None,
        "fa_bg": int(fa),
        "fa_per_hour_background": fa / bg_hours if bg_hours > 0 else None,
        "fa_per_hour_event_outside_gt": fa_ev / ev_hours if ev_hours > 0 else None,
        "latency_median_s": float(np.median(lat)) if lat else None,
        "latency_p90_s": float(np.percentile(lat, 90)) if lat else None,
        "thresholds": {k: r["thr"] for k, r in rows.items()},
        "per_class_recall": {k: (r["tp"] / r["n_gt"] if r["n_gt"] else None) for k, r in rows.items()},
        "per_class_fa_bg": {k: r["fa_bg"] for k, r in rows.items()},
    }


def pareto_front(points: list[dict]) -> list[int]:
    """Indices of non-dominated points: maximize weighted_recall, minimize FA/h and latency."""

    def vec(p):
        return (
            -(p["weighted_recall"] or 0.0),
            p["fa_per_hour_background"] or 0.0,
            p["latency_median_s"] or 99.0,
        )

    vs = [vec(p) for p in points]
    front = []
    for i, a in enumerate(vs):
        dominated = any(
            all(b[j] <= a[j] for j in range(3)) and any(b[j] < a[j] for j in range(3))
            for k, b in enumerate(vs)
            if k != i
        )
        if not dominated:
            front.append(i)
    return front


def pick_best(points: list[dict], budget: float, max_latency: float) -> dict | None:
    ok = [
        p
        for p in points
        if p["budget_per_h"] == budget and p["feasible"] and (p["latency_median_s"] or 99.0) <= max_latency
    ]
    if not ok:
        return None
    return max(ok, key=lambda p: (p["weighted_recall"] or 0.0, -(p["latency_median_s"] or 99.0), -p["fa_bg"]))


# ---------------------------------------------------------------- driver
_RUNS: list[ScenarioRun] = []
_CFG: Config | None = None
_GRID = THR_GRID


def _tables_worker(setting: PPSetting) -> tuple[PPSetting, dict]:
    assert _CFG is not None
    return setting, class_tables(_CFG, _RUNS, setting, _GRID)


def tune(
    cfg: Config,
    runs: list[ScenarioRun],
    settings: list[PPSetting] | None = None,
    budgets=BUDGETS,
    max_latency: float = MAX_LATENCY_S,
    jobs: int = 1,
    verbose: bool = True,
    grid=THR_GRID,
) -> dict:
    global _RUNS, _CFG, _GRID
    settings = settings or all_settings()
    weights = {c.key: (DANGER_WEIGHT if c.level == Level.DANGER else 1.0) for c in cfg.classes[:-1]}
    bg_h = sum(r.duration_s for r in runs if r.is_background) / 3600.0
    ev_h = 0.0
    from soundwatch.eval.streaming import gt_covered_seconds

    for r in runs:
        if not r.is_background:
            ev_h += (r.duration_s - gt_covered_seconds(r.gt, r.duration_s)) / 3600.0
    t0 = time.perf_counter()
    tables: dict[PPSetting, dict] = {}
    if jobs > 1:
        import multiprocessing as mp

        _RUNS, _CFG, _GRID = runs, cfg, tuple(grid)
        try:
            with ProcessPoolExecutor(jobs, mp_context=mp.get_context("fork")) as ex:
                for s, tab in ex.map(_tables_worker, settings):
                    tables[s] = tab
        finally:
            _RUNS, _CFG, _GRID = [], None, THR_GRID
    else:
        for s in settings:
            tables[s] = class_tables(cfg, runs, s, tuple(grid))
    if verbose:
        print(
            f"[tune] {len(settings)} settings x {len(grid)} thresholds replayed in "
            f"{time.perf_counter() - t0:.0f}s",
            flush=True,
        )
    points = []
    for s in settings:
        for b in budgets:
            idx, feasible = greedy_thresholds(tables[s], weights, b * bg_h)
            p = score_choice(tables[s], idx, weights, bg_h, ev_h)
            points.append(
                {
                    "setting": s.label,
                    "smooth_n": s.smooth_n,
                    "k_of_n": [s.k, s.n],
                    "bg_margin": s.bg_margin,
                    "budget_per_h": b,
                    "feasible": feasible,
                    "latency_ok": (p["latency_median_s"] or 99.0) <= max_latency,
                    **p,
                }
            )
    front = set(pareto_front(points))
    for i, p in enumerate(points):
        p["pareto"] = i in front
    best = {f"{b:g}": pick_best(points, b, max_latency) for b in budgets}
    n_gt = {c.key: sum(1 for r in runs for e in r.gt if e.class_id == c.id) for c in cfg.classes[:-1]}
    return {
        "n_event_scenarios": sum(1 for r in runs if not r.is_background),
        "n_background_scenarios": sum(1 for r in runs if r.is_background),
        "background_hours": bg_h,
        "n_gt": n_gt,
        "threshold_grid": list(grid),
        "budgets_per_h": list(budgets),
        "max_latency_s": max_latency,
        "danger_weight": DANGER_WEIGHT,
        "points": points,
        "best": best,
        # compact per-class tables: {setting label: {class: [[thr, tp, fa_bg, fa_ev], ...]}}
        "tables": {
            s.label: {
                k: [[r["thr"], r["tp"], r["fa_bg"], r["fa_ev"]] for r in rows] for k, rows in tab.items()
            }
            for s, tab in tables.items()
        },
    }


# ---------------------------------------------------------------- output
def _f(v, fmt="{:.3f}") -> str:
    return "–" if v is None else fmt.format(v)


def render_md(res: dict, model: str, scen_dir: str) -> str:
    out = [
        "# Tinh chỉnh hậu xử lý (WP8) – kịch bản **val**",
        f"Model `{model}`; kịch bản `{scen_dir}`: {res['n_event_scenarios']} kịch bản có sự kiện "
        f"({sum(res['n_gt'].values())} sự kiện), {res['n_background_scenarios']} kịch bản chỉ có nền "
        f"({res['background_hours']:.2f} giờ). Mục tiêu: tối đa recall có trọng số (DANGER ×"
        f"{res['danger_weight']:g}) với báo nhầm nền ≤ ngân sách và trễ median ≤ {res['max_latency_s']:g} s. "
        "Ngưỡng từng lớp chọn tham lam trên lưới "
        f"{res['threshold_grid'][0]:g}–{res['threshold_grid'][-1]:g}. Sinh bởi `python -m soundwatch tune`.",
        "## Cấu hình tốt nhất theo ngân sách báo nhầm",
    ]
    hdr = (
        "| Ngân sách FA/h | Hậu xử lý | Recall (trọng số) | Recall | Recall DANGER TB / min | FA/h nền | "
        "Trễ median / p90 (s) | Ngưỡng |"
    )
    out += [hdr, "|---|---|---|---|---|---|---|---|"]
    for b, p in res["best"].items():
        if p is None:
            out.append(f"| ≤ {b} | (không có điểm khả thi) | | | | | | |")
            continue
        thr = ", ".join(f"{k}={v:.2f}" for k, v in p["thresholds"].items())
        out.append(
            f"| ≤ {b} | {p['setting']} | {_f(p['weighted_recall'])} | {_f(p['recall'])} | "
            f"{_f(p['danger_recall_mean'])} / {_f(p['danger_recall_min'])} | "
            f"{_f(p['fa_per_hour_background'], '{:.1f}')} | "
            f"{_f(p['latency_median_s'], '{:.2f}')} / {_f(p['latency_p90_s'], '{:.2f}')} | {thr} |"
        )
    out.append("## Pareto front (recall có trọng số ↑, FA/h ↓, trễ median ↓)")
    out.append(f"Chỉ các điểm khả thi với ngân sách của chúng; ✓ = trễ median ≤ {res['max_latency_s']:g} s.")
    out += [
        "| Hậu xử lý | Ngân sách | Recall (trọng số) | Recall | FA/h nền | Trễ median / p90 (s) | Trễ OK |",
        "|---|---|---|---|---|---|---|",
    ]
    front = [p for p in res["points"] if p["pareto"] and p["feasible"]]
    front.sort(key=lambda p: (p["fa_per_hour_background"] or 0.0, -(p["weighted_recall"] or 0.0)))
    for p in front:
        out.append(
            f"| {p['setting']} | {p['budget_per_h']:g} | {_f(p['weighted_recall'])} | {_f(p['recall'])} | "
            f"{_f(p['fa_per_hour_background'], '{:.1f}')} | "
            f"{_f(p['latency_median_s'], '{:.2f}')} / {_f(p['latency_p90_s'], '{:.2f}')} | "
            f"{'✓' if p['latency_ok'] else '✗'} |"
        )
    out.append("## Tất cả điểm (theo ngân sách)")
    for b in res["budgets_per_h"]:
        pts = [p for p in res["points"] if p["budget_per_h"] == b]
        pts.sort(key=lambda p: (not p["feasible"], -(p["weighted_recall"] or 0.0)))
        out.append(f"### Ngân sách ≤ {b:g} báo nhầm / giờ")
        out += [
            "| Hậu xử lý | Khả thi | Recall (trọng số) | Recall | FA/h nền | Trễ median (s) |",
            "|---|---|---|---|---|---|",
        ]
        for p in pts:
            out.append(
                f"| {p['setting']} | {'✓' if p['feasible'] else '✗'} | {_f(p['weighted_recall'])} | "
                f"{_f(p['recall'])} | {_f(p['fa_per_hour_background'], '{:.1f}')} | "
                f"{_f(p['latency_median_s'], '{:.2f}')}{'' if p['latency_ok'] else ' ✗'} |"
            )
    return "\n\n".join(out[:3]) + "\n\n" + "\n".join(out[3:]) + "\n"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--scenarios", default=None, help=f"val scenario dir (default {DEFAULT_VAL_DIR})")
    parser.add_argument("--model", default=None, help="model (.tflite / .keras); default config model")
    parser.add_argument("--stats", default=None, help="feature_stats.json (default from config)")
    parser.add_argument(
        "--budgets", default=",".join(f"{b:g}" for b in BUDGETS), help="FA/h budgets, comma list"
    )
    parser.add_argument("--max-latency", type=float, default=MAX_LATENCY_S, help="max median latency (s)")
    parser.add_argument("--no-margin", action="store_true", help="do not search the bg_margin rule")
    parser.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--out-dir", default=None, help="report dir (default artifacts/reports)")
    parser.add_argument("--mode", default=None, help="context mode at start-up (config `modes:`)")
    parser.add_argument("--thr-grid", default=None,
                        help="comma list of thresholds replacing the default grid 0.30..0.95,0.97,0.99 "
                        "(calibrated models rarely need the 0.99 end)")


def main(args: argparse.Namespace) -> int:
    from soundwatch.config import with_mode

    cfg = with_mode(load_config(args.config), getattr(args, "mode", None))
    grid = THR_GRID
    if getattr(args, "thr_grid", None):
        grid = tuple(sorted(float(v) for v in args.thr_grid.split(",") if v.strip()))
    scen_dir = Path(args.scenarios) if args.scenarios else DEFAULT_VAL_DIR
    scen = list_scenarios(scen_dir)
    if not scen:
        print(
            f"[tune] no scenarios in {scen_dir}: run `python -m soundwatch make-scenarios --split val "
            f"--out {scen_dir} --no-demo` first"
        )
        return 1
    from soundwatch.engine.classifier import load_model

    mpath = Path(args.model) if args.model else cfg.model.resolved_tflite()
    spath = Path(args.stats) if args.stats else cfg.model.resolved_stats()
    if not mpath.exists() or not spath.exists():
        print(f"[tune] model {mpath} or stats {spath} missing")
        return 1
    clf, stats, _meta = load_model(mpath, spath, num_threads=1)
    runs = run_scenarios(cfg, scen, lambda gt: clf, stats, verbose=False)
    budgets = tuple(float(b) for b in str(args.budgets).split(",") if b.strip())
    settings = all_settings(margins=(None,) if args.no_margin else MARGIN_GRID)
    res = tune(cfg, runs, settings, budgets, args.max_latency, jobs=max(1, args.jobs), grid=grid)
    model = str(mpath.relative_to(paths.ROOT)) if mpath.is_relative_to(paths.ROOT) else str(mpath)
    sdir = str(scen_dir.relative_to(paths.ROOT)) if scen_dir.is_relative_to(paths.ROOT) else str(scen_dir)
    res["model"], res["scenarios"] = model, sdir
    out_dir = Path(args.out_dir) if args.out_dir else paths.REPORTS
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "tuning.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    (out_dir / "tuning.md").write_text(render_md(res, model, sdir), encoding="utf-8")
    for b, p in res["best"].items():
        if p is None:
            print(f"[tune] budget {b}/h: no admissible setting")
            continue
        print(
            f"[tune] budget {b}/h: {p['setting']} wrecall {p['weighted_recall']:.3f} "
            f"recall {p['recall']:.3f} FA/h {p['fa_per_hour_background']:.1f} "
            f"latency {p['latency_median_s']:.2f}s thr {p['thresholds']}"
        )
    print(f"[tune] -> {out_dir / 'tuning.md'}")
    return 0
