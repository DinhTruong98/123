# ruff: noqa: E501
"""Choose post-processing for a model on VAL ONLY (val + vn_val scenario sets, cached runs).

For each danger_fast_threshold x k-of-n setting: per-class threshold tables (tune.class_tables) on
val ∪ vn_val, greedy thresholds (tune.greedy_thresholds) for several FA budgets; then every
candidate is replayed separately on val and vn_val. Admissible: FA/h(val) <= MAX_VAL,
FA/h(vn_val) <= MAX_VNVAL, median latency (val) <= 1.5 s. Best = max weighted recall on val ∪ vn_val
(DANGER x2), ties: fewer FA.

usage: select_pp.py MODEL CONFIG OUT.json [MAX_VAL MAX_VNVAL [FLOOR_DROP]]
With FLOOR_DROP: every class must keep a recall on val ∪ vn_val of at least
(recall of cu with config/soundwatch.yaml) - FLOOR_DROP; greedy starts from the highest thresholds
meeting these floors.
"""
import dataclasses
import json
import pickle
import sys

import numpy as np

from soundwatch import paths
from soundwatch.config import load_config
from soundwatch.eval import tune
from soundwatch.eval.streaming import DANGER_WEIGHT, aggregate, replay_all, with_thresholds
from soundwatch.types import Level

RUNS = paths.REPORTS / "v2" / "runs"
GRID = tuple(round(float(v), 2) for v in np.arange(0.30, 0.951, 0.05)) + (0.97, 0.99, 0.99609375)
FAST = (None, 0.8, 0.85, 0.9, 0.95, 0.97, 0.99, 0.99609375)
SETTINGS = [tune.PPSetting(1, 2, 2), tune.PPSetting(1, 1, 1), tune.PPSetting(2, 2, 3), tune.PPSetting(1, 2, 3)]
BUDGETS = (10.0, 15.0, 20.0, 25.0, 30.0, 35.0)

model, cfgp, outp = sys.argv[1:4]
max_val = float(sys.argv[4]) if len(sys.argv) > 4 else 22.0
max_vnval = float(sys.argv[5]) if len(sys.argv) > 5 else 32.0
floor_drop = float(sys.argv[6]) if len(sys.argv) > 6 else None
base = load_config(cfgp)
runs = {s: pickle.load(open(RUNS / f"{model}_{s}.pkl", "rb")) for s in ("val", "vn_val", "short_val")}
both = runs["val"] + runs["vn_val"]
bg_h = sum(r.duration_s for r in both if r.is_background) / 3600
weights = {c.key: (DANGER_WEIGHT if c.level == Level.DANGER else 1.0) for c in base.classes[:-1]}
floors = None
if floor_drop is not None:
    cu_cfg = load_config(paths.ROOT / "config/soundwatch.yaml")
    cu_runs = [r for s in ("val", "vn_val") for r in pickle.load(open(RUNS / f"cu_{s}.pkl", "rb"))]
    a = aggregate(replay_all(cu_cfg, cu_runs, cu_cfg.classes), cu_cfg, with_details=False)
    floors = {k: max(0.0, v["recall"] - floor_drop) for k, v in a["per_class"].items()}
    print("floors", {k: round(v, 3) for k, v in floors.items()})


def greedy_floor(tabs, budget):
    idx = {}
    for k, rows in tabs.items():
        ok = [j for j, r in enumerate(rows) if r["n_gt"] and r["tp"] / r["n_gt"] >= floors[k] - 1e-9]
        if not ok:
            return None, False
        idx[k] = max(ok)
    if sum(tabs[k][i]["fa_bg"] for k, i in idx.items()) > budget + 1e-9:
        return None, False
    while True:
        cur_fa = sum(tabs[k][i]["fa_bg"] for k, i in idx.items())
        best, best_score = None, -1.0
        for k in tabs:
            cur = tabs[k][idx[k]]
            j = idx[k] - 1
            while j >= 0 and tabs[k][j]["tp"] <= cur["tp"]:
                j -= 1
            if j < 0:
                continue
            gain = (tabs[k][j]["tp"] - cur["tp"]) * weights[k]
            dfa = tabs[k][j]["fa_bg"] - cur["fa_bg"]
            if cur_fa + dfa > budget + 1e-9:
                continue
            score = gain / (max(dfa, 0) + 0.5)
            if score > best_score:
                best, best_score = (k, j), score
        if best is None:
            return idx, True
        idx[best[0]] = best[1]


cands = []
for fast in FAST:
    cfg = dataclasses.replace(base, postprocess=dataclasses.replace(base.postprocess, danger_fast_threshold=fast))
    for st in SETTINGS:
        tabs = tune.class_tables(cfg, both, st, GRID)
        seen = set()
        for b in BUDGETS:
            if floors is None:
                idx, ok = tune.greedy_thresholds(tabs, weights, b * bg_h)
            else:
                idx, ok = greedy_floor(tabs, b * bg_h)
            if not ok:
                continue
            thr = {k: tabs[k][i]["thr"] for k, i in idx.items()}
            key = tuple(sorted(thr.items()))
            if key in seen:
                continue
            seen.add(key)
            sc = tune.score_choice(tabs, idx, weights, bg_h, 1.0)
            c2 = st.apply(cfg)
            tid = {c2.class_keys.index(k): v for k, v in thr.items()}
            cl = with_thresholds(c2, tid)
            res = {}
            for s in ("val", "vn_val", "short_val"):
                a = aggregate(replay_all(c2, runs[s], cl), c2, with_details=False)
                res[s] = {"recall": a["event_recall"], "danger": a["danger_event_recall_mean"],
                          "fa_h": a["fa_per_hour_background"], "lat": a["latency_median_s"],
                          "per_class": {k: v["recall"] for k, v in a["per_class"].items()}}
            cands.append({"fast": fast, "setting": st.label, "smooth_n": st.smooth_n, "k_of_n": [st.k, st.n],
                          "budget": b, "thresholds": thr, "weighted_recall": sc["weighted_recall"],
                          "lat_both": sc["latency_median_s"], **res})
        print(f"fast={fast} {st.label}: {len(seen)} candidates", flush=True)


def admissible(c):
    return ((c["val"]["fa_h"] or 0) <= max_val + 1e-9 and (c["vn_val"]["fa_h"] or 0) <= max_vnval + 1e-9
            and (c["val"]["lat"] or 99) <= 1.5)


ok = [c for c in cands if admissible(c)]
ok.sort(key=lambda c: (-c["weighted_recall"], (c["val"]["fa_h"] or 0) + (c["vn_val"]["fa_h"] or 0)))
json.dump({"max_val": max_val, "max_vnval": max_vnval, "floor_drop": floor_drop, "floors": floors,
           "candidates": cands, "best": ok[:10]},
          open(outp, "w"), indent=1)
for c in ok[:8]:
    print(f"wrec {c['weighted_recall']:.3f} fast={c['fast']} {c['setting']} b={c['budget']} | "
          f"val rec {c['val']['recall']:.3f} dang {c['val']['danger']:.3f} FA {c['val']['fa_h']:.1f} lat {c['val']['lat']:.2f} | "
          f"vn_val rec {c['vn_val']['recall']:.3f} FA {c['vn_val']['fa_h']:.1f} | short {c['short_val']['recall']:.3f} | "
          + " ".join(f"{k[:5]}={v:.2f}" for k, v in c["thresholds"].items()))
