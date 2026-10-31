"""Run each model once per scenario set and pickle the per-hop runs (for cheap replays).

usage: runs_cache.py NAME CONFIG MODEL STATS [sets]
"""
import pickle
import sys
import time

from soundwatch import paths
from soundwatch.config import load_config
from soundwatch.engine.classifier import load_model
from soundwatch.eval.fast_path import SETS
from soundwatch.eval.streaming import ScenarioRun, list_scenarios, run_scenarios

name, cfgp, model, stats = sys.argv[1:5]
sets = sys.argv[5].split(",") if len(sys.argv) > 5 else list(SETS)
cfg = load_config(cfgp)
clf, st, meta = load_model(model, stats)
out = paths.REPORTS / "v2" / "runs"
out.mkdir(parents=True, exist_ok=True)
for s in sets:
    p = out / f"{name}_{s}.pkl"
    if p.exists():
        continue
    t0 = time.time()
    runs = run_scenarios(cfg, list_scenarios(SETS[s]), lambda gt: clf, st, verbose=False)
    slim = [ScenarioRun(r.name, r.duration_s, r.gt, r.alerts, r.summary, None) for r in runs]
    for r in slim:
        r.summary.logmels = []
    pickle.dump(slim, open(p, "wb"))
    print(f"{name} {s}: {len(runs)} scenarios in {time.time() - t0:.0f}s", flush=True)
