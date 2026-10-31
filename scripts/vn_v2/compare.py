# ruff: noqa: E501
"""Replay cached runs (runs_cache.py) with a config (+ mode) and print / save metrics.

usage: compare.py OUT.json NAME=CONFIG[@MODE] ...   (sets: all six)
"""
import json
import pickle
import sys

from soundwatch import paths
from soundwatch.config import load_config, with_mode
from soundwatch.eval.fast_path import SETS, short_metrics
from soundwatch.eval.streaming import aggregate, replay_all

RUNS = paths.REPORTS / "v2" / "runs"


def load(name, s):
    return pickle.load(open(RUNS / f"{name}_{s}.pkl", "rb"))


def evaluate(cfg, runs):
    rr = replay_all(cfg, runs, cfg.classes)
    a = aggregate(rr, cfg, with_details=False)
    pc = a["per_class"]
    return {
        "event_recall": a["event_recall"], "danger_recall_mean": a["danger_event_recall_mean"],
        "recall": {k: v["recall"] for k, v in pc.items()},
        "n_gt": {k: v["n_gt"] for k, v in pc.items()},
        "latency_median_s": a["latency_median_s"], "latency_p90_s": a["latency_p90_s"],
        "latency_median_by_class": {k: v["latency_median_s"] for k, v in pc.items()},
        "fa_per_hour_background": a["fa_per_hour_background"], "background_hours": a["background_hours"],
        "fa_background_by_class": {k: v["fa_background"] for k, v in pc.items()},
        "fa_per_hour_event_outside_gt": a["fa_per_hour_event_outside_gt"],
        "n_gt_events": a["n_gt_events"], "short": short_metrics(cfg, rr),
    }


def main():
    out = {}
    for spec in sys.argv[2:]:
        name, rest = spec.split("=", 1)
        cfgp, _, mode = rest.partition("@")
        model = name.split("+")[0]
        cfg = with_mode(load_config(cfgp), mode or None)
        label = name + (f"@{mode}" if mode else "")
        out[label] = {}
        for s in SETS:
            try:
                runs = load(model, s)
            except FileNotFoundError:
                continue
            out[label][s] = evaluate(cfg, runs)
            r = out[label][s]
            print(f"{label:<22}{s:<11} recall {r['event_recall']:.3f} danger {r['danger_recall_mean']:.3f} "
                  f"FA/h {r['fa_per_hour_background'] if r['fa_per_hour_background'] is None else round(r['fa_per_hour_background'],1)} "
                  f"lat {r['latency_median_s'] and round(r['latency_median_s'],2)} "
                  + " ".join(f"{k[:5]}={v:.2f}" for k, v in r["recall"].items() if v is not None), flush=True)
    json.dump(out, open(sys.argv[1], "w"), indent=1)


if __name__ == "__main__":
    main()
