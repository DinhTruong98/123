# ruff: noqa: E501
"""Headline vn-v3 metric: siren <-> smoke_alarm confusion, window and event level (new label definition).

Window level (artifacts/data/windows/<split>.npz, VN and original data separately):
  P(pred = siren | smoke_alarm), P(pred = smoke_alarm | siren), and the "DANGER alarm detected at all"
  recall: pred in {siren, smoke_alarm, car_horn} for windows of those three classes.
Event level (cached scenario runs of scripts/vn_v2/runs_cache.py, replayed with each model's config):
  for every ground-truth siren / smoke_alarm event, which alerts fire inside [onset, offset + 2 s]:
  correct class only, wrong class only, both ("flip"), none. Fresh alerts only (repeats ignored).
  "any DANGER" = at least one siren / smoke_alarm / car_horn alert inside the window.

usage: confusion.py OUT.json NAME=CONFIG[@MODE]:MODEL:STATS ...   (runs read from $SW_RUNS, see runs_cache.py)
"""

from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

from soundwatch import paths
from soundwatch.config import load_config, with_mode
from soundwatch.eval.streaming import TOL_AFTER_S, replay_all

RUNS = Path(os.environ.get("SW_RUNS", paths.REPORTS / "v2" / "runs"))
PRED_CACHE = paths.ARTIFACTS / "vn_v3" / "pred_cache"
SETS = ("test", "val", "vn_test", "vn_val", "short_test", "short_val")


def window_preds(name: str, model: str, stats: str, split: str, cfg):
    from soundwatch.eval.vn import _features, _predict
    from soundwatch.models import common

    PRED_CACHE.mkdir(parents=True, exist_ok=True)
    d = common.load_split(split)
    src = paths.WINDOWS / f"{split}.npz"
    key = f"{name}_{split}_{src.stat().st_mtime_ns}.npy"
    p = PRED_CACHE / key
    if p.exists():
        probs = np.load(p)
    else:
        probs = _predict(Path(model), Path(stats), _features(d["wave"], cfg, Path(stats)))
        np.save(p, probs)
    return d["label"], d["source_clip"].astype(str), probs


RELABEL = paths.ROOT / "data" / "vn" / "relabel_v3.csv"


def window_level(cfg, y, src, probs):
    import csv

    k = cfg.class_keys
    si, sm, ch = k.index("siren"), k.index("smoke_alarm"), k.index("car_horn")
    pred = probs.argmax(1)
    is_vn = np.char.startswith(src, "vn/")
    moved = {"vn/" + r["file"] for r in csv.DictReader(open(RELABEL, encoding="utf-8")) if r["new_class"] == "siren"}
    is_moved = np.isin(src, list(moved))
    out = {}
    for part, m in (("vn", is_vn), ("orig", ~is_vn), ("all", np.ones_like(is_vn)),
                    ("vn_relabelled", is_moved), ("vn_not_relabelled", is_vn & ~is_moved)):
        r = {}
        for a, b, nm in ((sm, si, "smoke_as_siren"), (si, sm, "siren_as_smoke")):
            mm = m & (y == a)
            r[nm] = float((pred[mm] == b).mean()) if mm.any() else None
            r[f"n_{k[a]}"] = int(mm.sum())
        for c in (si, sm):
            mm = m & (y == c)
            r[f"recall_{k[c]}"] = float((pred[mm] == c).mean()) if mm.any() else None
        dm = m & np.isin(y, [si, sm, ch])
        r["danger_any_recall"] = float(np.isin(pred[dm], [si, sm, ch]).mean()) if dm.any() else None
        r["siren_or_smoke_recall"] = (float(np.isin(pred[m & np.isin(y, [si, sm])], [si, sm]).mean())
                                      if (m & np.isin(y, [si, sm])).any() else None)
        r["n_windows"] = int(m.sum())
        out[part] = r
    return out


def event_level(cfg, runs):
    k = cfg.class_keys
    si, sm, ch = k.index("siren"), k.index("smoke_alarm"), k.index("car_horn")
    rr = replay_all(cfg, runs, cfg.classes)
    cnt = {c: {"n": 0, "correct_only": 0, "wrong_only": 0, "flip": 0, "none": 0, "any_danger": 0,
               "hops": 0, "hops_wrong": 0, "hop_switches": 0}
           for c in ("siren", "smoke_alarm")}
    for r0, r in zip(runs, rr, strict=True):
        t, P = r0.summary.t_audio, r0.summary.probs
        for g in r.gt:  # hop level: confident hops inside the event where the other alarm class wins
            if g.class_id not in (si, sm):
                continue
            m = (t >= g.onset + 0.5) & (t <= g.offset) & (P[:, si] + P[:, sm] >= 0.5)
            other = sm if g.class_id == si else si
            wrong = P[m, other] > P[m, g.class_id]
            c = cnt[k[g.class_id]]
            c["hops"] += int(m.sum())
            c["hops_wrong"] += int(wrong.sum())
            c["hop_switches"] += int((wrong[1:] != wrong[:-1]).sum())
        fresh = [a for a in r.alerts if not a.repeat]
        for g in r.gt:
            if g.class_id not in (si, sm):
                continue
            inside = {a.class_id for a in fresh if g.onset <= a.t_audio <= g.offset + TOL_AFTER_S}
            other = sm if g.class_id == si else si
            c = cnt[k[g.class_id]]
            c["n"] += 1
            has, wrong = g.class_id in inside, other in inside
            c["flip" if has and wrong else "correct_only" if has else "wrong_only" if wrong else "none"] += 1
            c["any_danger"] += int(bool(inside & {si, sm, ch}))
    for c in cnt.values():
        n = max(c["n"], 1)
        c.update({f"{x}_rate": c[x] / n for x in ("correct_only", "wrong_only", "flip", "none", "any_danger")})
        c["wrong_any_rate"] = (c["wrong_only"] + c["flip"]) / n
        c["hop_wrong_rate"] = c["hops_wrong"] / max(c["hops"], 1)
        c["hop_switches_per_event"] = c["hop_switches"] / n
    return cnt


def _f(v):
    return "-" if v is None else f"{v:.3f}"


def main():
    out = {}
    for spec in sys.argv[2:]:
        name, rest = spec.split("=", 1)
        cfgp, model, stats = rest.split(":")
        cfgp, _, mode = cfgp.partition("@")
        cfg = with_mode(load_config(cfgp), mode or None)
        runname = name.split("@")[0]
        res = {"window": {}, "event": {}}
        for split in ("test", "val"):
            y, src, probs = window_preds(runname, model, stats, split, cfg)
            res["window"][split] = w = window_level(cfg, y, src, probs)
            print(f"{name:<10} win {split:<5} " + " | ".join(
                f"{p}: smoke->siren {_f(w[p]['smoke_as_siren'])} siren->smoke {_f(w[p]['siren_as_smoke'])} danger-any {_f(w[p]['danger_any_recall'])}"
                for p in ("vn", "orig", "vn_relabelled")), flush=True)
        for s in SETS:
            f = RUNS / f"{runname}_{s}.pkl"
            if not f.exists():
                continue
            res["event"][s] = e = event_level(cfg, pickle.load(open(f, "rb")))
            print(f"{name:<10} evt {s:<10} " + " | ".join(
                f"{c}: n {e[c]['n']} ok {e[c]['correct_only_rate']:.2f} wrong {e[c]['wrong_only_rate']:.2f} flip {e[c]['flip_rate']:.2f} none {e[c]['none_rate']:.2f} anyD {e[c]['any_danger_rate']:.2f} hopW {e[c]['hop_wrong_rate']:.3f}"
                for c in ("siren", "smoke_alarm") if e[c]["n"]), flush=True)
        out[name] = res
    json.dump(out, open(sys.argv[1], "w"), indent=1)


if __name__ == "__main__":
    main()
