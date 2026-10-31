# ruff: noqa: E501
"""Combined val score of each model's chosen config: weighted recall over val ∪ vn_val ∪ short_val."""
import json
import sys
from pathlib import Path

R = Path(__file__).resolve().parents[2] / "artifacts" / "reports" / "v2"
n_gt = json.load(open(R / "cmp_base.json"))["cu"]
DANGER = {"smoke_alarm", "car_horn", "siren"}
for t in sys.argv[1:]:
    for kind in ("selectF", "select"):
        p = R / f"{kind}_{t}.json"
        if not p.exists():
            continue
        best = json.load(open(p))["best"]
        if not best:
            continue
        x = best[0]
        num = den = 0.0
        for s in ("val", "vn_val", "short_val"):
            for k, r in x[s]["per_class"].items():
                n = n_gt[s]["n_gt"].get(k, 0)
                if not n or r is None:
                    continue
                w = 2.0 if k in DANGER else 1.0
                num += w * r * n
                den += w * n
        print(f"{t:<20}{kind:<8} score3 {num / den:.3f}  wrec {x['weighted_recall']:.3f}  FA val {x['val']['fa_h']} vn {x['vn_val']['fa_h']}")
