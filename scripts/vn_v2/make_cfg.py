"""Write a config YAML from a select_pp.py result (best candidate, or index).

usage: make_cfg.py SELECT.json BASE.yaml OUT.yaml MODEL_DIR [index]
"""
import json
import re
import sys

sel = json.load(open(sys.argv[1]))
base = open(sys.argv[2]).read()
out, mdir = sys.argv[3], sys.argv[4]
i = int(sys.argv[5]) if len(sys.argv) > 5 else 0
c = sel["best"][i]
s = base
s = re.sub(r"(?m)^  smooth_n: .*$", f"  smooth_n: {c['smooth_n']}", s)
s = re.sub(r"(?m)^  k_of_n: .*$", f"  k_of_n: [{c['k_of_n'][0]}, {c['k_of_n'][1]}]", s)
fast = "null" if c["fast"] is None else repr(c["fast"])
s = re.sub(r"(?m)^  danger_fast_threshold: .*$", f"  danger_fast_threshold: {fast}", s)
s = re.sub(r"(?m)^(  tflite_path|  bundled_tflite_path): .*$", rf"\1: {mdir}/student_int8.tflite", s)
s = re.sub(r"(?m)^(  stats_path|  bundled_stats_path): .*$", rf"\1: {mdir}/feature_stats.json", s)
for k, v in c["thresholds"].items():
    s = re.sub(rf"(?m)^(  - {{key: {k},.*threshold: )[0-9.]+(,)", rf"\g<1>{v!r}\2", s)
open(out, "w").write(s)
print(out, c["fast"], c["setting"], c["thresholds"])
