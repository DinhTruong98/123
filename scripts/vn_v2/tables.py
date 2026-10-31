"""Markdown tables from compare.py JSON (event level). usage: tables.py CMP.json [names...]"""
import json
import sys

d = json.load(open(sys.argv[1]))
names = sys.argv[2:] or list(d)
SETS = [("test", "test (tổng hợp)"), ("val", "val"), ("vn_test", "VN test"), ("vn_val", "VN val"),
        ("short_test", "DANGER ngắn test"), ("short_val", "DANGER ngắn val")]
KEYS = ["smoke_alarm", "car_horn", "siren", "baby_cry", "door_knock", "dog_bark", "water_running"]


def f(v, fmt="{:.2f}"):
    return "–" if v is None else fmt.format(v)


def arrow(vals, fmt="{:.2f}"):
    return " / ".join(f(v, fmt) for v in vals)


print("| Bộ kịch bản | Recall tổng | DANGER TB | Báo nhầm/giờ nền | Trễ median (s) |")
print("|---|---|---|---|---|")
for s, lab in SETS:
    rows = [d[n].get(s) for n in names]
    if not any(rows):
        continue
    print(f"| {lab} | {arrow([r and r['event_recall'] for r in rows])} | "
          f"{arrow([r and r['danger_recall_mean'] for r in rows])} | "
          f"{arrow([r and r['fa_per_hour_background'] for r in rows], '{:.1f}')} | "
          f"{arrow([r and r['latency_median_s'] for r in rows])} |")
print()
print("Recall từng lớp (" + " / ".join(names) + "):")
print()
print("| Bộ | " + " | ".join(KEYS) + " |")
print("|---|" + "---|" * len(KEYS))
for s, lab in SETS:
    rows = [d[n].get(s) for n in names]
    if not all(rows):
        continue
    cells = []
    for k in KEYS:
        if rows[0]["n_gt"].get(k, 0) == 0:
            cells.append("–")
        else:
            cells.append(arrow([r["recall"][k] for r in rows]))
    print(f"| {lab} | " + " | ".join(cells) + " |")
print()
print("Báo nhầm trên nền theo lớp (số lần; " + " / ".join(names) + "):")
print()
print("| Bộ | giờ nền | " + " | ".join(KEYS) + " |")
print("|---|---|" + "---|" * len(KEYS))
for s, lab in SETS:
    rows = [d[n].get(s) for n in names]
    if not all(rows) or not rows[0]["background_hours"]:
        continue
    cells = [arrow([r["fa_background_by_class"][k] for r in rows], "{:d}") for k in KEYS]
    print(f"| {lab} | {rows[0]['background_hours']:.1f} | " + " | ".join(cells) + " |")
