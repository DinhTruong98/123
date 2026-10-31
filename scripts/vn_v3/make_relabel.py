# ruff: noqa: E501
"""Write data/vn/relabel_v3.csv from the manual spectrogram review (vn-v3, docs/vn_v3_results.md §2).

The review indices refer to the VN smoke_alarm clips sorted by (source_id, file), i.e. the panel numbers
in artifacts/vn_v3/spec/smoke_*.png rendered by scripts/vn_v3/spectro.py.
usage: make_relabel.py data/vn/manifest.csv data/vn/relabel_v3.csv
"""

from __future__ import annotations

import csv
import sys

WAIL = "slow wail (motor/fire-truck siren): pitch glides up and down over seconds"
YELP = "yelp/sweep: tone sweeps up/down repeatedly (~2-4 sweeps/s), electronic siren"
DRILL = "fire-drill video: fire-truck wail/yelp dominates, no bell/beep visible"
SWEEP_SOUNDER = "electronic sounder with a repeating sawtooth sweep tone (no steady beep) = siren by sound"
BOTH = "ambiguous: clearly contains both a sweeping siren and steady beeps/tone; dropped"

# index -> (new_class, reason)
DECISIONS: dict[int, tuple[str, str]] = {}
for i in (12, 13):
    DECISIONS[i] = ("siren", SWEEP_SOUNDER + " (supermarket fire-alarm test)")
DECISIONS[20] = ("siren", WAIL)
DECISIONS[33] = ("siren", YELP + " ('coi hu den chop 103')")
for i in (37, 38, 39):
    DECISIONS[i] = ("siren", YELP + " (24 V alarm siren H207X)")
for i in (42, 43):
    DECISIONS[i] = ("siren", WAIL + " (motor siren LK-SCL)")
DECISIONS[44] = ("siren", YELP + " ('coi hu bao chay 12V' KM-628)")
for i in (66, 67):
    DECISIONS[i] = ("siren", SWEEP_SOUNDER + " (wireless strobe siren GS-WS26)")
for i in (68, 69):
    DECISIONS[i] = ("siren", YELP + " (fire drill)")
DECISIONS[70] = ("siren", "repeating rising whoop sweep (fire drill)")
for i in (84, 85, 86):
    DECISIONS[i] = ("siren", YELP + " ('coi hu den chop 103')")
for i in (87, 88, 89, 90, 91, 92):
    DECISIONS[i] = ("siren", DRILL)
DECISIONS[96] = ("siren", YELP + " (12 V strobe siren)")
for i in (97, 98, 99):
    DECISIONS[i] = ("siren", DRILL)
DECISIONS[100] = ("drop", BOTH + " (steady 4 kHz tone + fire-truck wail onset)")
DECISIONS[101] = ("drop", BOTH + " (steady 4 kHz tone + 3 s yelp)")
for i in (108, 109, 110):
    DECISIONS[i] = ("siren", YELP + " (remote alarm siren)")
for i in (117, 118):
    DECISIONS[i] = ("siren", YELP + " (siren tested as 'chuong bao chay')")
for i in (119, 120, 121, 122, 124):
    DECISIONS[i] = ("siren", DRILL)
DECISIONS[123] = ("drop", BOTH + " (yelp, then horn beeps)")
for i in (125, 126, 127):
    DECISIONS[i] = ("siren", SWEEP_SOUNDER + " (automatic fire-alarm sounder)")
for i in (128, 129):
    DECISIONS[i] = ("drop", BOTH + " (steady 2.8 kHz beeps + sweeping strobe siren)")
for i in (135, 136, 137):
    DECISIONS[i] = ("siren", DRILL)
for i in (138, 139, 140, 141, 142):
    DECISIONS[i] = ("drop", BOTH + " (drill: yelp + 3.5 kHz horn-strobe buzz blocks)")
DECISIONS[157] = ("drop", BOTH + " (steady 2.9 kHz tone + whoop chirps)")


def main() -> None:
    rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8")))
    smoke = sorted((r for r in rows if r["class_key"] == "smoke_alarm"), key=lambda r: (r["source_id"], r["file"]))
    with open(sys.argv[2], "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "old_class", "new_class", "reason", "siren_score", "alarm_score", "title"])
        for i, r in enumerate(smoke):
            if i in DECISIONS:
                new, why = DECISIONS[i]
                w.writerow([r["file"], "smoke_alarm", new, f"[{r['split']}/{r['label_confidence']}] {why}",
                            r["siren_score"], r["alarm_score"], r["title"]])
    n_s = sum(v[0] == "siren" for v in DECISIONS.values())
    n_d = sum(v[0] == "drop" for v in DECISIONS.values())
    print(f"{n_s} -> siren, {n_d} dropped, {len(smoke) - n_s - n_d} stay smoke_alarm (of {len(smoke)})")


if __name__ == "__main__":
    main()
