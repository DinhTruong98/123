"""WP8: diagnose false alarms on the background-only test streams.

    python -m soundwatch fa-analysis     # after eval-streaming -> artifacts/reports/fa_analysis.{md,json}

For every false alarm on a `bg_XX` scenario (top `--top` by confidence):
  1. find the background clip(s) that were playing in the analysed audio [t - span, t] using the
     `bg_segments` recorded by `make-scenarios` in index.json (start_s, end_s, clip);
  2. look up the clip's original dataset label (UrbanSound8K class / ESC-50 category) and, for
     UrbanSound8K, whether *other slices of the same Freesound recording* are labelled with the
     alarmed class (strong hint that the sound is really present in the "background" clip);
  3. score the audio window with YAMNet (AudioSet, 521 classes; `soundwatch.models.yamnet_loader`):
     score of the AudioSet classes matching the alarmed class (`YAMNET_GROUPS`, max over classes,
     mean over YAMNet patches) and top-3 labels; also the max over 1 s windows of the whole clip.
A false alarm is classified as likely *label noise* (the sound is really there) if YAMNet's group
score on the window is >= `YAMNET_HIT` or the same recording has slices labelled with that class.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from soundwatch import paths
from soundwatch.config import load_config

YAMNET_HIT = 0.2
WINDOW_SPAN_S = 1.5  # smooth_n / k_of_n look at the last two 1 s windows (hop 0.5 s)
YAMNET_GROUPS: dict[str, tuple[str, ...]] = {
    "smoke_alarm": (
        "Alarm",
        "Alarm clock",
        "Smoke detector, smoke alarm",
        "Fire alarm",
        "Beep, bleep",
        "Buzzer",
        "Reversing beeps",
        "Car alarm",
    ),
    "car_horn": ("Vehicle horn, car horn, honking", "Air horn, truck horn", "Train horn"),
    "siren": (
        "Siren",
        "Civil defense siren",
        "Police car (siren)",
        "Ambulance (siren)",
        "Fire engine, fire truck (siren)",
        "Emergency vehicle",
    ),
    "baby_cry": ("Baby cry, infant cry", "Crying, sobbing"),
    "door_knock": ("Knock", "Doorbell", "Tap"),
    "dog_bark": ("Dog", "Bark", "Yip", "Howl", "Bow-wow", "Whimper (dog)", "Canidae, dogs, wolves"),
    "water_running": (
        "Water tap, faucet",
        "Pour",
        "Toilet flush",
        "Drip",
        "Water",
        "Sink (filling or washing)",
        "Bathtub (filling or washing)",
    ),
}
US8K_CLASSES = (
    "air_conditioner",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music",
)
# target SoundWatch class -> UrbanSound8K class name with the same meaning
US8K_TARGET = {"car_horn": "car_horn", "siren": "siren", "dog_bark": "dog_bark"}

_US8K_RE = re.compile(r"^us8k_(?:fold\d+_)?(\d+)-(\d+)-(\d+)-(\d+)\.wav$")
_ESC_RE = re.compile(r"^esc50_(\d)-(\d+)-([A-Z])-(\d+)\.wav$")


def parse_clip(name: str) -> dict:
    """Noise-bank file name -> {dataset, fsid, label, ...}."""
    m = _US8K_RE.match(name)
    if m:
        cid = int(m.group(2))
        return {
            "dataset": "UrbanSound8K",
            "fsid": int(m.group(1)),
            "class_id": cid,
            "label": US8K_CLASSES[cid] if 0 <= cid < len(US8K_CLASSES) else str(cid),
        }
    m = _ESC_RE.match(name)
    if m:
        return {"dataset": "ESC-50", "fsid": int(m.group(2)), "class_id": int(m.group(4)), "label": None}
    return {"dataset": "?", "fsid": None, "class_id": None, "label": None}


def segments_at(segments: list, t0: float, t1: float) -> list[tuple[str, float]]:
    """Clips overlapping [t0, t1], as (name, overlap seconds), largest overlap first."""
    out = []
    for s, e, name in segments:
        ov = min(e, t1) - max(s, t0)
        if ov > 0:
            out.append((name, round(float(ov), 3)))
    return sorted(out, key=lambda x: -x[1])


def group_indices(names: list[str], key: str) -> list[int]:
    want = set(YAMNET_GROUPS.get(key, ()))
    return [i for i, n in enumerate(names) if n in want]


def collect_fas(st: dict, top: int | None = None) -> list[dict]:
    """Background-scenario false alarms from streaming_metrics.json, highest confidence first."""
    out = []
    for sc in st.get("scenarios", []):
        if not sc["name"].startswith("bg_"):
            continue
        conf = {(a["class_key"], a["t"]): a["conf"] for a in sc["alerts"] if not a["repeat"]}
        for fa in sc["fa_alerts"]:
            out.append(
                {
                    "scenario": sc["name"],
                    "class_key": fa["class_key"],
                    "t": fa["t"],
                    "conf": conf.get((fa["class_key"], fa["t"])),
                }
            )
    out.sort(key=lambda d: (-(d["conf"] or 0.0), d["scenario"], d["t"]))
    return out[:top] if top else out


def _us8k_siblings(meta_csv: Path) -> dict[int, set[str]]:
    import csv

    sib: dict[int, set[str]] = {}
    if not meta_csv.exists():
        return sib
    with open(meta_csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            sib.setdefault(int(r["fsID"]), set()).add(r["class"])
    return sib


def _esc_categories(meta_csv: Path) -> dict[int, str]:
    import csv

    if not meta_csv.exists():
        return {}
    with open(meta_csv, newline="", encoding="utf-8") as f:
        return {int(r["target"]): r["category"] for r in csv.DictReader(f)}


def analyse(
    cfg, st: dict, index: dict, scen_dir: Path, noise_dir: Path, top: int = 20, use_yamnet: bool = True
) -> dict:
    from soundwatch.data.scenarios import load_wav

    sr = cfg.audio.sample_rate
    segs = {b["wav"][:-4]: b.get("bg_segments") for b in index.get("background_scenarios", [])}
    sib = _us8k_siblings(paths.RAW / "urbansound8k" / "UrbanSound8K" / "metadata" / "UrbanSound8K.csv")
    esc = _esc_categories(paths.RAW / "esc50" / "meta" / "esc50.csv")
    yam = None
    if use_yamnet:
        from soundwatch.models.yamnet_loader import Yamnet

        yam = Yamnet()
    rows = []
    wav_cache: dict[str, np.ndarray] = {}
    for fa in collect_fas(st, top):
        key = fa["class_key"]
        t = float(fa["t"])
        row = dict(fa)
        seg = segments_at(segs.get(fa["scenario"]) or [], t - WINDOW_SPAN_S, t)
        row["clips"] = seg
        clip = seg[0][0] if seg else None
        info = parse_clip(clip) if clip else {"dataset": "?", "fsid": None, "label": None}
        if info["dataset"] == "ESC-50":
            info["label"] = esc.get(info["class_id"], str(info["class_id"]))
        row["clip"] = clip
        row["clip_label"] = f"{info['dataset']}: {info['label']}"
        sibs = sib.get(info["fsid"], set()) if info["dataset"] == "UrbanSound8K" else set()
        row["recording_has_target_label"] = US8K_TARGET.get(key) in sibs if key in US8K_TARGET else False
        row["recording_labels"] = sorted(sibs)
        if yam is not None:
            if fa["scenario"] not in wav_cache:
                wav_cache[fa["scenario"]] = load_wav(scen_dir / f"{fa['scenario']}.wav", sr)
            x = wav_cache[fa["scenario"]]
            w = x[max(0, int((t - WINDOW_SPAN_S) * sr)) : int(t * sr)]
            s, _ = yam(w)
            gi = group_indices(yam.names, key)
            row["yamnet_window"] = float(s[0, gi].max()) if gi else None
            row["yamnet_top3"] = [(yam.names[i], round(float(s[0, i]), 3)) for i in np.argsort(-s[0])[:3]]
            row["yamnet_clip_max"] = None
            if clip and (noise_dir / clip).exists():
                c = load_wav(noise_dir / clip, sr)
                if len(c) >= sr:
                    starts = range(0, len(c) - sr + 1, sr // 2)
                    sc, _ = yam(np.stack([c[i : i + sr] for i in starts]))
                    row["yamnet_clip_max"] = float(sc[:, gi].max()) if gi else None
        hit = (row.get("yamnet_window") or 0.0) >= YAMNET_HIT
        row["label_noise_likely"] = bool(hit or row["recording_has_target_label"])
        rows.append(row)
    n = len(rows)
    n_noise = sum(r["label_noise_likely"] for r in rows)
    return {
        "n_analysed": n,
        "n_label_noise_likely": n_noise,
        "yamnet_hit": YAMNET_HIT,
        "window_span_s": WINDOW_SPAN_S,
        "rows": rows,
        "total_background_fa": len(collect_fas(st)),
    }


def render_md(res: dict, st: dict) -> str:
    def f(v):
        return "–" if v is None else f"{v:.2f}"

    lines = [
        "# Phân tích báo nhầm trên kịch bản chỉ có nền (test)",
        "",
        f"Model `{st.get('model')}`, hậu xử lý {st.get('postprocess')}. {res['n_analysed']} / "
        f"{res['total_background_fa']} báo nhầm (độ tin cậy cao nhất trước). Mỗi báo nhầm được truy ngược về "
        f"clip nền gốc (`bg_segments` trong `index.json`), tra nhãn gốc của dataset, và chấm đoạn âm thanh "
        f"[t − {res['window_span_s']:g} s, t] bằng YAMNet (AudioSet). *YAMNet (cửa sổ)* = điểm cao nhất của "
        "các lớp AudioSet tương ứng với lớp bị báo; *YAMNet (cả clip)* = max trên các cửa sổ 1 s "
        "của clip gốc. *Ghi âm có nhãn* = các slice khác của cùng bản ghi Freesound trong UrbanSound8K "
        "được gán đúng lớp đó.",
        "",
        "| # | Kịch bản | t (s) | Báo | Conf | Clip nền (nhãn gốc) | Ghi âm có nhãn | YAMNet (cửa sổ) | "
        "YAMNet (cả clip) | YAMNet top-3 | Nhiễu nhãn? |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(res["rows"], 1):
        top3 = ", ".join(f"{n} {s:.2f}" for n, s in r.get("yamnet_top3") or [])
        lines.append(
            f"| {i} | {r['scenario']} | {r['t']:.1f} | {r['class_key']} | {f(r['conf'])} | "
            f"`{r['clip']}` ({r['clip_label']}) | {'có' if r['recording_has_target_label'] else '–'} | "
            f"{f(r.get('yamnet_window'))} | {f(r.get('yamnet_clip_max'))} | {top3} | "
            f"{'**có**' if r['label_noise_likely'] else 'không'} |"
        )
    lines += ["", "## Tóm tắt", ""] + summary_lines(res)
    return "\n".join(lines) + "\n"


def summarize(res: dict) -> dict:
    """Counts by alarmed class and by (alarmed class, source label)."""
    by_class: dict[str, list[int]] = {}
    by_source: dict[str, int] = {}
    for r in res["rows"]:
        v = by_class.setdefault(r["class_key"], [0, 0])
        v[0] += 1
        v[1] += int(r["label_noise_likely"])
        k = f"{r['class_key']} ← {r['clip_label']}"
        by_source[k] = by_source.get(k, 0) + 1
    top = sorted(by_source.items(), key=lambda kv: -kv[1])
    n = res["n_analysed"]
    return {
        "by_class": by_class,
        "top_sources": top[:8],
        "label_noise_fraction": res["n_label_noise_likely"] / n if n else None,
        "recording_label_hits": sum(1 for r in res["rows"] if r["recording_has_target_label"]),
    }


def summary_lines(res: dict) -> list[str]:
    sm = summarize(res)
    n, k = res["n_analysed"], res["n_label_noise_likely"]
    out = [
        f"- Có khả năng là nhiễu nhãn (YAMNet ≥ {res['yamnet_hit']:g} hoặc bản ghi có nhãn đó): "
        f"**{k}/{n}**; bản ghi UrbanSound8K có slice cùng lớp: {sm['recording_label_hits']}.",
        "- Theo lớp bị báo (số báo nhầm / nghi nhiễu nhãn): "
        + ", ".join(
            f"{c} {v[0]}/{v[1]}" for c, v in sorted(sm["by_class"].items(), key=lambda kv: -kv[1][0])
        ),
        "- Nguồn nền gây báo nhầm nhiều nhất: " + ", ".join(f"{s} ({c})" for s, c in sm["top_sources"]),
    ]
    return out


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--metrics", default=None, help="streaming_metrics.json (default artifacts/reports)")
    parser.add_argument("--scenarios", default=None, help="scenario dir (default artifacts/data/scenarios)")
    parser.add_argument("--noise", default=None, help="noise bank dir (default artifacts/data/noise/test)")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--no-yamnet", action="store_true", help="skip YAMNet scoring")
    parser.add_argument("--out-dir", default=None, help="default artifacts/reports")


def main(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    out_dir = Path(args.out_dir) if args.out_dir else paths.REPORTS
    mpath = Path(args.metrics) if args.metrics else paths.REPORTS / "streaming_metrics.json"
    scen_dir = Path(args.scenarios) if args.scenarios else paths.SCENARIOS
    noise_dir = Path(args.noise) if args.noise else paths.NOISE / "test"
    try:
        st = json.loads(mpath.read_text(encoding="utf-8"))
        index = json.loads((scen_dir / "index.json").read_text(encoding="utf-8"))
    except OSError as e:
        print(f"[fa-analysis] {e}: run make-scenarios + eval-streaming first")
        return 1
    if not any(b.get("bg_segments") for b in index.get("background_scenarios", [])):
        print("[fa-analysis] index.json has no bg_segments: re-run make-scenarios (same seed -> same audio)")
    res = analyse(cfg, st, index, scen_dir, noise_dir, args.top, use_yamnet=not args.no_yamnet)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "fa_analysis.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    (out_dir / "fa_analysis.md").write_text(render_md(res, st), encoding="utf-8")
    print(
        f"[fa-analysis] {res['n_label_noise_likely']}/{res['n_analysed']} FAs look like label noise "
        f"-> {out_dir / 'fa_analysis.md'}"
    )
    return 0
