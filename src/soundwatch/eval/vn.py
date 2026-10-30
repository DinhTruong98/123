"""Window-level comparison of int8 models on the VN custom data vs the original datasets.

Splits the test (or val) windows by origin: ``source_clip`` starting with ``vn/`` (clips added with
``build-dataset --custom-manifest``) vs everything else (ESC-50 / US8K / FSD50K). For the VN siren windows
the recall is also broken down by the manifest columns ``vehicle_type``, ``passing`` and ``distance``.

Each model is a ``(name, tflite, feature_stats.json)`` triple, so the deployed baseline
(``assets/models/student_int8.tflite`` + its stats) and a newly trained model are scored on exactly the
same windows.
Output: reports/vn_eval.json and reports/vn_eval.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from soundwatch import paths
from soundwatch.config import Config, load_config


def _features(waves: np.ndarray, cfg: Config, stats_path: Path) -> np.ndarray:
    from soundwatch.audio.features import compute_logmel_batch, load_stats, normalize

    return normalize(compute_logmel_batch(waves, cfg.audio), load_stats(stats_path))[..., None]


def _recall(y: np.ndarray, p: np.ndarray, cls: int) -> tuple[float, int]:
    m = y == cls
    return (float((p[m] == cls).mean()) if m.any() else float("nan")), int(m.sum())


def evaluate(cfg: Config, models: list[tuple[str, Path, Path]], split: str, manifest: Path | None) -> dict:
    from soundwatch.eval.offline import metrics_from_predictions, predict_tflite
    from soundwatch.models import common

    d = common.load_split(split)
    is_vn = np.char.startswith(d["source_clip"].astype(str), "vn/")
    meta = None
    if manifest is not None:
        df = pd.read_csv(manifest)
        meta = df.set_index("file")
    siren = cfg.class_keys.index("siren")
    bg = cfg.background_id
    out: dict = {"split": split, "n_vn": int(is_vn.sum()), "n_orig": int((~is_vn).sum()), "models": {}}
    for name, model, stats in models:
        x = _features(d["wave"], cfg, stats)
        pred = predict_tflite(model, x).argmax(axis=1)
        res: dict = {}
        for part, mask in (("orig", ~is_vn), ("vn", is_vn)):
            if not mask.any():
                continue
            m = metrics_from_predictions(d["label"][mask], pred[mask], cfg)
            res[part] = {
                "macro_f1": m["macro_f1"],
                "per_class_recall": {k: v["recall"] for k, v in m["per_class"].items() if v["support"]},
                "support": {k: v["support"] for k, v in m["per_class"].items() if v["support"]},
                # background windows predicted as any alert class
                "bg_false_alert_rate": float((pred[mask][d["label"][mask] == bg] != bg).mean())
                if (d["label"][mask] == bg).any()
                else None,
                "bg_as_siren_rate": float((pred[mask][d["label"][mask] == bg] == siren).mean())
                if (d["label"][mask] == bg).any()
                else None,
            }
        if meta is not None and is_vn.any():
            files = np.char.replace(d["source_clip"][is_vn].astype(str), "vn/", "", count=1)
            y, p = d["label"][is_vn], pred[is_vn]
            breakdown = {}
            for col in ("vehicle_type", "passing", "distance"):
                vals = meta.reindex(files)[col].astype(str).to_numpy()
                breakdown[col] = {}
                for v in sorted(set(vals[y == siren])):
                    r, n = _recall(y[vals == v], p[vals == v], siren)
                    breakdown[col][v] = {"recall": r, "windows": n}
            res["vn_siren_breakdown"] = breakdown
        out["models"][name] = res
    return out


def to_markdown(r: dict) -> str:
    names = list(r["models"])
    lines = [
        f"# So sánh model trên dữ liệu VN ({r['split']}, mức cửa sổ 1 s)",
        "",
        f"Cửa sổ VN: {r['n_vn']}, cửa sổ gốc (ESC-50/US8K/FSD50K): {r['n_orig']}.",
        "",
    ]
    for part, title in (("vn", "Dữ liệu VN"), ("orig", "Dữ liệu gốc")):
        if not any(part in r["models"][n] for n in names):
            continue
        keys = list(r["models"][names[0]][part]["per_class_recall"])
        lines += [f"## {title}", "", "| Chỉ số | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
        lines.append(
            "| macro-F1 | " + " | ".join(f"{r['models'][n][part]['macro_f1']:.3f}" for n in names) + " |"
        )
        for k in keys:
            sup = r["models"][names[0]][part]["support"][k]
            lines.append(
                f"| recall `{k}` ({sup}) | "
                + " | ".join(f"{r['models'][n][part]['per_class_recall'][k]:.3f}" for n in names)
                + " |"
            )
        for k, lab in (
            ("bg_false_alert_rate", "nền → báo động (tỷ lệ cửa sổ)"),
            ("bg_as_siren_rate", "nền → siren"),
        ):
            if r["models"][names[0]][part][k] is not None:
                lines.append(
                    f"| {lab} | " + " | ".join(f"{r['models'][n][part][k]:.3f}" for n in names) + " |"
                )
        lines.append("")
    if all("vn_siren_breakdown" in r["models"][n] for n in names):
        lines += [
            "## Recall còi VN theo nhóm",
            "",
            "| Nhóm | " + " | ".join(names) + " |",
            "|---|" + "---|" * len(names),
        ]
        b0 = r["models"][names[0]]["vn_siren_breakdown"]
        for col, vals in b0.items():
            for v, d in vals.items():
                cells = [f"{r['models'][n]['vn_siren_breakdown'][col][v]['recall']:.3f}" for n in names]
                lines.append(f"| {col}={v} ({d['windows']} cửa sổ) | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--model",
        action="append",
        nargs=3,
        metavar=("NAME", "TFLITE", "STATS"),
        required=True,
        help="repeatable: name, int8 .tflite, feature_stats.json",
    )
    parser.add_argument("--manifest", default=None, help="custom manifest.csv for the VN siren breakdown")
    parser.add_argument("--out-dir", default=None)


def main(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    models = [(n, Path(m), Path(s)) for n, m, s in args.model]
    r = evaluate(cfg, models, args.split, Path(args.manifest) if args.manifest else None)
    out = Path(args.out_dir) if args.out_dir else paths.REPORTS
    out.mkdir(parents=True, exist_ok=True)
    (out / f"vn_eval_{args.split}.json").write_text(json.dumps(r, indent=2, ensure_ascii=False))
    md = to_markdown(r)
    (out / f"vn_eval_{args.split}.md").write_text(md + "\n")
    print(md)
    return 0
