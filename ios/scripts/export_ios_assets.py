# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#     "numpy==2.4.6",
#     "pyyaml==6.0.3",
#     "scipy==1.17.1",
#     "soundfile==0.14.0",
#     "ai-edge-litert==2.3.0",
# ]
# ///
"""Export model, config, golden vectors and demo expectations for the iOS app.

    uv run ios/scripts/export_ios_assets.py          # macOS (tensorflow-cpu has no macOS wheel)
    uv sync && uv run python ios/scripts/export_ios_assets.py   # Linux, project environment

Writes to ios/SoundWatch/Resources/:
    student_int8.tflite        copy of assets/models/student_int8.tflite
    soundwatch_config.json     audio + postprocess params, classes, feature stats, Hann, sparse mel fb
    demo_scenario.wav          copy of assets/demo/demo_scenario.wav
    golden_features.json       5 golden cases (int16 input, logmel 49x40) + classifier reference
    demo_expected_events.json  events (and per-hop probs) of the Python pipeline on the demo file
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from soundwatch.audio import features as F  # noqa: E402
from soundwatch.audio.sources import FileSource  # noqa: E402
from soundwatch.config import load_config  # noqa: E402
from soundwatch.engine.classifier import TFLiteClassifier  # noqa: E402
from soundwatch.engine.pipeline import run_offline  # noqa: E402
from soundwatch.tools import export_golden  # noqa: E402

OUT = ROOT / "ios" / "SoundWatch" / "Resources"
MODEL = ROOT / "assets" / "models" / "student_int8.tflite"
STATS = ROOT / "assets" / "models" / "feature_stats.json"
DEMO_WAV = ROOT / "assets" / "demo" / "demo_scenario.wav"

SF_SYMBOLS = {
    "smoke_alarm": "flame.fill",
    "car_horn": "car.fill",
    "siren": "light.beacon.max.fill",
    "baby_cry": "figure.and.child.holdinghands",
    "door_knock": "door.left.hand.closed",
    "dog_bark": "pawprint.fill",
    "water_running": "drop.fill",
    "background": "ear",
}
GOLDEN_CLASSIFIER_CASE = 4


def f32(a) -> list:
    """float32 values as JSON floats that round-trip exactly to the same float32."""
    return [float(np.float32(v)) for v in np.asarray(a, np.float32).ravel()]


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")


def export_config(cfg, stats: dict) -> dict:
    a, p = cfg.audio, cfg.postprocess
    start, length, weights = F.mel_filterbank_sparse(
        F.mel_filterbank(a.sample_rate, a.n_fft, a.n_mels, a.fmin, a.fmax))
    offset = np.concatenate([[0], np.cumsum(length)[:-1]])
    bands = [{"start": int(s), "len": int(n), "weights": f32(weights[o:o + n])}
             for s, n, o in zip(start, length, offset, strict=True)]
    return {
        "audio": {
            "sample_rate": a.sample_rate, "window_samples": a.window_samples, "hop_samples": a.hop_samples,
            "n_fft": a.n_fft, "frame_step": a.frame_step, "n_frames": a.n_frames, "n_mels": a.n_mels,
            "fmin": a.fmin, "fmax": a.fmax, "log_eps": a.log_eps, "gate_dbfs": a.gate_dbfs,
        },
        "postprocess": {
            "smooth_n": p.smooth_n, "k_of_n": list(p.k_of_n), "danger_repeat_s": p.danger_repeat_s,
            "danger_clear_s": p.danger_clear_s, "bg_margin": p.bg_margin,
            "danger_fast_threshold": p.danger_fast_threshold,
        },
        "ui": {"alert_timeout_s": cfg.ui.alert_timeout_s, "history_len": cfg.ui.history_len},
        "classes": [{
            "key": c.key, "label_vi": c.label_vi, "level": int(c.level), "color": list(c.color),
            "vibration": list(c.vibration), "threshold": c.threshold, "cooldown_s": c.cooldown_s,
            "enabled": c.enabled, "sf_symbol": SF_SYMBOLS[c.key],
        } for c in cfg.classes],
        "feature_stats": {"mean": stats["mean"], "std": stats["std"]},
        "hann": f32(F.hann_window(a.n_fft)),
        "mel_bands": bands,
        **export_extras(cfg, stats),
    }


def export_extras(cfg, stats: dict) -> dict:
    """vn-v2 additions, present only when used (so the historical model/config export is unchanged):
    `model_frontend` (PCEN constants, sigmoid output, window length) and `modes` (context modes)."""
    out: dict = {}
    meta = F.model_meta(stats)
    if not meta.is_default:
        out["model_frontend"] = meta.to_params(cfg.audio.sample_rate, cfg.audio.frame_step)
    if cfg.modes is not None:
        m = cfg.modes
        out["modes"] = {
            "default": m.default,
            "modes": {n: {"disable": list(sp.disable), "level": sp.level_map} for n, sp in m.modes},
            "auto": None if m.auto is None else {
                "street_above_dbfs": m.auto.street_above_dbfs, "window_s": m.auto.window_s,
                "hysteresis_db": m.auto.hysteresis_db, "min_s": m.auto.min_s},
        }
    return out


def export_golden_json(cfg, stats: dict, clf: TFLiteClassifier) -> dict:
    meta = F.model_meta(stats)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        # == python -m soundwatch export-golden (+ PCEN references for a PCEN model)
        manifest = export_golden.export(out, cfg.audio, meta=meta if meta.frontend == "pcen" else None)
        cases = []
        for m in manifest:
            d = out / m["dir"]
            x = np.load(d / "input_f32.npy")
            lm = np.load(d / "logmel.npy")
            pcm = np.round(x.astype(np.float64) * 32768.0).astype(np.int16)
            assert np.array_equal(pcm.astype(np.float32) / np.float32(32768.0), x)
            case = {"name": m["name"], "description": m["description"],
                    "input_int16": pcm.tolist(), "logmel": f32(lm)}
            if (d / "pcen.npy").exists():
                case["pcen"] = f32(np.load(d / "pcen.npy"))
            cases.append(case)
    c = cases[GOLDEN_CLASSIFIER_CASE]
    lm = np.asarray(c.get("pcen", c["logmel"]), np.float32).reshape(cfg.audio.n_frames, cfg.audio.n_mels)
    feats = F.normalize(lm, stats)
    probs = clf.predict(feats)
    return {
        "n_frames": cfg.audio.n_frames, "n_mels": cfg.audio.n_mels, "cases": cases,
        "classifier": {"case": GOLDEN_CLASSIFIER_CASE, "features": f32(feats), "probs": f32(probs)},
    }


def export_demo(cfg, stats: dict) -> dict:
    clf = TFLiteClassifier(MODEL, output="sigmoid" if F.model_meta(stats).output == "sigmoid" else None)
    events, summary, st = run_offline(cfg, FileSource(DEMO_WAV, realtime=False), clf, stats)
    keys = cfg.class_keys
    return {
        "events": [{"class_key": keys[e.class_id], "t_audio": e.t_audio, "repeat": e.repeat,
                    "confidence": e.confidence} for e in events],
        "hops": [{"t_audio": float(t), "gated": bool(g), "probs": f32(p)}
                 for t, g, p in zip(summary.t_audio, summary.gated, summary.probs, strict=True)],
        "infer_ms_mean": st["infer_ms_mean"], "feature_ms_mean": st["feature_ms_mean"],
    }


def main(argv: list[str] | None = None) -> int:
    """Defaults = the deployed app model; `--config/--model/--stats` export another model (e.g. vn-v2,
    whose config adds `modes` and whose stats add `model_frontend`) - the Swift app must support PCEN /
    sigmoid / modes before such an export is used (docs/vn_v2_results.md)."""
    import argparse

    global MODEL, STATS
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--stats", default=None)
    args = ap.parse_args(argv)
    MODEL = Path(args.model) if args.model else MODEL
    STATS = Path(args.stats) if args.stats else STATS
    cfg = load_config(args.config)
    stats = F.load_stats(STATS)
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(MODEL, OUT / MODEL.name)
    shutil.copyfile(DEMO_WAV, OUT / DEMO_WAV.name)
    write_json(OUT / "soundwatch_config.json", export_config(cfg, stats))
    clf = TFLiteClassifier(MODEL, output="sigmoid" if F.model_meta(stats).output == "sigmoid" else None)
    write_json(OUT / "golden_features.json", export_golden_json(cfg, stats, clf))
    demo = export_demo(cfg, stats)
    write_json(OUT / "demo_expected_events.json", demo)
    for e in demo["events"]:
        print(f"  {e['t_audio']:6.1f} s  {e['class_key']:<14} {'repeat' if e['repeat'] else ''}")
    print(f"wrote {sorted(p.name for p in OUT.iterdir())} to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
