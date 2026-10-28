"""WP7: model size / MACs / arena / PC latency benchmark + estimated MCU latency.

For each available model (student_int8.tflite, student_float.tflite, student_distill.keras,
student_nodistill.keras, or `--model` paths):
  * params, analytic MACs (Keras: models.dscnn.count_macs; TFLite: from CONV_2D /
    DEPTHWISE_CONV_2D / FULLY_CONNECTED op shapes), file size
  * tensor arena estimate (TFLite: models.quantize.inspect_tflite - peak of simultaneously live
    activation tensors in execution order + margin)
  * PC inference ms median / p95 over N runs, single thread; feature (log-mel) ms median / p95
  * gated fraction on background-only scenarios (engine stats from streaming_metrics.json, else a
    run_offline pass with a DummyClassifier)
  * ESTIMATED MCU latency = MACs / (MAC_per_cycle * f_clk) for the assumptions in MCU_TARGETS
Output: reports/benchmark.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from soundwatch import paths
from soundwatch.config import Config, load_config

# (name, clock Hz, MAC/cycle low, MAC/cycle high, note)
MCU_TARGETS = [
    {
        "target": "Cortex-M4F @ 64 MHz (nRF52840) + CMSIS-NN int8",
        "clock_hz": 64e6,
        "mac_per_cycle": [0.25, 0.5],
        "note": "SIMD SMLAD 2x16-bit MAC/cycle peak; ~0.25-0.5 sustained incl. im2col/requant overhead",
    },
    {
        "target": "ESP32-S3 @ 240 MHz + ESP-NN int8",
        "clock_hz": 240e6,
        "mac_per_cycle": [1.0, 2.0],
        "note": "PIE 128-bit SIMD; ~1-2 MAC/cycle sustained for conv/depthwise kernels",
    },
]


def _stats(ms: list[float]) -> dict:
    a = np.asarray(ms, dtype=np.float64)
    return {
        "median": float(np.median(a)),
        "p95": float(np.percentile(a, 95)),
        "mean": float(a.mean()),
        "n": int(a.size),
    }


def tflite_macs_params(model_path: Path) -> dict:
    """MACs from op shapes and parameter count (constant tensors) of a .tflite model."""
    from soundwatch.models.quantize import _interpreter  # builtin kernels, no delegate

    it = _interpreter(Path(model_path).read_bytes())
    ops = it._get_ops_details()
    tensors = {t["index"]: t for t in it.get_tensor_details()}
    produced = {o for op in ops for o in op["outputs"]}
    inputs = {d["index"] for d in it.get_input_details()}
    macs = 0
    per_op: dict[str, int] = {}
    for op in ops:
        name = op["op_name"]
        ins = [i for i in op["inputs"] if i >= 0]
        if name not in ("CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED") or len(ins) < 2:
            continue
        out = tensors[op["outputs"][0]]["shape"]
        w = tensors[ins[1]]["shape"]
        if name == "CONV_2D":  # filter [oc, kh, kw, ic]
            m = int(np.prod(out)) * int(w[1]) * int(w[2]) * int(w[3])
        elif name == "DEPTHWISE_CONV_2D":  # filter [1, kh, kw, oc*mult]
            m = int(np.prod(out)) * int(w[1]) * int(w[2])
        else:  # FC weights [units, in]
            m = int(np.prod(out)) * int(w[-1])
        macs += m
        per_op[name] = per_op.get(name, 0) + m
    used = {i for op in ops for i in op["inputs"] if i >= 0}
    consts = [
        t
        for i, t in tensors.items()  # constant op inputs (weights, biases, axes); not scratch
        if i in used and i not in produced and i not in inputs and t["shape"].size > 0
    ]
    params = int(sum(int(np.prod(t["shape"])) for t in consts))
    return {
        "macs": int(macs),
        "macs_per_op": per_op,
        "params": params,
        "n_ops": len(ops),
        "ops": sorted({op["op_name"] for op in ops}),
    }


def keras_macs_params(model_path: Path) -> dict:
    import keras

    from soundwatch.models.dscnn import count_macs, count_params

    m = keras.models.load_model(str(model_path), compile=False)
    p = count_params(m)
    return {"macs": int(count_macs(m)), "params": int(p["total"]), "params_detail": p}


def time_inference(model_path: Path, n_frames: int, n_mels: int, runs: int = 1000, warmup: int = 20) -> dict:
    from soundwatch.engine.classifier import load_classifier

    clf = load_classifier(model_path, num_threads=1)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((n_frames, n_mels)).astype(np.float32)
    for _ in range(warmup):
        clf.predict(x)
    ms = []
    for _ in range(runs):
        t0 = time.perf_counter()
        clf.predict(x)
        ms.append((time.perf_counter() - t0) * 1000.0)
    return _stats(ms)


def time_features(cfg: Config, stats: dict | None, runs: int = 1000) -> dict:
    from soundwatch.audio.features import compute_logmel, normalize

    st = stats or {"mean": 0.0, "std": 1.0}
    rng = np.random.default_rng(0)
    w = (0.1 * rng.standard_normal(cfg.audio.window_samples)).astype(np.float32)
    for _ in range(20):
        normalize(compute_logmel(w, cfg.audio), st)
    ms = []
    for _ in range(runs):
        t0 = time.perf_counter()
        normalize(compute_logmel(w, cfg.audio), st)
        ms.append((time.perf_counter() - t0) * 1000.0)
    return _stats(ms)


def mcu_estimates(macs: int, hop_s: float = 0.5) -> list[dict]:
    out = []
    for t in MCU_TARGETS:
        lo, hi = t["mac_per_cycle"]
        ms_best = macs / (hi * t["clock_hz"]) * 1000.0
        ms_worst = macs / (lo * t["clock_hz"]) * 1000.0
        out.append(
            {
                **t,
                "macs": int(macs),
                "latency_ms_best": round(ms_best, 2),
                "latency_ms_worst": round(ms_worst, 2),
                "duty_cycle_worst": round(ms_worst / (hop_s * 1000.0), 4),
                "estimate": True,
            }
        )
    return out


def gated_fraction_background(
    cfg: Config, reports_dir: Path, scen_dir: Path, limit: int | None = None
) -> dict:
    sm = reports_dir / "streaming_metrics.json"
    if sm.exists():
        try:
            d = json.loads(sm.read_text(encoding="utf-8"))
            g = d.get("engine", {}).get("gated_fraction_background")
            if g is not None:
                return {"gated_fraction": g, "source": "streaming_metrics.json (engine stats)"}
        except (OSError, ValueError):
            pass
    files = sorted(scen_dir.glob("bg_*.wav"))[:limit] if scen_dir.exists() else []
    if not files:
        return {"gated_fraction": None, "source": "no background scenarios"}
    from soundwatch.audio.sources import FileSource
    from soundwatch.engine.classifier import DummyClassifier
    from soundwatch.engine.pipeline import run_offline

    hops = gated = 0
    for f in files:
        _, _, st = run_offline(
            cfg, FileSource(f, realtime=False), DummyClassifier(cfg.num_classes), {"mean": 0.0, "std": 1.0}
        )
        hops += st["hops"]
        gated += st["gated_hops"]
    return {
        "gated_fraction": gated / hops if hops else None,
        "source": f"engine run_offline (DummyClassifier) on {len(files)} bg_*.wav",
        "hops": hops,
    }


def default_models(models_dir: Path) -> list[Path]:
    names = [
        "student_int8.tflite",
        "student_float.tflite",
        "student_distill.keras",
        "student_nodistill.keras",
    ]
    return [models_dir / n for n in names]


def benchmark_model(path: Path, cfg: Config, runs: int) -> dict:
    from soundwatch.models.quantize import inspect_tflite

    r: dict = {
        "path": str(path.relative_to(paths.ROOT)) if path.is_relative_to(paths.ROOT) else str(path),
        "format": path.suffix.lstrip("."),
        "size_bytes": int(path.stat().st_size),
    }
    if path.suffix == ".tflite":
        r.update(tflite_macs_params(path))
        info = inspect_tflite(path.read_bytes())
        r["arena_estimate_bytes"] = info["arena_estimate_bytes"]
        r["activation_peak_bytes"] = info["activation_peak_bytes"]
        r["unsupported_ops"] = info["unsupported_ops"]
        r["input"] = info["input"]
    else:
        r.update(keras_macs_params(path))
    r["infer_ms"] = time_inference(path, cfg.audio.n_frames, cfg.audio.n_mels, runs)
    return r


def run(
    cfg: Config,
    models: list[Path],
    runs: int = 1000,
    models_dir: Path | None = None,
    reports_dir: Path | None = None,
    scen_dir: Path | None = None,
    bg_limit: int | None = None,
    verbose: bool = True,
) -> dict:
    from soundwatch.audio.features import load_stats

    models_dir = Path(models_dir) if models_dir else paths.MODELS
    reports_dir = Path(reports_dir) if reports_dir else paths.REPORTS
    scen_dir = Path(scen_dir) if scen_dir else paths.SCENARIOS
    spath = models_dir / "feature_stats.json"
    stats = load_stats(spath) if spath.exists() else None
    res: dict = {"runs": runs, "threads": 1, "models": {}, "missing": []}
    res["feature_ms"] = time_features(cfg, stats, runs)
    if verbose:
        print(f"[benchmark] feature: median {res['feature_ms']['median']:.3f} ms")
    for p in models:
        if not p.exists():
            res["missing"].append(p.name)
            if verbose:
                print(f"[benchmark] skip {p.name}: not found")
            continue
        try:
            r = benchmark_model(p, cfg, runs)
        except Exception as e:  # noqa: BLE001 - keep benchmarking the others
            res["missing"].append(f"{p.name} (failed: {e!r})")
            continue
        r["total_ms_median"] = r["infer_ms"]["median"] + res["feature_ms"]["median"]
        r["mcu_estimates"] = mcu_estimates(r["macs"], cfg.audio.hop_s)
        res["models"][p.stem] = r
        if verbose:
            print(
                f"[benchmark] {p.name}: {r['size_bytes'] / 1024:.1f} KB, params {r['params']}, "
                f"MACs {r['macs'] / 1e6:.2f} M, infer median {r['infer_ms']['median']:.3f} ms"
                + (
                    f", arena ~{r['arena_estimate_bytes'] / 1024:.1f} KB"
                    if "arena_estimate_bytes" in r
                    else ""
                )
            )
    res["gating"] = gated_fraction_background(cfg, reports_dir, scen_dir, bg_limit)
    res["mcu_assumptions"] = {
        "targets": MCU_TARGETS,
        "estimate": True,
        "formula": "latency = MACs / (MAC_per_cycle * f_clk); inference only "
        "(features, framework overhead and memory stalls not included)",
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "benchmark.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    return res


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", action="append", default=None, help="model path (repeatable)")
    parser.add_argument("--runs", type=int, default=1000)


def main(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    try:
        from soundwatch.models.common import setup_tf

        setup_tf(1)
    except Exception:  # noqa: BLE001
        pass
    models = [Path(m) for m in args.model] if args.model else default_models(paths.MODELS)
    res = run(cfg, models, args.runs)
    print(
        f"[benchmark] gated fraction (bg-only): {res['gating']['gated_fraction']} "
        f"-> {paths.REPORTS / 'benchmark.json'}"
    )
    return 0
