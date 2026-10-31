"""WP4: convert the trained student to TFLite (float + full-int8) and check accuracy.

  models/student_float.keras    probability-output Keras model (softmax appended to student_<tag>)
  models/student_float.tflite   float32 reference
  models/student_int8.tflite    full int8 (int8 input/output), the model the engine + firmware run
  models/quant_metrics.json     ops, sizes, arena estimate, quant params, float-keras / float-tflite /
                                int8-tflite metrics on val + test
  assets/models/student_int8.tflite + feature_stats.json   bundled copy for the demo (not with --no-copy)

vn-v2 options: the output activation (softmax / sigmoid) and the front-end come from the run's
feature stats (`params`, see audio.features.ModelMeta); `--stats` picks them (default
models/feature_stats_<tag>.json when it exists, else the config path). `--out-dir DIR` writes all
outputs (and a copy of the stats as feature_stats.json) to DIR instead of artifacts/models.
`--calibrate` fits a temperature on val (one for softmax, one per class for sigmoid), folds it into
the last Dense layer (no new ops) and reports ECE / reliability / saturation before vs after
(`calibration` in quant_metrics.json).
"""

from __future__ import annotations

import argparse
import shutil
import time

import numpy as np

from soundwatch import paths
from soundwatch.config import load_config
from soundwatch.models import common

# Builtin ops that TFLite Micro supports and that this model family may use.
TFLM_OPS = {"CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED", "MEAN", "AVERAGE_POOL_2D", "SOFTMAX",
            "RESHAPE", "RELU", "QUANTIZE", "DEQUANTIZE", "ADD", "MUL", "MAX_POOL_2D", "LOGISTIC"}


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tag", default="distill",
                        help="student run to convert (models/student_<tag>.keras)")
    parser.add_argument("--n-rep", type=int, default=500, help="representative windows (normalized train)")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--no-copy", action="store_true", help="do not copy the int8 model to assets/models")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--stats", default=None,
                        help="feature_stats.json of the run (default models/feature_stats_<tag>.json if "
                        "present, else config model.stats_path)")
    parser.add_argument("--out-dir", default=None,
                        help="write outputs here instead of artifacts/models (implies no default paths)")
    common.add_data_dir_args(parser)
    parser.add_argument("--calibrate", action="store_true",
                        help="fit temperature on val and fold it into the last Dense layer")


def convert(model, rep_data: np.ndarray | None = None) -> bytes:
    """Keras model -> TFLite flatbuffer. With rep_data: full-int8 (int8 in/out), else float32."""
    import tensorflow as tf

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    if rep_data is not None:
        def rep():
            for x in rep_data:
                yield [np.asarray(x, np.float32)[None]]

        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.representative_dataset = rep
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv.inference_input_type = tf.int8
        conv.inference_output_type = tf.int8
    return conv.convert()


def _interpreter(content: bytes):
    """Interpreter with builtin kernels only (no XNNPACK delegate, so ops stay visible and int8
    numerics follow the builtin kernels that TFLite Micro mirrors)."""
    import tensorflow as tf

    it = tf.lite.Interpreter(
        model_content=content, num_threads=1,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    it.allocate_tensors()
    return it


def inspect_tflite(content: bytes) -> dict:
    """Op list, I/O quantization and a tensor-arena estimate (peak bytes of simultaneously live
    activation tensors under the op execution order, 16-byte aligned, + 25% / 4 KB margin for TFLM
    runtime structs). Weights live in flash (the flatbuffer) and are not counted."""
    it = _interpreter(content)
    ops = it._get_ops_details()
    tensors = {t["index"]: t for t in it.get_tensor_details()}
    produced = {o for op in ops for o in op["outputs"]}
    inputs = [d["index"] for d in it.get_input_details()]
    outputs = [d["index"] for d in it.get_output_details()]

    def nbytes(i: int) -> int:
        t = tensors[i]
        n = int(np.prod(t["shape"])) * np.dtype(t["dtype"]).itemsize
        return (n + 15) // 16 * 16

    act = set(produced) | set(inputs)
    first = {i: -1 for i in inputs}
    last: dict[int, int] = {}
    for k, op in enumerate(ops):
        for o in op["outputs"]:
            first.setdefault(o, k)
        for i in op["inputs"]:
            if i in act:
                last[i] = k
    for o in outputs:
        last[o] = len(ops)
    peak, peak_op = 0, -1
    for k in range(len(ops)):
        live = sum(nbytes(i) for i in act if first.get(i, 99999) <= k <= last.get(i, first.get(i, 0)))
        if live > peak:
            peak, peak_op = live, k
    names = [op["op_name"] for op in ops]
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    return {
        "ops": names,
        "op_set": sorted(set(names)),
        "unsupported_ops": sorted(set(names) - TFLM_OPS),
        "size_bytes": len(content),
        "activation_peak_bytes": int(peak),
        "activation_peak_op": names[peak_op] if peak_op >= 0 else None,
        "arena_estimate_bytes": int(max(peak * 1.25, peak + 4096)),
        "input": {"shape": [int(v) for v in inp["shape"]], "dtype": np.dtype(inp["dtype"]).name,
                  "scale": float(inp["quantization"][0]), "zero_point": int(inp["quantization"][1])},
        "output": {"shape": [int(v) for v in out["shape"]], "dtype": np.dtype(out["dtype"]).name,
                   "scale": float(out["quantization"][0]), "zero_point": int(out["quantization"][1])},
    }


def tflite_predict(content: bytes, x: np.ndarray) -> np.ndarray:
    """Run every sample of x (N, ...) float32 through the model; returns dequantized outputs (N, C)."""
    it = _interpreter(content)
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    in_s, in_z = inp["quantization"]
    out_s, out_z = out["quantization"]
    quant_in = np.issubdtype(inp["dtype"], np.integer)
    quant_out = np.issubdtype(out["dtype"], np.integer)
    res = np.empty((len(x), int(out["shape"][-1])), np.float32)
    for i in range(len(x)):
        xi = np.asarray(x[i], np.float32).reshape(inp["shape"])
        if quant_in:
            info = np.iinfo(inp["dtype"])
            xi = np.clip(np.round(xi / in_s + in_z), info.min, info.max).astype(inp["dtype"])
        it.set_tensor(inp["index"], xi)
        it.invoke()
        y = it.get_tensor(out["index"]).reshape(-1)
        res[i] = (y.astype(np.float32) - out_z) * out_s if quant_out else y
    return res


def _resolve_stats(args, cfg):
    from pathlib import Path

    if getattr(args, "stats", None):
        return Path(args.stats)
    per_run = paths.MODELS / f"feature_stats_{args.tag}.json"
    return per_run if per_run.exists() else cfg.model.stats_path


def main(args: argparse.Namespace) -> int:
    from pathlib import Path

    import keras

    from soundwatch.audio.features import load_stats, logmel_to_features, model_meta, normalize, save_stats
    from soundwatch.eval import calibration as cal
    from soundwatch.models.dscnn import count_macs, count_params, with_output

    common.apply_data_dirs(args)
    cfg = load_config(args.config)
    common.setup_tf(args.threads, seed=args.seed)
    src = paths.MODELS / f"student_{args.tag}.keras"
    if not src.exists():
        raise FileNotFoundError(f"{src} not found (run train-student first)")
    stats_path = _resolve_stats(args, cfg)
    stats = load_stats(stats_path)
    meta = model_meta(stats)
    out_dir = Path(args.out_dir) if getattr(args, "out_dir", None) else None
    dst = out_dir if out_dir is not None else paths.MODELS
    dst.mkdir(parents=True, exist_ok=True)
    base = keras.models.load_model(src, compile=False)

    data = {s: common.load_split(s) for s in common.SPLITS}

    def feats_of(s):
        lm = common.cached_logmel(s, cfg.audio, data[s]["wave"])
        return normalize(np.concatenate([logmel_to_features(lm[i:i + 4096], meta, cfg.audio)
                                         for i in range(0, len(lm), 4096)]), stats)[..., None]

    feats = {s: feats_of(s) for s in common.SPLITS}
    calib: dict | None = None
    if getattr(args, "calibrate", False):
        z = {s: base.predict(feats[s], batch_size=512, verbose=0) for s in ("val", "test")}
        temp = cal.fit_temperature(z["val"], data["val"]["label"], meta.output)
        calib = {"output": meta.output, "temperature": temp.tolist(), "fit_split": "val",
                 "float_before": {}, "float_after": {}}
        for s in ("val", "test"):
            y = data[s]["label"]
            calib["float_before"][s] = cal.calibration_report(cal.probs_from_logits(z[s], meta.output), y)
            calib["float_after"][s] = cal.calibration_report(
                cal.probs_from_logits(z[s], meta.output, temp), y)
            calib["float_before"][s]["nll"] = cal.nll(z[s], y, meta.output)
            calib["float_after"][s]["nll"] = cal.nll(z[s], y, meta.output, temp)
        # int8 "before" for the report: convert the uncalibrated model once
        rng0 = np.random.default_rng(args.seed)
        rep0 = feats["train"][rng0.choice(len(feats["train"]), min(args.n_rep, len(feats["train"])),
                                          replace=False)]
        int8_before = convert(with_output(base, meta.output), rep0)
        calib["int8_before"] = {s: cal.calibration_report(tflite_predict(int8_before, feats[s]),
                                                          data[s]["label"]) for s in ("val", "test")}
        cal.fold_temperature(base, temp)
        print(f"[quantize] calibration: temperature {np.round(temp, 3).tolist()} folded into the last Dense")
    probs_model = with_output(base, meta.output)
    probs_model.save(dst / "student_float.keras")

    rng = np.random.default_rng(args.seed)
    rep = feats["train"][rng.choice(len(feats["train"]), min(args.n_rep, len(feats["train"])), replace=False)]

    t0 = time.time()
    float_tfl = convert(probs_model)
    int8_tfl = convert(probs_model, rep)
    (dst / "student_float.tflite").write_bytes(float_tfl)
    int8_path = (out_dir / "student_int8.tflite") if out_dir is not None else cfg.model.tflite_path
    int8_path.parent.mkdir(parents=True, exist_ok=True)
    int8_path.write_bytes(int8_tfl)
    if out_dir is not None:
        save_stats(stats, out_dir / "feature_stats.json")
    elif Path(stats_path).resolve() != Path(cfg.model.stats_path).resolve():
        # keep the int8 model and its stats file a pair (vn-v3): the run's stats go next to the model
        save_stats(stats, cfg.model.stats_path)
    print(f"[quantize] converted in {time.time() - t0:.1f}s: float {len(float_tfl) / 1024:.1f} KB, "
          f"int8 {len(int8_tfl) / 1024:.1f} KB -> {int8_path}")

    info = inspect_tflite(int8_tfl)
    print(f"[quantize] int8 ops: {' '.join(info['ops'])}")
    print(f"[quantize] op set {info['op_set']}; not TFLM-builtin: {info['unsupported_ops'] or 'none'}")
    print(f"[quantize] activation peak {info['activation_peak_bytes'] / 1024:.1f} KB "
          f"(at {info['activation_peak_op']}), arena estimate {info['arena_estimate_bytes'] / 1024:.1f} KB")
    print(f"[quantize] input scale={info['input']['scale']:.6g} zp={info['input']['zero_point']}; "
          f"output scale={info['output']['scale']:.6g} zp={info['output']['zero_point']}")

    result: dict = {
        "tag": args.tag, "source": src.name, "n_representative": len(rep),
        "files": {"keras_float": "student_float.keras", "tflite_float": "student_float.tflite",
                  "tflite_int8": int8_path.name},
        "size_bytes": {"tflite_float": len(float_tfl), "tflite_int8": len(int8_tfl)},
        "params": count_params(base), "macs": count_macs(base), "int8": info,
    }
    if not meta.is_default:
        result["frontend"] = {"frontend": meta.frontend, "output": meta.output, "stats": str(stats_path)}
    for s in ("val", "test"):
        y = data[s]["label"]
        preds = {
            "keras_float": probs_model.predict(feats[s], batch_size=512, verbose=0),
            "tflite_float": tflite_predict(float_tfl, feats[s]),
            "tflite_int8": tflite_predict(int8_tfl, feats[s]),
        }
        result[s] = {k: common.classification_metrics(y, p.argmax(1), cfg) for k, p in preds.items()}
        kf, q = result[s]["keras_float"], result[s]["tflite_int8"]
        result[s]["int8_drop_accuracy_pts"] = round(100 * (kf["accuracy"] - q["accuracy"]), 3)
        result[s]["int8_drop_macro_f1_pts"] = round(100 * (kf["macro_f1"] - q["macro_f1"]), 3)
        result[s]["int8_vs_float_top1_agreement"] = float(np.mean(
            preds["tflite_int8"].argmax(1) == preds["keras_float"].argmax(1)))
        result[s]["int8_vs_float_max_abs_prob_diff"] = float(np.abs(
            preds["tflite_int8"] - preds["keras_float"]).max())
        if calib is not None:
            calib.setdefault("int8_after", {})[s] = cal.calibration_report(preds["tflite_int8"], y)
        for k in preds:
            print(common.format_metrics_row(f"{s} {k}", result[s][k], cfg))
        print(f"[quantize] {s}: int8 drop acc {result[s]['int8_drop_accuracy_pts']:+.2f} pts, "
              f"macro-F1 {result[s]['int8_drop_macro_f1_pts']:+.2f} pts, "
              f"agreement {result[s]['int8_vs_float_top1_agreement']:.4f}")
    if calib is not None:
        result["calibration"] = calib
        for s in ("val", "test"):
            b, a = calib["int8_before"][s], calib["int8_after"][s]
            print(f"[quantize] {s} int8 calibration: ECE top {b['ece_top_label']:.4f} -> "
                  f"{a['ece_top_label']:.4f}, class-wise {b['ece_classwise_mean']:.4f} -> "
                  f"{a['ece_classwise_mean']:.4f}, saturated windows {b['saturated_windows']} -> "
                  f"{a['saturated_windows']} / {a['n']}")
    common.write_json(result, dst / "quant_metrics.json")

    if not args.no_copy and out_dir is None:
        for a, b in ((int8_path, cfg.model.bundled_tflite_path),
                     (cfg.model.stats_path, cfg.model.bundled_stats_path)):
            b.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(a, b)
        print(f"[quantize] copied int8 model + feature stats to {cfg.model.bundled_tflite_path.parent}")
    return 0
