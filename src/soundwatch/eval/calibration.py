"""Probability calibration (vn-v2): temperature fit on val, folding into the last Dense layer,
reliability / ECE / saturation metrics.

Softmax models: one temperature T, p = softmax(z / T). Sigmoid models: one temperature per class,
p_c = sigmoid(z_c / T_c), fitted against the one-vs-rest targets (y == c). Both are folded into the
final ``Dense`` (``logits``) layer, W' = W / T, b' = b / T, so the exported graph and its TFLite ops
do not change.

Metrics (all on probabilities, e.g. the dequantized int8 outputs):
  * top-label ECE: bins of max_c p_c (15 equal-width bins), |accuracy - mean confidence| weighted by
    the bin size; comparable between softmax and sigmoid models.
  * class-wise ECE: mean over classes of the ECE of p_c against (y == c) (the multi-label view).
  * saturation: fraction of windows whose max output reaches 255/256 (the int8 ceiling with scale
    1/256, zero point -128) - the over-confidence that forced thresholds of 0.99 before.
"""

from __future__ import annotations

import numpy as np

SATURATED = 255.0 / 256.0
N_BINS = 15


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def probs_from_logits(logits: np.ndarray, output: str, temperature=1.0) -> np.ndarray:
    z = np.asarray(logits, np.float64) / np.asarray(temperature, np.float64)
    return _sigmoid(z) if output == "sigmoid" else _softmax(z)


def nll(logits: np.ndarray, labels: np.ndarray, output: str, temperature=1.0) -> float:
    """Mean negative log-likelihood (softmax) or mean per-class BCE (sigmoid)."""
    z = np.asarray(logits, np.float64) / np.asarray(temperature, np.float64)
    y = np.asarray(labels, int)
    if output == "sigmoid":
        t = np.eye(z.shape[1])[y]
        # BCE with logits, numerically stable
        return float(np.mean(np.maximum(z, 0) - z * t + np.log1p(np.exp(-np.abs(z)))))
    z = z - z.max(axis=1, keepdims=True)
    lse = np.log(np.exp(z).sum(axis=1))
    return float(np.mean(lse - z[np.arange(len(y)), y]))


def fit_temperature(logits: np.ndarray, labels: np.ndarray, output: str,
                    bounds: tuple[float, float] = (0.05, 20.0)) -> np.ndarray:
    """Temperature minimizing the val NLL / BCE: shape (1,) for softmax, (C,) for sigmoid."""
    from scipy.optimize import minimize_scalar

    z = np.asarray(logits, np.float64)
    y = np.asarray(labels, int)
    lo, hi = np.log(bounds[0]), np.log(bounds[1])
    if output != "sigmoid":
        r = minimize_scalar(lambda lt: nll(z, y, "softmax", np.exp(lt)), bounds=(lo, hi), method="bounded")
        return np.array([float(np.exp(r.x))])
    temps = np.ones(z.shape[1])
    for c in range(z.shape[1]):
        t = (y == c).astype(np.float64)
        zc = z[:, c]

        def f(lt, zc=zc, t=t):
            u = zc / np.exp(lt)
            return float(np.mean(np.maximum(u, 0) - u * t + np.log1p(np.exp(-np.abs(u)))))

        temps[c] = float(np.exp(minimize_scalar(f, bounds=(lo, hi), method="bounded").x))
    return temps


def fold_temperature(model, temperature) -> None:
    """Divide the kernel and bias of the model's last Dense layer by T (in place)."""
    from keras import layers

    dense = [ly for ly in model.layers if isinstance(ly, layers.Dense)]
    if not dense:
        raise ValueError("model has no Dense layer to fold the temperature into")
    d = dense[-1]
    w, b = d.get_weights()
    t = np.broadcast_to(np.asarray(temperature, np.float64), b.shape)
    d.set_weights([(w / t[None, :]).astype(w.dtype), (b / t).astype(b.dtype)])


def _ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = N_BINS) -> tuple[float, list[dict]]:
    conf = np.asarray(conf, np.float64)
    correct = np.asarray(correct, np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    ece, rows = 0.0, []
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        acc, cf = float(correct[m].mean()), float(conf[m].mean())
        ece += m.mean() * abs(acc - cf)
        rows.append({"lo": float(edges[b]), "hi": float(edges[b + 1]), "n": int(m.sum()),
                     "confidence": cf, "accuracy": acc})
    return float(ece), rows


def calibration_report(probs: np.ndarray, labels: np.ndarray, n_bins: int = N_BINS) -> dict:
    """Top-label ECE + reliability bins, class-wise ECE, saturation counts."""
    p = np.asarray(probs, np.float64)
    y = np.asarray(labels, int)
    conf = p.max(axis=1)
    ece_top, rel = _ece(conf, p.argmax(axis=1) == y, n_bins)
    cw = []
    for c in range(p.shape[1]):
        cw.append(_ece(p[:, c], y == c, n_bins)[0])
    sat = conf >= SATURATED - 1e-6
    sat_true = p[np.arange(len(y)), y] >= SATURATED - 1e-6
    return {
        "n": int(len(y)),
        "ece_top_label": ece_top,
        "ece_classwise_mean": float(np.mean(cw)),
        "ece_classwise": [float(v) for v in cw],
        "saturated_windows": int(sat.sum()),
        "saturated_fraction": float(sat.mean()) if len(y) else 0.0,
        "saturated_true_class_fraction": float(sat_true.mean()) if len(y) else 0.0,
        "mean_confidence": float(conf.mean()) if len(y) else 0.0,
        "accuracy": float((p.argmax(axis=1) == y).mean()) if len(y) else 0.0,
        "reliability": rel,
    }
