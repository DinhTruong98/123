"""Knowledge-distillation loss (plan WP3).

    loss = alpha * CE(y, s) + (1 - alpha) * T^2 * KL(softmax(t / T) || softmax(s / T))

The student is trained on logits. Keras losses only receive (y_true, y_pred), so the teacher logits
travel inside y_true: y_true = concat([one_hot(y) (C), teacher_logits (C)], axis=-1).
Augmented copies of a training window reuse the teacher logits of their clean source window.
"""

from __future__ import annotations

import numpy as np


def pack_targets(labels: np.ndarray, num_classes: int,
                 teacher_logits: np.ndarray | None = None) -> np.ndarray:
    """(N,) int labels [+ (N, C) teacher logits] -> (N, 2C) float32 targets for `kd_loss`.
    Without teacher logits the second half is zeros (only usable with alpha = 1)."""
    y = np.eye(num_classes, dtype=np.float32)[labels]
    t = np.zeros_like(y) if teacher_logits is None else np.asarray(teacher_logits, dtype=np.float32)
    return np.concatenate([y, t], axis=-1)


def kd_loss(num_classes: int, alpha: float = 0.5, temperature: float = 4.0, label_smoothing: float = 0.0):
    """Returns a Keras loss fn(y_true (N, 2C), logits (N, C)) -> (N,) per-sample loss.
    `label_smoothing` applies to the hard-label CE term only."""
    import keras
    from keras import ops

    a, temp = float(alpha), float(temperature)

    def loss(y_true, y_pred):
        y = y_true[:, :num_classes]
        ce = keras.losses.categorical_crossentropy(y, y_pred, from_logits=True,
                                                   label_smoothing=label_smoothing)
        if a >= 1.0:
            return ce
        t = y_true[:, num_classes:]
        log_pt = ops.log_softmax(t / temp, axis=-1)
        log_ps = ops.log_softmax(y_pred / temp, axis=-1)
        kl = ops.sum(ops.exp(log_pt) * (log_pt - log_ps), axis=-1)
        return a * ce + (1.0 - a) * temp * temp * kl

    loss.__name__ = f"kd_a{a:g}_t{temp:g}_ls{label_smoothing:g}"
    return loss


def bce_loss():
    """Multi-label loss for the sigmoid head: mean binary cross-entropy over the C outputs.
    y_true (N, C) holds the (soft) targets built by the data pipeline (hard union of the mixed
    labels, optionally blended with teacher probabilities)."""
    import keras

    def loss(y_true, y_pred):
        return keras.losses.binary_crossentropy(y_true, y_pred, from_logits=True)

    loss.__name__ = "bce_multilabel"
    return loss


def kd_loss_numpy(labels: np.ndarray, student_logits: np.ndarray, teacher_logits: np.ndarray,
                  alpha: float = 0.5, temperature: float = 4.0) -> np.ndarray:  # fmt: skip
    """Reference implementation (tests)."""

    def log_softmax(z: np.ndarray) -> np.ndarray:
        z = z - z.max(axis=-1, keepdims=True)
        return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))

    ce = -log_softmax(student_logits)[np.arange(len(labels)), labels]
    lt, ls = log_softmax(teacher_logits / temperature), log_softmax(student_logits / temperature)
    kl = (np.exp(lt) * (lt - ls)).sum(-1)
    return alpha * ce + (1 - alpha) * temperature**2 * kl
