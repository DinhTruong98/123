"""DS-CNN student (plan WP3). Only TFLite-Micro-friendly ops: Conv2D, DepthwiseConv2D, ReLU,
Mean (GlobalAveragePooling2D), FullyConnected, Softmax. Convs have no bias so BN folds into them.

`build_dscnn` returns the *training* model (logits); `with_softmax` wraps it for export.
"""

from __future__ import annotations

import numpy as np


def build_dscnn(input_shape: tuple[int, int, int] = (49, 40, 1), num_classes: int = 8, width: int = 64,
                n_blocks: int = 4, dropout: float = 0.3, first_kernel: tuple[int, int] = (10, 4),
                first_stride: tuple[int, int] = (2, 2), name: str | None = None,
                bn_momentum: float = 0.9, block_widths: list[int] | None = None,
                block_strides: list[int] | None = None):  # fmt: skip
    """Conv(width,(10,4),s2) -> BN -> ReLU -> n_blocks x [DW3x3 -> BN -> ReLU -> PW1x1 -> BN -> ReLU]
    -> GAP -> Dropout -> Dense(num_classes) logits.

    BN momentum 0.9 (not Keras' 0.99): moving stats must track within a few hundred steps, otherwise
    eval-mode outputs are garbage early in training and early stopping misjudges the model.

    Optional `block_widths` / `block_strides` (one entry per block) give a pyramid variant, e.g.
    widths [64,128,128,128] + strides [1,2,1,1]: wider late blocks at 13x10 resolution for fewer MACs.
    Default (None) is the plan architecture: every block `width` channels, stride 1."""
    import keras
    from keras import layers

    inp = keras.Input(shape=input_shape, name="logmel")
    x = layers.Conv2D(width, first_kernel, strides=first_stride, padding="same", use_bias=False,
                      name="conv0")(inp)
    x = layers.BatchNormalization(momentum=bn_momentum, name="conv0_bn")(x)
    x = layers.ReLU(name="conv0_relu")(x)
    widths = list(block_widths) if block_widths else [width] * n_blocks
    strides = list(block_strides) if block_strides else [1] * len(widths)
    if len(widths) != len(strides):
        raise ValueError("block_widths and block_strides must have the same length")
    for i, (w, s) in enumerate(zip(widths, strides, strict=True)):
        x = layers.DepthwiseConv2D(3, strides=s, padding="same", use_bias=False, name=f"dw{i + 1}")(x)
        x = layers.BatchNormalization(momentum=bn_momentum, name=f"dw{i + 1}_bn")(x)
        x = layers.ReLU(name=f"dw{i + 1}_relu")(x)
        x = layers.Conv2D(w, 1, padding="same", use_bias=False, name=f"pw{i + 1}")(x)
        x = layers.BatchNormalization(momentum=bn_momentum, name=f"pw{i + 1}_bn")(x)
        x = layers.ReLU(name=f"pw{i + 1}_relu")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    logits = layers.Dense(num_classes, name="logits")(x)
    default = f"dscnn_w{width}_b{len(widths)}"
    if block_widths is not None:
        default += "_" + "-".join(map(str, widths))
    return keras.Model(inp, logits, name=name or default)


def with_softmax(model):
    """Same layers/weights, output = softmax(logits). This is the model that gets exported."""
    import keras
    from keras import layers

    probs = layers.Softmax(name="probs")(model.output)
    return keras.Model(model.input, probs, name=model.name + "_probs")


def _shape(t) -> tuple[int, ...]:
    return tuple(int(v) for v in t.shape[1:])


def count_macs(model) -> int:
    """Analytic multiply-accumulates per inference for Conv2D / DepthwiseConv2D / Dense (BN folded,
    ReLU / pooling / softmax ignored)."""
    import keras
    from keras import layers

    total = 0
    for layer in model.layers:
        if isinstance(layer, keras.Model):  # nested sub-model
            total += count_macs(layer)
        elif isinstance(layer, layers.DepthwiseConv2D):
            kh, kw = layer.kernel_size
            oh, ow, oc = _shape(layer.output)
            total += oh * ow * oc * kh * kw
        elif isinstance(layer, layers.Conv2D):
            kh, kw = layer.kernel_size
            in_c = _shape(layer.input)[-1]
            oh, ow, oc = _shape(layer.output)
            total += oh * ow * oc * kh * kw * in_c // layer.groups
        elif isinstance(layer, layers.Dense):
            total += int(np.prod(_shape(layer.input))) * layer.units
    return int(total)


def count_params(model) -> dict[str, int]:
    """Parameter counts. `deployed` = after BN folding (conv kernels + one bias per conv channel + dense)."""
    from keras import layers

    trainable = int(sum(np.prod(w.shape) for w in model.trainable_weights))
    total = int(sum(np.prod(w.shape) for w in model.weights))
    deployed = 0
    for layer in model.layers:
        if isinstance(layer, (layers.Conv2D, layers.DepthwiseConv2D)):
            k = layer.weights[0]
            deployed += int(np.prod(k.shape)) + int(_shape(layer.output)[-1])
        elif isinstance(layer, layers.Dense):
            deployed += int(sum(np.prod(w.shape) for w in layer.weights))
    return {"total": total, "trainable": trainable, "deployed": deployed}
