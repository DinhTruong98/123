"""MobileNetV2-style inverted-residual student (ESP32-S3 budget variant, WP3).

TFLite-Micro ops only: CONV_2D, DEPTHWISE_CONV_2D, ADD (residuals), MEAN, FULLY_CONNECTED, SOFTMAX
(ReLU is fused into the convs). Like `dscnn.build_dscnn`, returns a logits model; export with
`dscnn.with_softmax`. MACs: `dscnn.count_macs` (residual ADDs are not counted).
"""

from __future__ import annotations

# (expansion t, out channels c, repeats n, first stride s) — input (49, 40) -> stem s2 -> 25x20
# ~19.8 M MACs, ~390 k weights (~400 KB int8) at width_mult 1.0, head 256
DEFAULT_BLOCKS: tuple[tuple[int, int, int, int], ...] = (
    (1, 16, 1, 1),
    (6, 24, 2, 2),   # -> 13x10
    (6, 32, 3, 1),
    (6, 64, 3, 2),   # -> 7x5
    (6, 96, 2, 1),
)


def _round(c: float, divisor: int = 8) -> int:
    return max(divisor, int(c + divisor / 2) // divisor * divisor)


def build_mbv2(input_shape: tuple[int, int, int] = (49, 40, 1), num_classes: int = 8, width_mult: float = 1.0,
               blocks: tuple[tuple[int, int, int, int], ...] = DEFAULT_BLOCKS, stem: int = 32,
               head: int = 256, dropout: float = 0.3, bn_momentum: float = 0.9,
               name: str | None = None):  # fmt: skip
    import keras
    from keras import layers

    def bn(n: str):
        return layers.BatchNormalization(momentum=bn_momentum, name=n)

    inp = keras.Input(shape=input_shape, name="logmel")
    x = layers.Conv2D(_round(stem * width_mult), 3, strides=2, padding="same", use_bias=False,
                      name="stem")(inp)
    x = layers.ReLU(name="stem_relu")(bn("stem_bn")(x))
    k = 0
    for t, c, n, s in blocks:
        out_c = _round(c * width_mult)
        for r in range(n):
            k += 1
            stride = s if r == 0 else 1
            in_c = int(x.shape[-1])
            y = x
            if t != 1:
                y = layers.Conv2D(in_c * t, 1, use_bias=False, name=f"b{k}_expand")(y)
                y = layers.ReLU(name=f"b{k}_expand_relu")(bn(f"b{k}_expand_bn")(y))
            y = layers.DepthwiseConv2D(3, strides=stride, padding="same", use_bias=False, name=f"b{k}_dw")(y)
            y = layers.ReLU(name=f"b{k}_dw_relu")(bn(f"b{k}_dw_bn")(y))
            y = layers.Conv2D(out_c, 1, use_bias=False, name=f"b{k}_project")(y)
            y = bn(f"b{k}_project_bn")(y)
            if stride == 1 and in_c == out_c:
                y = layers.Add(name=f"b{k}_add")([x, y])
            x = y
    x = layers.Conv2D(_round(head * width_mult), 1, use_bias=False, name="head")(x)
    x = layers.ReLU(name="head_relu")(bn("head_bn")(x))
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    logits = layers.Dense(num_classes, name="logits")(x)
    return keras.Model(inp, logits, name=name or f"mbv2_w{width_mult:g}")
