"""WP3: DS-CNN shapes / MACs / params, KD loss, teacher head."""

from __future__ import annotations

import numpy as np
import pytest

keras = pytest.importorskip("keras")

from soundwatch.models.distill import kd_loss, kd_loss_numpy, pack_targets  # noqa: E402
from soundwatch.models.dscnn import build_dscnn, count_macs, count_params, with_softmax  # noqa: E402


def test_dscnn_default_shapes_and_macs():
    m = build_dscnn()
    assert m.input_shape == (None, 49, 40, 1)
    assert m.output_shape == (None, 8)
    # conv0 25x20x64x(10*4) + 4 x (DW 25x20x64x9 + PW 25x20x64x64) + dense 64x8
    assert count_macs(m) == 25 * 20 * 64 * 40 + 4 * (25 * 20 * 64 * 9 + 25 * 20 * 64 * 64) + 64 * 8
    p = count_params(m)
    assert p["deployed"] == (10 * 4 * 64 + 64) + 4 * (9 * 64 + 64 + 64 * 64 + 64) + (64 * 8 + 8)


def test_dscnn_tiny_hand_macs():
    m = build_dscnn((10, 8, 1), num_classes=3, width=4, n_blocks=1)
    # conv0 -> (5, 4, 4): 5*4*4*40 = 3200; dw 5*4*4*9 = 720; pw 5*4*4*4 = 320; dense 4*3 = 12
    assert count_macs(m) == 3200 + 720 + 320 + 12


def test_with_softmax_outputs_probabilities():
    m = build_dscnn((49, 40, 1), num_classes=8, width=8, n_blocks=1)
    pm = with_softmax(m)
    x = np.random.default_rng(0).normal(size=(3, 49, 40, 1)).astype(np.float32)
    p = np.asarray(pm(x, training=False))
    assert p.shape == (3, 8)
    np.testing.assert_allclose(p.sum(1), 1.0, atol=1e-5)
    logits = np.asarray(m(x, training=False))
    np.testing.assert_array_equal(p.argmax(1), logits.argmax(1))


def test_kd_loss_alpha1_equals_ce():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 8, size=16)
    s = rng.normal(size=(16, 8)).astype(np.float32)
    t = rng.normal(size=(16, 8)).astype(np.float32) * 3
    packed = pack_targets(y, 8, t)
    ce = np.asarray(keras.losses.categorical_crossentropy(np.eye(8)[y], s, from_logits=True))
    got = np.asarray(kd_loss(8, alpha=1.0)(packed, s))
    np.testing.assert_allclose(got, ce, rtol=1e-5, atol=1e-6)
    got_half = np.asarray(kd_loss(8, alpha=0.5, temperature=4.0)(packed, s))
    np.testing.assert_allclose(got_half, kd_loss_numpy(y, s, t, 0.5, 4.0), rtol=1e-4, atol=1e-5)
    # identical teacher/student -> KL term vanishes
    same = np.asarray(kd_loss(8, alpha=0.0)(pack_targets(y, 8, s), s))
    np.testing.assert_allclose(same, 0.0, atol=1e-5)


def test_teacher_head_shapes():
    from soundwatch.models.teacher import build_head

    m = build_head(8)
    assert m.input_shape == (None, 1024)
    assert m.output_shape == (None, 8)
