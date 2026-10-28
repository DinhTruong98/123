import numpy as np
import pytest

from soundwatch.engine.classifier import (
    DummyClassifier,
    KerasClassifier,
    TFLiteClassifier,
    as_probs,
    load_classifier,
)

N_CLASSES = 8
SHAPE = (49, 40, 1)


def _tiny_dscnn(softmax: bool = True):
    import keras
    from keras import layers

    keras.utils.set_random_seed(0)
    inp = keras.Input(SHAPE)
    x = layers.Conv2D(8, 3, strides=2, padding="same", use_bias=False)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.DepthwiseConv2D(3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(16, 1, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling2D()(x)
    out = layers.Dense(N_CLASSES, activation="softmax" if softmax else None)(x)
    return keras.Model(inp, out)


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    import tensorflow as tf

    d = tmp_path_factory.mktemp("models")
    rng = np.random.default_rng(0)
    rep = rng.standard_normal((16, *SHAPE)).astype(np.float32)

    model = _tiny_dscnn()
    keras_path = d / "tiny.keras"
    model.save(keras_path)

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = lambda: ([r[None]] for r in rep)
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    int8_path = d / "tiny_int8.tflite"
    int8_path.write_bytes(conv.convert())

    logits_model = _tiny_dscnn(softmax=False)
    logits_path = d / "tiny_logits.tflite"
    logits_path.write_bytes(tf.lite.TFLiteConverter.from_keras_model(logits_model).convert())
    return {"keras": keras_path, "int8": int8_path, "logits": logits_path, "model": model,
            "logits_model": logits_model}


def test_tflite_int8_matches_keras(models):
    clf = TFLiteClassifier(models["int8"])
    assert clf.num_classes == N_CLASSES and clf.is_quantized
    assert clf.input_shape == (1, *SHAPE)
    rng = np.random.default_rng(1)
    ref = KerasClassifier(models["keras"])
    assert ref.num_classes == N_CLASSES
    for _ in range(5):
        x = rng.standard_normal((49, 40)).astype(np.float32)
        p = clf.predict(x)
        assert p.shape == (N_CLASSES,) and p.dtype == np.float32
        assert abs(p.sum() - 1) < 0.05 and (p >= 0).all()
        assert clf.last_infer_ms > 0
        pk = ref.predict(x)
        np.testing.assert_allclose(pk, np.asarray(models["model"](x[None, ..., None])).ravel(), atol=1e-5)
        assert np.abs(p - pk).max() < 0.1


def test_tflite_float_logits_get_softmax(models):
    clf = TFLiteClassifier(models["logits"])
    assert not clf.is_quantized
    x = np.random.default_rng(2).standard_normal((49, 40)).astype(np.float32)
    p = clf.predict(x)
    logits = np.asarray(models["logits_model"](x[None, ..., None])).ravel()
    e = np.exp(logits - logits.max())
    np.testing.assert_allclose(p, e / e.sum(), atol=1e-4)
    assert isinstance(load_classifier(models["int8"]), TFLiteClassifier)
    with pytest.raises(ValueError):
        clf.predict(np.zeros((10, 40), np.float32))


def test_as_probs():
    p = np.array([0.2, 0.8], np.float32)
    np.testing.assert_array_equal(as_probs(p), p)
    q = as_probs(np.array([1.0, 3.0]))
    assert q.sum() == pytest.approx(1.0) and q[1] > q[0]


def test_dummy_classifier():
    clf = DummyClassifier(8, [(1.0, 2.0, 3)], confidence=0.86)
    clf.set_time(0.5)
    p = clf.predict(np.zeros((49, 40)))
    assert p.argmax() == 7 and p[7] == pytest.approx(0.86)
    clf.set_time(1.0)
    p = clf.predict(np.zeros((49, 40)))
    assert p.argmax() == 3 and p.sum() == pytest.approx(1.0)
    assert p[0] == pytest.approx(0.14 / 7)
    clf.set_time(2.0)
    assert clf.predict(None).argmax() == 7
    with pytest.raises(ValueError):
        DummyClassifier(8, [(0, 1, 8)])
