"""WP3: train-student --quick on a tiny synthetic dataset (all paths redirected to tmp)."""

from __future__ import annotations

import argparse

import numpy as np
import pytest

pytest.importorskip("keras")

from soundwatch import paths  # noqa: E402


def _make_split(rng: np.random.Generator, n: int, split: str) -> dict:
    t = np.arange(16000) / 16000.0
    labels = np.arange(n) % 8
    waves = np.stack([0.3 * np.sin(2 * np.pi * (200 + 400 * c) * t) for c in labels])
    waves = waves + 0.01 * rng.normal(size=waves.shape)
    return {"wave": waves.astype(np.float16), "label": labels.astype(np.int16),
            "source_clip": np.array([f"{split}_{i // 2}" for i in range(n)])}


@pytest.fixture
def tiny_env(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    win = tmp_path / "windows"
    win.mkdir()
    for s, n in (("train", 48), ("val", 24), ("test", 24)):
        np.savez(win / f"{s}.npz", **_make_split(rng, n, s))
    for name, sub in (("WINDOWS", "windows"), ("CACHE", "cache"), ("MODELS", "models"), ("LOGS", "logs"),
                      ("NOISE", "noise")):
        monkeypatch.setattr(paths, name, tmp_path / sub)
    (tmp_path / "cache").mkdir()
    return tmp_path


def _args(argv: list[str]) -> argparse.Namespace:
    from soundwatch.models import train

    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    train.add_arguments(p)
    return p.parse_args(argv)


def test_train_student_quick_nodistill(tiny_env):
    from soundwatch.models import train

    m = train.run(_args(["--quick", "--no-distill", "--width", "8", "--blocks", "1", "--batch-size", "16"]))
    assert m["epochs_run"] == 1
    assert (tiny_env / "models" / "student_nodistill_quick.keras").exists()
    assert (tiny_env / "models" / "student_nodistill_quick_metrics.json").exists()
    assert (tiny_env / "models" / "feature_stats_quick.json").exists()
    assert (tiny_env / "logs" / "nodistill_quick" / "epochs.csv").exists()
    assert set(m["test"]) >= {"accuracy", "macro_f1", "per_class", "danger_recall"}


def test_train_student_quick_distill(tiny_env):
    from soundwatch.models import train

    logits = np.random.default_rng(1).normal(size=(48, 8)).astype(np.float32)
    from soundwatch.models import common

    np.save(tiny_env / "cache" / "teacher_logits_train.npy", logits)
    common.mark_cache(tiny_env / "cache" / "teacher_logits_train.npy", "train", len(logits))
    m = train.run(_args(["--quick", "--distill", "--width", "8", "--blocks", "1", "--batch-size", "16"]))
    assert m["distill"] is True
    assert (tiny_env / "models" / "student_distill_quick.keras").exists()
