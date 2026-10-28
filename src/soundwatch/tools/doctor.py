"""Environment check: python -m soundwatch doctor."""

from __future__ import annotations

import argparse
import os
import platform

from soundwatch import paths
from soundwatch.config import load_config


def add_arguments(parser: argparse.ArgumentParser) -> None:
    pass


def _check(name: str, fn) -> bool:
    try:
        detail = fn()
        print(f"[ OK ] {name}: {detail}")
        return True
    except Exception as e:  # noqa: BLE001 - report every failure, keep going
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")
        return False


def _tensorflow() -> str:
    import numpy as np
    import tensorflow as tf

    interp = tf.lite.Interpreter
    del interp
    return f"tensorflow {tf.__version__}, numpy {np.__version__}"


def _pygame() -> str:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame

    pygame.init()
    pygame.display.set_mode((10, 10))
    pygame.quit()
    return f"pygame {pygame.version.ver} (video driver {os.environ['SDL_VIDEODRIVER']})"


def _mic() -> str:
    import sounddevice as sd  # raises OSError if PortAudio is missing

    dev = sd.query_devices(kind="input")
    return f"default input: {dev['name']}"


def _exists(p) -> str:
    if not p.exists():
        raise FileNotFoundError(p)
    return str(p.relative_to(paths.ROOT))


def main(args: argparse.Namespace) -> int:
    print(f"python {platform.python_version()} on {platform.system()} {platform.machine()}")
    ok = _check("config", lambda: f"{load_config(args.config).num_classes} classes")
    ok &= _check("tensorflow + tflite", _tensorflow)
    ok &= _check("pygame headless", _pygame)
    _check("microphone (optional)", _mic)
    _check("YAMNet weights (WP3)", lambda: _exists(paths.CACHE / "yamnet" / "yamnet.h5"))
    _check("windowed dataset (WP1)", lambda: _exists(paths.WINDOWS / "train.npz"))
    _check("int8 model (WP4)", lambda: _exists(load_config(args.config).model.resolved_tflite()))
    return 0 if ok else 1
