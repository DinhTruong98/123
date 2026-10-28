"""WP8 (optional): record labelled clips from the microphone for later fine-tuning.

    python -m soundwatch record --class door_knock --seconds 5            # one clip
    python -m soundwatch record --class background --seconds 10 --count 6 # several clips

Saves 16 kHz mono PCM-16 WAVs to ``artifacts/data/custom/<class_key>/<YYYYmmdd-HHMMSS>.wav``.
``sounddevice`` (PortAudio) is imported lazily; without it a clear error is printed (exit 1).

Fine-tuning (not wired into build-dataset yet): the files have the same layout as
``artifacts/data/events/<split>/<class_key>/``, so they can be windowed exactly like the other clips
(1 s windows, hop 0.5 s, silent windows dropped) and appended to ``train.npz`` (keep a few
recordings per class aside for a "real microphone" evaluation set), then the student is fine-tuned
for a few epochs with a low learning rate starting from ``student_float.keras`` and re-quantized.
This reduces the domain shift between the public datasets and the real microphone.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from soundwatch import paths
from soundwatch.config import load_config

CUSTOM_DIR = paths.DATA / "custom"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--class", dest="class_key", required=True, help="class key, e.g. door_knock")
    parser.add_argument("--seconds", type=float, default=5.0, help="length of each clip")
    parser.add_argument("--count", type=int, default=1, help="number of clips to record")
    parser.add_argument("--device", default=None, help="sounddevice input device (index or name)")
    parser.add_argument("--out-dir", default=None, help=f"default {CUSTOM_DIR}")
    parser.add_argument("--countdown", type=float, default=1.0, help="seconds to wait before each clip")


def _import_sounddevice():
    try:
        import sounddevice as sd  # noqa: PLC0415 - lazy: PortAudio may be missing
    except (ImportError, OSError) as e:
        raise RuntimeError(
            f"cannot use the microphone: sounddevice/PortAudio not available ({e}). "
            "Install PortAudio (e.g. `sudo apt install libportaudio2`) and run on a machine "
            "with a microphone."
        ) from e
    return sd


def record_clip(seconds: float, sample_rate: int = 16000, device=None, sd=None) -> np.ndarray:
    """Blocking recording -> float32 mono (n,) in [-1, 1]."""
    sd = sd or _import_sounddevice()
    n = int(round(seconds * sample_rate))
    x = sd.rec(n, samplerate=sample_rate, channels=1, dtype="float32", device=device)
    sd.wait()
    return np.clip(np.asarray(x, dtype=np.float32).reshape(-1)[:n], -1.0, 1.0)


def clip_path(out_dir: Path, class_key: str, now: datetime | None = None) -> Path:
    now = now or datetime.now()
    d = Path(out_dir) / class_key
    p = d / f"{now:%Y%m%d-%H%M%S}.wav"
    i = 1
    while p.exists():  # several clips within the same second
        p = d / f"{now:%Y%m%d-%H%M%S}_{i}.wav"
        i += 1
    return p


def save_wav(path: Path, x: np.ndarray, sample_rate: int = 16000) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), x, sample_rate, subtype="PCM_16")


def main(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    if args.class_key not in cfg.class_keys:
        print(f"unknown class {args.class_key!r}; choose one of {', '.join(cfg.class_keys)}", file=sys.stderr)
        return 2
    if args.seconds <= 0 or args.count < 1:
        print("--seconds must be > 0 and --count >= 1", file=sys.stderr)
        return 2
    try:
        sd = _import_sounddevice()
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 1
    device = int(args.device) if args.device is not None and str(args.device).isdigit() else args.device
    out_dir = Path(args.out_dir) if args.out_dir else CUSTOM_DIR
    sr = cfg.audio.sample_rate
    for i in range(args.count):
        if args.countdown > 0:
            print(f"[record] {args.class_key} {i + 1}/{args.count}: starting in {args.countdown:g} s ...",
                  flush=True)
            time.sleep(args.countdown)
        print(f"[record] recording {args.seconds:g} s ...", flush=True)
        try:
            x = record_clip(args.seconds, sr, device, sd)
        except Exception as e:  # noqa: BLE001 - PortAudio errors (no device, ...)
            print(f"[record] recording failed: {e}", file=sys.stderr)
            return 1
        p = clip_path(out_dir, args.class_key)
        save_wav(p, x, sr)
        peak = float(np.max(np.abs(x))) if len(x) else 0.0
        print(f"[record] -> {p} (peak {peak:.2f})", flush=True)
    return 0
