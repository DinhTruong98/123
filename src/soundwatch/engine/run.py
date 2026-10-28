"""`python -m soundwatch run`: real-time engine + watch simulator UI (or headless console). Owner: WP5.

    python -m soundwatch run --source mic
    python -m soundwatch run --source sample.wav --headless --fast
    python -m soundwatch run --source sample.wav --dummy            # no model needed
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
from pathlib import Path

from soundwatch.config import Config, load_config
from soundwatch.types import AlertEvent, EngineStopped, Level

DUMMY_ON_S = 3.0      # each scheduled class lasts this long ...
DUMMY_PERIOD_S = 8.0  # ... and a new one starts every DUMMY_PERIOD_S seconds
DUMMY_START_S = 2.0


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source", default="mic", help="'mic' or path to a WAV/FLAC file")
    parser.add_argument("--headless", action="store_true", help="print alerts to the console, no UI")
    parser.add_argument("--fast", action="store_true", help="file source: process as fast as possible")
    parser.add_argument("--model", default=None, help=".tflite (or .keras) model (default: from config)")
    parser.add_argument("--stats", default=None, help="feature_stats.json (default: from config)")
    parser.add_argument("--dummy", action="store_true", help="use DummyClassifier (scripted alerts)")
    parser.add_argument("--no-debug", action="store_true", help="hide UI debug panel")
    parser.add_argument("--max-seconds", type=float, default=None, help="stop after this many seconds")
    parser.add_argument("--device", default=None, help="sounddevice input device (mic only)")
    parser.add_argument("--log", default=None, help="event log JSONL path (default: artifacts/logs/...)")
    parser.add_argument("--no-log", action="store_true", help="do not write an event log")


def _warn(msg: str) -> None:
    print(f"WARNING: {msg}", file=sys.stderr)


def dummy_schedule(cfg: Config, duration_s: float) -> list[tuple[float, float, int]]:
    """Cycle through the enabled alert classes: class i is 'heard' for DUMMY_ON_S seconds,
    starting every DUMMY_PERIOD_S seconds."""
    ids = [c.id for c in cfg.classes if c.id != cfg.background_id and c.enabled]
    sched: list[tuple[float, float, int]] = []
    t, i = DUMMY_START_S, 0
    while ids and t < duration_s:
        sched.append((t, t + DUMMY_ON_S, ids[i % len(ids)]))
        t += DUMMY_PERIOD_S
        i += 1
    return sched


def load_feature_stats(path: Path | None) -> dict:
    if path is None or not Path(path).exists():
        _warn(f"feature stats {path} not found; using mean=0, std=1")
        return {"mean": 0.0, "std": 1.0}
    try:
        from soundwatch.audio.features import load_stats

        return load_stats(path)
    except NotImplementedError:
        with open(path, encoding="utf-8") as f:
            return json.load(f)


def format_alert(ev: AlertEvent, cfg: Config) -> str:
    sc = cfg.classes[ev.class_id]
    s = f"[t={ev.t_audio:.1f}s] {Level(sc.level).name} {sc.label_vi} ({ev.confidence:.2f})"
    return s + " [repeat]" if ev.repeat else s


class _ConsoleSink:
    """Queue-like sink for headless mode: prints alerts as they happen."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.n_events = 0
        self.stopped: EngineStopped | None = None

    def put(self, item, block: bool = True, timeout: float | None = None) -> None:
        if isinstance(item, AlertEvent):
            self.n_events += 1
            print(format_alert(item, self.cfg), flush=True)
        elif isinstance(item, EngineStopped):
            self.stopped = item

    def put_nowait(self, item) -> None:
        self.put(item, block=False)


def _print_stats(stats: dict) -> None:
    print(
        f"-- {stats['events']} alerts | {stats['audio_s']:.1f} s audio in {stats['wall_s']:.2f} s "
        f"(x{stats['realtime_factor']:.1f}) | hops {stats['hops']}, gated {stats['gated_fraction']:.0%} | "
        f"feature {stats['feature_ms_mean']:.2f} ms (p95 {stats['feature_ms_p95']:.2f}) | "
        f"infer {stats['infer_ms_mean']:.2f} ms (p95 {stats['infer_ms_p95']:.2f}) | "
        f"stop: {stats['stop_reason']}",
        flush=True,
    )


def main(args: argparse.Namespace) -> int:
    from soundwatch.audio.sources import FileSource, MicSource
    from soundwatch.engine.classifier import DummyClassifier, load_classifier
    from soundwatch.engine.logger import default_log_path
    from soundwatch.engine.pipeline import Engine

    cfg = load_config(getattr(args, "config", None))
    sr = cfg.audio.sample_rate

    # ---- source
    is_mic = args.source == "mic"
    if is_mic:
        if args.fast:
            print("--fast is only valid with a file source", file=sys.stderr)
            return 2
        device = args.device
        if device is not None and device.isdigit():
            device = int(device)
        try:
            source = MicSource(sample_rate=sr, blocksize=1600, device=device)
        except RuntimeError as e:
            print(str(e), file=sys.stderr)
            return 1
        duration = 3600.0  # dummy schedule length for the mic
    else:
        path = Path(args.source)
        if not path.exists():
            print(f"source file not found: {path}", file=sys.stderr)
            return 2
        source = FileSource(path, sample_rate=sr, blocksize=1600, realtime=not args.fast)
        duration = source.duration_s

    # ---- classifier
    model_path = Path(args.model) if args.model else cfg.model.resolved_tflite()
    if not args.dummy and not model_path.exists():
        _warn(f"model {model_path} not found; falling back to DummyClassifier (scripted alerts)")
        args.dummy = True
    if args.dummy:
        sched = dummy_schedule(cfg, duration)
        print(f"DummyClassifier: cycling {len(sched)} scripted alerts "
              f"({DUMMY_ON_S:.0f} s every {DUMMY_PERIOD_S:.0f} s; gated (quiet) audio stays silent)",
              file=sys.stderr)
        classifier = DummyClassifier(cfg.num_classes, sched)
    else:
        classifier = load_classifier(model_path)

    stats_path = Path(args.stats) if args.stats else cfg.model.resolved_stats()
    feat_stats = load_feature_stats(stats_path)

    log_path = None if args.no_log else (Path(args.log) if args.log else default_log_path())

    # ---- run
    if args.headless:
        sink = _ConsoleSink(cfg)
        engine = Engine(cfg, source, classifier, feat_stats, out_queue=sink,  # type: ignore[arg-type]
                        event_log_path=log_path, emit_logmel=False, max_audio_s=args.max_seconds)
        engine.run()
        _print_stats(engine.stats)
        if log_path is not None and sink.n_events:
            print(f"events logged to {log_path}")
        return 0

    try:
        from soundwatch.ui.app import run_app
    except ImportError as e:
        print(f"cannot import UI ({e}); use --headless", file=sys.stderr)
        return 1
    out_q: queue.Queue = queue.Queue(maxsize=256)
    in_q: queue.Queue = queue.Queue()
    engine = Engine(cfg, source, classifier, feat_stats, out_queue=out_q, in_queue=in_q,
                    event_log_path=log_path, emit_logmel=not args.no_debug)
    engine.start()
    try:
        run_app(cfg, out_q, in_q, show_debug=not args.no_debug, max_seconds=args.max_seconds,
                close_on_stop=False)
    except KeyboardInterrupt:
        pass
    except NotImplementedError as e:
        print(f"UI not available ({e}); use --headless", file=sys.stderr)
        return 1
    finally:
        engine.stop()
        # unblock an engine waiting on a full queue
        try:
            while True:
                out_q.get_nowait()
        except queue.Empty:
            pass
        engine.join(timeout=3.0)
    _print_stats(engine.stats)
    return 0
