"""Real-time engine: source -> ring buffer -> gate -> features -> classifier -> postprocess. Owner: WP5.

Runs in a worker thread. Puts StatusUpdate / AlertEvent / EngineStopped on `out_queue`; reads
AckCommand / SettingsCommand from `in_queue`.

Per hop (every `hop_samples`, starting once one full window has been written; `t_audio` = end of
the window in stream seconds, i.e. 1.0, 1.5, 2.0, ...):
    1. drain in_queue (AckCommand -> PostProcessor.ack, SettingsCommand -> set_enabled)
    2. rms_dbfs(window) < gate_dbfs -> gated: probs = one-hot background, no features/inference
       else compute_logmel -> normalize -> classifier.predict (DummyClassifier: set_time first)
    3. PostProcessor.update(probs, t_audio) -> events
    4. out_queue: StatusUpdate (dropped if a bounded queue is full), then each AlertEvent
       (never dropped: blocks until there is room), events also go to the EventLogger
At the end (EOF, stop(), max_audio_s or error) EngineStopped(reason) is put on out_queue.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from soundwatch.audio import features
from soundwatch.audio.ringbuffer import RingBuffer
from soundwatch.audio.sources import AudioSource
from soundwatch.config import Config
from soundwatch.engine.classifier import Classifier
from soundwatch.engine.logger import EventLogger
from soundwatch.engine.postprocess import PostProcessor
from soundwatch.types import (
    AckCommand,
    AlertEvent,
    EngineStopped,
    ModeCommand,
    SettingsCommand,
    StatusUpdate,
)

log = logging.getLogger(__name__)


def _summ(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    a = np.asarray(values, dtype=np.float64)
    return float(a.mean()), float(np.percentile(a, 95))


class Engine:
    """See module docstring. Options beyond the contract:

    event_log_path: JSONL path for AlertEvents (None = no log file).
    emit_logmel: include the un-normalized log-mel in StatusUpdate (debug view); default True.
    max_audio_s: stop (reason "max_seconds") once t_audio reaches this value.
    mode: start-up context mode (overrides cfg.modes.default; needs a `modes:` config section).
    """

    def __init__(self, cfg: Config, source: AudioSource, classifier: Classifier, stats: dict,
                 out_queue: queue.Queue, in_queue: queue.Queue | None = None,
                 event_log_path=None, emit_logmel: bool = True, max_audio_s: float | None = None,
                 mode: str | None = None):
        # per-model front-end (feature_stats.json params): log-mel / PCEN, window length
        self.meta = features.model_meta(stats)
        cfg = features.apply_meta_to_config(cfg, self.meta)
        if classifier.num_classes != cfg.num_classes:
            raise ValueError(f"classifier has {classifier.num_classes} outputs, config has "
                             f"{cfg.num_classes} classes")
        if getattr(source, "sample_rate", cfg.audio.sample_rate) != cfg.audio.sample_rate:
            raise ValueError("source sample_rate differs from config")
        self.cfg = cfg
        self.source = source
        self.classifier = classifier
        self.feature_stats = stats
        self.out_queue = out_queue
        self.in_queue: queue.Queue = in_queue if in_queue is not None else queue.Queue()
        self.emit_logmel = emit_logmel
        self.max_audio_s = max_audio_s
        self.postprocessor = PostProcessor(cfg.classes, cfg.postprocess, cfg.modes, mode)
        self.logger = EventLogger(cfg.classes, event_log_path) if event_log_path is not None else None

        a = cfg.audio
        self._sr = a.sample_rate
        self._win = a.window_samples
        self._hop = a.hop_samples
        self._rb = RingBuffer(self._win)
        self._window = np.zeros(self._win, dtype=np.float32)
        self._bg_onehot = np.zeros(cfg.num_classes, dtype=np.float32)
        self._bg_onehot[cfg.background_id] = 1.0
        self._has_set_time = callable(getattr(classifier, "set_time", None))

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.error: BaseException | None = None
        self.stop_reason: str | None = None

        self._hops = 0
        self._gated = 0
        self._events = 0
        self._dropped_status = 0
        self._infer_ms: list[float] = []
        self._feature_ms: list[float] = []
        self._wall_s = 0.0

    # ------------------------------------------------------------ lifecycle
    def start(self) -> None:
        """Start worker thread."""
        if self._thread is not None:
            raise RuntimeError("engine already started")
        self._thread = threading.Thread(target=self._thread_main, name="soundwatch-engine", daemon=True)
        self._thread.start()

    def _thread_main(self) -> None:
        try:
            self.run()
        except Exception:  # already reported via EngineStopped("error: ...")
            log.exception("engine thread crashed")

    def run(self) -> None:
        """Blocking loop (used by start() and by --fast evaluation)."""
        reason = "eof"
        t0 = time.perf_counter()
        next_hop_at = self._win
        try:
            for block in self.source.blocks():
                if self._stop.is_set():
                    reason = "stopped"
                    break
                x = np.asarray(block, dtype=np.float32).reshape(-1)
                off, n = 0, x.shape[0]
                done = False
                while off < n:
                    take = min(n - off, next_hop_at - self._rb.total_written)
                    self._rb.write(x[off:off + take])
                    off += take
                    if self._rb.total_written == next_hop_at:
                        t_audio = next_hop_at / self._sr
                        self._process_hop(t_audio)
                        next_hop_at += self._hop
                        if self.max_audio_s is not None and t_audio >= self.max_audio_s:
                            reason, done = "max_seconds", True
                            break
                        if self._stop.is_set():
                            reason, done = "stopped", True
                            break
                if done:
                    break
            else:
                if self._stop.is_set():
                    reason = "stopped"
        except KeyboardInterrupt:
            reason = "interrupted"
        except Exception as e:
            reason = f"error: {e!r}"
            self.error = e
            raise
        finally:
            self._wall_s = time.perf_counter() - t0
            self.stop_reason = reason
            try:
                self.source.close()
            finally:
                if self.logger is not None:
                    self.logger.close()
                self._put_reliable(EngineStopped(reason=reason))

    def stop(self) -> None:
        """Thread-safe; the loop exits at the next block / hop boundary."""
        self._stop.set()
        try:
            self.source.close()
        except Exception:
            pass

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------ per hop
    def _drain_commands(self) -> None:
        while True:
            try:
                cmd = self.in_queue.get_nowait()
            except queue.Empty:
                return
            if isinstance(cmd, AckCommand):
                self.postprocessor.ack(cmd.class_id)
            elif isinstance(cmd, SettingsCommand):
                self.postprocessor.set_enabled(cmd.class_id, cmd.enabled)
            elif isinstance(cmd, ModeCommand):
                try:
                    self.postprocessor.set_mode(cmd.mode, self._rb.total_written / self._sr, "command")
                except ValueError as e:
                    log.warning("ignoring %r: %s", cmd, e)
            else:
                log.warning("ignoring unknown command %r", cmd)

    def _process_hop(self, t_audio: float) -> None:
        self._drain_commands()
        window = self._rb.latest(self._window)
        level = float(features.rms_dbfs(window))
        logmel = None
        infer_ms = 0.0
        if level < self.cfg.audio.gate_dbfs:
            gated = True
            probs = self._bg_onehot.copy()
            self._gated += 1
        else:
            gated = False
            t0 = time.perf_counter()
            logmel = features.compute_features(window, self.cfg.audio, self.meta)
            feats = features.normalize(logmel, self.feature_stats)
            t1 = time.perf_counter()
            if self._has_set_time:
                self.classifier.set_time(t_audio)  # type: ignore[attr-defined]
            probs = np.asarray(self.classifier.predict(feats), dtype=np.float32).reshape(-1)
            t2 = time.perf_counter()
            infer_ms = (t2 - t1) * 1000.0
            self._feature_ms.append((t1 - t0) * 1000.0)
            self._infer_ms.append(infer_ms)
        self._hops += 1
        events = self.postprocessor.update(probs, t_audio, rms_dbfs=level)

        status = StatusUpdate(t_audio=t_audio, rms_dbfs=level, probs=probs, gated=gated,
                              infer_ms=infer_ms, logmel=logmel if self.emit_logmel else None,
                              mode=self.postprocessor.mode, mode_selector=self.postprocessor.mode_selector)
        try:
            self.out_queue.put_nowait(status)
        except queue.Full:
            self._dropped_status += 1
        for ev in events:
            self._events += 1
            if self.logger is not None:
                self.logger.log(ev)
            self._put_reliable(ev)

    def _put_reliable(self, item: Any) -> None:
        """Blocking put that never drops; on a full bounded queue, retries until there is room.
        (Gives up only if stop() was requested and the queue stays full for 5 s.)"""
        waited = 0.0
        while True:
            try:
                self.out_queue.put(item, timeout=0.1)
                return
            except queue.Full:
                waited += 0.1
                if self._stop.is_set() and waited > 5.0:
                    log.warning("out_queue full; dropping %r after stop()", item)
                    return

    # ------------------------------------------------------------ stats
    @property
    def stats(self) -> dict:
        infer_mean, infer_p95 = _summ(self._infer_ms)
        feat_mean, feat_p95 = _summ(self._feature_ms)
        audio_s = self._rb.total_written / self._sr
        return {
            "hops": self._hops,
            "gated_hops": self._gated,
            "gated_fraction": self._gated / self._hops if self._hops else 0.0,
            "inferences": len(self._infer_ms),
            "infer_ms_mean": infer_mean,
            "infer_ms_p95": infer_p95,
            "feature_ms_mean": feat_mean,
            "feature_ms_p95": feat_p95,
            "events": self._events,
            "dropped_status": self._dropped_status,
            "audio_s": audio_s,
            "wall_s": self._wall_s,
            "realtime_factor": audio_s / self._wall_s if self._wall_s > 0 else 0.0,
            "stop_reason": self.stop_reason,
            "mode": self.postprocessor.mode,
            "mode_changes": [list(m) for m in self.postprocessor.mode_log],
        }


# ---------------------------------------------------------------- offline helper
class _Collector:
    """Queue-like sink used by run_offline: keeps events, compacts StatusUpdates into arrays."""

    def __init__(self, keep_logmel: bool = False):
        self.events: list[AlertEvent] = []
        self.t: list[float] = []
        self.rms: list[float] = []
        self.gated: list[bool] = []
        self.infer_ms: list[float] = []
        self.probs: list[np.ndarray] = []
        self.logmels: list[np.ndarray | None] = []
        self.keep_logmel = keep_logmel
        self.stopped: EngineStopped | None = None

    def put(self, item: Any, block: bool = True, timeout: float | None = None) -> None:
        if isinstance(item, StatusUpdate):
            self.t.append(item.t_audio)
            self.rms.append(item.rms_dbfs)
            self.gated.append(item.gated)
            self.infer_ms.append(item.infer_ms)
            self.probs.append(item.probs)
            if self.keep_logmel:
                self.logmels.append(item.logmel)
        elif isinstance(item, AlertEvent):
            self.events.append(item)
        elif isinstance(item, EngineStopped):
            self.stopped = item

    def put_nowait(self, item: Any) -> None:
        self.put(item, block=False)


@dataclass
class StatusSummary:
    """Per-hop arrays from run_offline (index i = hop i)."""
    t_audio: np.ndarray                  # (H,) float
    rms_dbfs: np.ndarray                 # (H,) float
    gated: np.ndarray                    # (H,) bool
    infer_ms: np.ndarray                 # (H,) float (0 when gated)
    probs: np.ndarray                    # (H, num_classes) float32, raw classifier output
    stop_reason: str = "eof"
    logmels: list = field(default_factory=list)  # only with keep_logmel=True

    def __len__(self) -> int:
        return int(self.t_audio.shape[0])


def run_offline(cfg: Config, source: AudioSource, classifier: Classifier, stats: dict,
                postprocessor_classes=None, keep_logmel: bool = False,
                event_log_path=None) -> tuple[list[AlertEvent], StatusSummary, dict]:
    """Run the engine synchronously in the calling thread, as fast as the source allows
    (use FileSource(path, realtime=False) / ArraySource), and collect everything.

    `postprocessor_classes`: optional replacement tuple of SoundClass (e.g. with swept thresholds).
    Returns (events, per-hop StatusSummary, engine.stats). The raw `probs` in the summary can be
    replayed through a fresh PostProcessor (see `replay_postprocess`) to sweep thresholds without
    re-running inference.
    """
    sink = _Collector(keep_logmel=keep_logmel)
    eng = Engine(cfg, source, classifier, stats, out_queue=sink,  # type: ignore[arg-type]
                 event_log_path=event_log_path, emit_logmel=keep_logmel)
    if postprocessor_classes is not None:
        eng.postprocessor = PostProcessor(tuple(postprocessor_classes), cfg.postprocess, cfg.modes)
    eng.run()
    n = cfg.num_classes
    summary = StatusSummary(
        t_audio=np.asarray(sink.t, dtype=np.float64),
        rms_dbfs=np.asarray(sink.rms, dtype=np.float64),
        gated=np.asarray(sink.gated, dtype=bool),
        infer_ms=np.asarray(sink.infer_ms, dtype=np.float64),
        probs=np.stack(sink.probs).astype(np.float32) if sink.probs else np.zeros((0, n), np.float32),
        stop_reason=sink.stopped.reason if sink.stopped else "eof",
        logmels=sink.logmels,
    )
    return sink.events, summary, eng.stats


def replay_postprocess(cfg: Config, summary: StatusSummary, classes=None) -> list[AlertEvent]:
    """Re-run only the PostProcessor over recorded per-hop probs (t_wall is set to 0). Context modes
    come from `cfg.modes` (start-up mode = cfg.modes.default); 'auto' uses the recorded RMS."""
    pp = PostProcessor(tuple(classes) if classes is not None else cfg.classes, cfg.postprocess, cfg.modes)
    out: list[AlertEvent] = []
    for t, p, r in zip(summary.t_audio, summary.probs, summary.rms_dbfs, strict=True):
        out.extend(pp.update(p, float(t), t_wall=0.0, rms_dbfs=float(r)))
    return out
