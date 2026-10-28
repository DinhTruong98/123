"""Watch simulator (pygame). Owner: WP6.

Contract used by `soundwatch run`:
    run_app(cfg, out_queue, in_queue, show_debug=True, max_seconds=None, screenshot_dir=None,
            close_on_stop=False, record_gif=None, stats_path=None) -> dict (run_loop stats)
consumes StatusUpdate / AlertEvent / EngineStopped from out_queue (engine -> UI) and
puts AckCommand / SettingsCommand on in_queue (UI -> engine). Returns when the user quits,
when max_seconds elapses, or (if close_on_stop) after EngineStopped.

`python -m soundwatch.ui.app --demo` runs with a fake event generator (no engine, no model).

Controls: Space / click on the face = acknowledge (AckCommand for the shown DANGER alert),
H = history, S = settings (Space/Enter/click toggles a class), Esc = back to idle,
arrow keys / mouse wheel = scroll, Q / window close = quit.
"""

from __future__ import annotations

import argparse
import math
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pygame

from soundwatch.config import Config, load_config
from soundwatch.types import AckCommand, AlertEvent, EngineStopped, Level, SettingsCommand, StatusUpdate
from soundwatch.ui.screens import WatchUI

MAX_MSGS_PER_FRAME = 200
GIF_FPS = 5
GIF_SCALE = 0.5


class GifRecorder:
    """Streams every `fps_ui / GIF_FPS`-th frame, scaled by GIF_SCALE, into an animated GIF (imageio)."""

    def __init__(self, path, ui_fps: int, fps: int = GIF_FPS, scale: float = GIF_SCALE):
        import imageio.v2 as imageio

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.every = max(1, round(ui_fps / fps))
        self.scale = scale
        self.frames = 0
        self._n = 0
        self._w = imageio.get_writer(str(self.path), mode="I", duration=1000.0 / fps, loop=0)

    def add(self, surface: pygame.Surface) -> None:
        self._n += 1
        if (self._n - 1) % self.every:
            return
        w, h = surface.get_size()
        small = pygame.transform.smoothscale(surface, (int(w * self.scale), int(h * self.scale)))
        arr = np.ascontiguousarray(pygame.surfarray.array3d(small).transpose(1, 0, 2))
        self._w.append_data(arr)
        self.frames += 1

    def close(self) -> None:
        self._w.close()


def write_ui_stats(path, stats: dict, target_fps: int, source: str = "") -> None:
    """ui_stats.json for the report: measured fps (frame cap = target_fps) + render capacity."""
    import json

    d = {k: v for k, v in stats.items() if k != "screenshots"}
    d.update({"target_fps": target_fps, "source": source})
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, indent=2), encoding="utf-8")


def _save_screenshot(surface: pygame.Surface, directory: Path, index: int, screen: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    p = directory / f"ui_{index:03d}_{screen}.png"
    pygame.image.save(surface, str(p))
    return p


def run_loop(cfg: Config, out_queue: queue.Queue, in_queue: queue.Queue, show_debug: bool = True,
             max_seconds: float | None = None, screenshot_dir=None, close_on_stop: bool = False,
             auto_ack_s: float | None = None, record_gif=None) -> dict:
    """Main loop behind run_app; returns stats {frames, seconds, fps, work_ms_mean, work_ms_p95,
    fps_capacity, gif_frames, screenshots}. `fps` is capped by cfg.ui.fps (clock.tick);
    `fps_capacity` = 1000 / p95 of the per-frame work time (events + update + draw + flip), i.e. the
    frame rate the loop could sustain without the cap.

    auto_ack_s (demo only): acknowledge a shown DANGER alert automatically after that many seconds.
    record_gif: path of an animated GIF to record (GIF_FPS frames/s, scaled by GIF_SCALE).
    """
    pygame.display.init()  # no pygame.init(): the mixer is not needed (and spams ALSA errors headless)
    pygame.font.init()
    ui = WatchUI(cfg, in_queue=in_queue, show_debug=show_debug)
    window = pygame.display.set_mode(ui.window_size)
    pygame.display.set_caption("SoundWatch – mô phỏng đồng hồ")
    clock = pygame.time.Clock()
    shot_dir = Path(screenshot_dir) if screenshot_dir else None
    shots: list[Path] = []
    gif = GifRecorder(record_gif, cfg.ui.fps) if record_gif else None
    t0 = time.monotonic()
    frames = 0
    work_ms: list[float] = []
    last_screen: str | None = None
    try:
        while True:
            now = time.monotonic()
            w0 = time.perf_counter()
            for ev in pygame.event.get():
                ui.handle_event(ev, now)
            for _ in range(MAX_MSGS_PER_FRAME):
                try:
                    msg = out_queue.get_nowait()
                except queue.Empty:
                    break
                ui.handle_message(msg, now)
            if auto_ack_s is not None and ui.screen == "alert" and ui.active is not None \
                    and ui.cfg.classes[ui.active.event.class_id].level == Level.DANGER \
                    and now - ui.active.shown_at >= auto_ack_s:
                ui.ack()
            ui.update(now)
            ui.fps = clock.get_fps()
            ui.draw(window)
            pygame.display.flip()
            work_ms.append((time.perf_counter() - w0) * 1000.0)
            if gif is not None:
                gif.add(window)
            frames += 1
            if shot_dir is not None and ui.screen != last_screen:
                shots.append(_save_screenshot(window, shot_dir, len(shots), ui.screen))
            last_screen = ui.screen
            if ui.quit_requested:
                break
            if close_on_stop and ui.engine_stopped is not None:
                break
            if max_seconds is not None and now - t0 >= max_seconds:
                break
            clock.tick(cfg.ui.fps)
        if shot_dir is not None:
            shots.append(_save_screenshot(window, shot_dir, len(shots), f"{ui.screen}_exit"))
    finally:
        elapsed = time.monotonic() - t0
        if gif is not None:
            gif.close()
        pygame.display.quit()
    wm = np.asarray(work_ms[1:] or work_ms or [0.0])  # first frame includes font/icon warm-up
    p95 = float(np.percentile(wm, 95))
    return {"frames": frames, "seconds": elapsed, "fps": frames / elapsed if elapsed > 0 else 0.0,
            "work_ms_mean": float(wm.mean()), "work_ms_p95": p95,
            "fps_capacity": 1000.0 / p95 if p95 > 0 else 0.0,
            "gif_frames": gif.frames if gif is not None else 0, "screenshots": shots}


def run_app(cfg: Config, out_queue: queue.Queue, in_queue: queue.Queue, show_debug: bool = True,
            max_seconds: float | None = None, screenshot_dir=None, close_on_stop: bool = False,
            record_gif=None, stats_path=None) -> dict:
    stats = run_loop(cfg, out_queue, in_queue, show_debug=show_debug, max_seconds=max_seconds,
                     screenshot_dir=screenshot_dir, close_on_stop=close_on_stop, record_gif=record_gif)
    if stats_path:
        write_ui_stats(stats_path, stats, cfg.ui.fps, source="run")
    return stats


# ---------------------------------------------------------------------- demo
class DemoGenerator(threading.Thread):
    """Fake engine: StatusUpdate every hop (random probs, synthetic log-mel, RMS) and AlertEvents
    cycling through every alert class. DANGER alerts repeat every `danger_repeat_s` until acked."""

    def __init__(self, cfg: Config, out_queue: queue.Queue, in_queue: queue.Queue, seed: int = 0,
                 event_every_s: float = 4.0) -> None:
        super().__init__(daemon=True, name="soundwatch-demo")
        self.cfg, self.out, self.inq = cfg, out_queue, in_queue
        self.rng = np.random.default_rng(seed)
        self.event_every_s = event_every_s
        self.stop_event = threading.Event()
        self.enabled = [c.enabled for c in cfg.classes]
        self.alert_ids = [c.id for c in cfg.classes if c.id != cfg.background_id]
        self.acks: list[int] = []

    def stop(self) -> None:
        self.stop_event.set()

    def _logmel(self, cls: int | None, t: float) -> np.ndarray:
        n_frames, n_mels = self.cfg.audio.n_frames, self.cfg.audio.n_mels
        base = -11.0 + 1.2 * self.rng.standard_normal((n_frames, n_mels)).astype(np.float32)
        base += np.linspace(2.0, -1.5, n_mels, dtype=np.float32)[None, :]  # pink-ish tilt
        if cls is not None:
            f = np.arange(n_frames, dtype=np.float32)
            centre = 8 + (cls * 4) % (n_mels - 12)
            for h in (0, 7, 13):  # a few "harmonics" that wobble per class
                m = centre + h + 1.5 * np.sin(2 * math.pi * (f / n_frames * (1 + cls % 3) + t))
                idx = np.clip(np.round(m).astype(int), 0, n_mels - 1)
                base[np.arange(n_frames), idx] += 9.0 - h * 0.4
            if cls % 2:  # pulsed sounds (horn, knock, bark ...)
                base *= np.where((f // 6) % 2 == 0, 1.0, 1.08)[:, None]
        return base

    def run(self) -> None:
        hop = self.cfg.audio.hop_s
        t_audio, k, next_event = 0.0, 0, 1.0
        active: int | None = None
        active_until = 0.0
        last_repeat: dict[int, float] = {}
        while not self.stop_event.is_set():
            while True:  # commands from the UI
                try:
                    cmd = self.inq.get_nowait()
                except queue.Empty:
                    break
                if isinstance(cmd, AckCommand):
                    self.acks.append(cmd.class_id)
                    last_repeat.pop(cmd.class_id, None)
                elif isinstance(cmd, SettingsCommand):
                    self.enabled[cmd.class_id] = cmd.enabled
                    if not cmd.enabled:
                        last_repeat.pop(cmd.class_id, None)
            if t_audio >= next_event:
                cid = self.alert_ids[k % len(self.alert_ids)]
                k += 1
                next_event = t_audio + self.event_every_s
                if self.enabled[cid]:
                    active, active_until = cid, t_audio + 2.0
                    conf = float(self.rng.uniform(max(0.6, self.cfg.classes[cid].threshold), 0.99))
                    self.out.put(AlertEvent(cid, conf, t_audio, time.time()))
                    if self.cfg.classes[cid].level == Level.DANGER:
                        last_repeat[cid] = t_audio
            for cid, t_last in list(last_repeat.items()):  # DANGER repeat until ack
                if t_audio - t_last >= self.cfg.postprocess.danger_repeat_s:
                    last_repeat[cid] = t_audio
                    self.out.put(AlertEvent(cid, float(self.rng.uniform(0.7, 0.95)), t_audio, time.time(),
                                            repeat=True))
            snd = active if active is not None and t_audio < active_until else None
            probs = self.rng.dirichlet(np.full(self.cfg.num_classes, 0.6)).astype(np.float32) * 0.35
            probs[snd if snd is not None else self.cfg.background_id] += 0.65
            probs /= probs.sum()
            rms = float(-30 + 6 * self.rng.standard_normal()) if snd is not None else \
                float(-58 + 8 * self.rng.standard_normal())
            gated = rms < self.cfg.audio.gate_dbfs
            self.out.put(StatusUpdate(t_audio=t_audio, rms_dbfs=rms, probs=probs, gated=gated,
                                      infer_ms=0.0 if gated else float(self.rng.uniform(2.5, 5.0)),
                                      logmel=self._logmel(snd, t_audio)))
            t_audio += hop
            self.stop_event.wait(hop)
        self.out.put(EngineStopped("demo stopped"))


def run_demo(cfg: Config, seconds: float | None = None, screenshot_dir=None, show_debug: bool = True,
             auto_ack_s: float | None = None, record_gif=None) -> dict:
    out_q: queue.Queue = queue.Queue()
    in_q: queue.Queue = queue.Queue()
    gen = DemoGenerator(cfg, out_q, in_q)
    gen.start()
    try:
        return run_loop(cfg, out_q, in_q, show_debug=show_debug, max_seconds=seconds,
                        screenshot_dir=screenshot_dir, auto_ack_s=auto_ack_s, record_gif=record_gif)
    finally:
        gen.stop()
        gen.join(timeout=2.0)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m soundwatch.ui.app", description="SoundWatch watch simulator")
    ap.add_argument("--demo", action="store_true", help="fake event generator (no engine / model)")
    ap.add_argument("--seconds", type=float, default=None, help="quit after N seconds")
    ap.add_argument("--screenshots", default=None, help="save a PNG on every screen change and on exit")
    ap.add_argument("--no-debug", action="store_true", help="watch face only (no debug panel)")
    ap.add_argument("--auto-ack", type=float, default=None, metavar="S",
                    help="demo: acknowledge DANGER alerts automatically after S seconds")
    ap.add_argument("--stats", default=None, metavar="JSON", help="write UI stats (fps ...) to this file")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    if not args.demo:
        print("Use `python -m soundwatch run` for the real engine, or --demo for fake events.",
              file=sys.stderr)
        return 2
    cfg = load_config(args.config)
    stats = run_demo(cfg, seconds=args.seconds, screenshot_dir=args.screenshots, show_debug=not args.no_debug,
                     auto_ack_s=args.auto_ack)
    print(f"frames={stats['frames']} seconds={stats['seconds']:.2f} fps={stats['fps']:.1f} "
          f"(capacity {stats['fps_capacity']:.0f} fps)")
    if args.stats:
        write_ui_stats(args.stats, stats, cfg.ui.fps, source="ui.app --demo")
    for p in stats["screenshots"]:
        print(f"screenshot: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
