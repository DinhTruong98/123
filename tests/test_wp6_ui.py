"""WP6 watch simulator UI tests (headless: SDL_VIDEODRIVER=dummy)."""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import queue  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pygame  # noqa: E402
import pytest  # noqa: E402

from soundwatch import paths  # noqa: E402
from soundwatch.config import load_config  # noqa: E402
from soundwatch.types import AckCommand, AlertEvent, EngineStopped, Level, SettingsCommand, StatusUpdate  # noqa: E402
from soundwatch.ui.screens import WatchUI  # noqa: E402
from soundwatch.ui.theme import Theme  # noqa: E402
from soundwatch.ui.vibration import VibrationPlayer, vibration_state  # noqa: E402

T0 = 1000.0


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture()
def ui(cfg):
    pygame.init()
    q: queue.Queue = queue.Queue()
    u = WatchUI(cfg, in_queue=q, show_debug=True)
    window = pygame.display.set_mode(u.window_size)
    u.window = window  # type: ignore[attr-defined]
    u.update(T0)
    yield u
    pygame.display.quit()


def _status(cfg, t: float, seed: int = 0, logmel: bool = True) -> StatusUpdate:
    rng = np.random.default_rng(seed)
    lm = None
    if logmel:
        lm = (rng.standard_normal((cfg.audio.n_frames, cfg.audio.n_mels)) - 8).astype(np.float32)
    probs = rng.dirichlet(np.ones(cfg.num_classes)).astype(np.float32)
    return StatusUpdate(t_audio=t, rms_dbfs=-32.0, probs=probs, gated=False, infer_ms=3.1, logmel=lm)


def _id(cfg, level: Level) -> int:
    return next(c.id for c in cfg.classes if c.level == level and c.id != cfg.background_id)


def _draw_and_save(u: WatchUI, name: str) -> None:
    u.draw(u.window)  # type: ignore[attr-defined]
    paths.REPORTS.mkdir(parents=True, exist_ok=True)
    out = paths.REPORTS / f"ui_{name}.png"
    pygame.image.save(u.window, str(out))  # type: ignore[attr-defined]
    assert out.exists() and out.stat().st_size > 1000


def test_all_screens_render_and_screenshot(ui, cfg):
    for i in range(6):
        ui.handle_message(_status(cfg, i * 0.5, seed=i), T0)
    ui.update(T0)
    _draw_and_save(ui, "idle")
    ui.handle_message(AlertEvent(_id(cfg, Level.INFO), 0.81, 1.0, time.time() - 120), T0)
    ui.handle_message(AlertEvent(_id(cfg, Level.WARN), 0.77, 2.0, time.time() - 60), T0)
    ui.handle_message(AlertEvent(_id(cfg, Level.DANGER), 0.93, 3.0, time.time()), T0)
    ui.update(T0 + 0.1)
    assert ui.screen == "alert"
    _draw_and_save(ui, "alert")
    ui.go("history")
    ui.update(T0 + 0.2)
    _draw_and_save(ui, "history")
    ui.go("settings")
    ui.update(T0 + 0.3)
    _draw_and_save(ui, "settings")
    # face is 240x240 with transparent corners (round mask)
    face = ui.render_face()
    assert face.get_size() == (cfg.ui.watch_size, cfg.ui.watch_size)
    assert face.get_at((0, 0)).a == 0 and face.get_at((120, 120)).a == 255


def test_every_class_alert_renders(ui, cfg):
    for sc in cfg.classes:
        ui.handle_message(AlertEvent(sc.id, 0.9, 0.0, time.time()), T0)
        ui.update(T0 + 0.01)
        ui.draw(ui.window)
        ui.ack()


def test_alert_event_switches_on_next_update(ui, cfg):
    assert ui.screen == "idle"
    ui.handle_message(AlertEvent(_id(cfg, Level.INFO), 0.8, 0.0, time.time()), T0)
    ui.update(T0 + 1 / cfg.ui.fps)
    assert ui.screen == "alert"
    # non-danger alerts time out
    ui.update(T0 + cfg.ui.alert_timeout_s + 0.1)
    assert ui.screen == "idle"


def test_danger_stays_until_ack_and_sends_ack(ui, cfg):
    cid = _id(cfg, Level.DANGER)
    ui.handle_message(AlertEvent(cid, 0.9, 0.0, time.time()), T0)
    ui.update(T0 + cfg.ui.alert_timeout_s * 3)
    assert ui.screen == "alert"
    ui.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE))
    assert ui.in_queue.get_nowait() == AckCommand(cid)
    assert ui.screen == "idle"


def test_click_on_face_acks(ui, cfg):
    cid = _id(cfg, Level.DANGER)
    ui.handle_message(AlertEvent(cid, 0.9, 0.0, time.time()), T0)
    centre = (ui.face_pos[0] + ui.face_px // 2, ui.face_pos[1] + ui.face_px // 2)
    ui.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=centre))
    assert ui.in_queue.get_nowait() == AckCommand(cid)


def test_priority_highest_level_then_most_recent(ui, cfg):
    d = [c.id for c in cfg.classes if c.level == Level.DANGER]
    info = _id(cfg, Level.INFO)
    ui.handle_message(AlertEvent(d[0], 0.9, 0.0, time.time()), T0)
    ui.handle_message(AlertEvent(info, 0.9, 0.5, time.time()), T0)  # lower: history only
    assert ui.active.event.class_id == d[0]
    ui.handle_message(AlertEvent(d[1], 0.9, 1.0, time.time()), T0)  # same level, more recent
    assert ui.active.event.class_id == d[1]
    assert [h.class_id for h in ui.history] == [d[1], info, d[0]]
    # ack the shown one -> the other unacknowledged DANGER comes back
    ui.ack()
    assert ui.screen == "alert" and ui.active.event.class_id == d[0]
    ui.ack()
    assert ui.screen == "idle"
    assert [ui.in_queue.get_nowait(), ui.in_queue.get_nowait()] == [AckCommand(d[1]), AckCommand(d[0])]


def test_repeat_danger_revibrates_without_history(ui, cfg):
    cid = _id(cfg, Level.DANGER)
    ui.handle_message(AlertEvent(cid, 0.9, 0.0, time.time()), T0)
    assert len(ui.history) == 1
    later = T0 + 5.0
    assert not ui.vib.state(later).active
    ui.handle_message(AlertEvent(cid, 0.9, 5.0, time.time(), repeat=True), later)
    assert len(ui.history) == 1
    assert ui.vib.state(later + 0.01).on
    assert ui.event_log[0].repeat


def test_settings_toggle_sends_command(ui, cfg):
    ui.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_s))
    assert ui.screen == "settings"
    ui.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_DOWN))
    ui.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE))
    cmd = ui.in_queue.get_nowait()
    assert cmd == SettingsCommand(cfg.classes[1].id, False)
    assert ui.enabled[1] is False
    # click on the first row toggles class 0
    ui.update(T0)
    ui.render_face()
    from soundwatch.ui.screens import SET_ROW, SET_TOP

    y = ui.face_pos[1] + (SET_TOP + SET_ROW // 2) * ui.scale
    x = ui.face_pos[0] + ui.face_px // 2
    ui.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(x, y)))
    assert ui.in_queue.get_nowait() == SettingsCommand(cfg.classes[0].id, False)


def test_navigation_and_quit(ui):
    ui.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_h))
    assert ui.screen == "history"
    ui.handle_event(pygame.event.Event(pygame.MOUSEWHEEL, x=0, y=-1))
    ui.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE))
    assert ui.screen == "idle"
    ui.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_q))
    assert ui.quit_requested


def test_history_len_and_scroll(ui, cfg):
    for i in range(cfg.ui.history_len + 5):
        ui.handle_message(AlertEvent(i % (cfg.num_classes - 1), 0.8, float(i), time.time()), T0)
    assert len(ui.history) == cfg.ui.history_len
    ui.go("history")
    for _ in range(50):
        ui.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_DOWN))
    assert ui.history_scroll > 0
    ui.draw(ui.window)


def test_spectrogram_keeps_last_when_logmel_none(ui, cfg):
    ui.handle_message(_status(cfg, 0.0), T0)
    before = ui.debug.spec.copy()
    ui.handle_message(_status(cfg, 0.5, logmel=False), T0)
    np.testing.assert_array_equal(np.nan_to_num(before), np.nan_to_num(ui.debug.spec))
    ui.handle_message(_status(cfg, 1.0, seed=3), T0)
    assert not np.array_equal(np.nan_to_num(before), np.nan_to_num(ui.debug.spec))
    ui.handle_message(EngineStopped("eof"), T0)
    assert ui.engine_stopped == "eof"
    ui.draw(ui.window)


def test_vietnamese_labels_render(cfg):
    pygame.font.init()
    th = Theme(cfg)
    font = th.font(24)
    missing = font.metrics("\uffff")[0]
    labels = ["Báo cháy", "Còi cấp cứu", "Gõ cửa / Chuông cửa", "Em bé khóc"]
    labels += [c.label_vi for c in cfg.classes]
    for text in labels:
        for ch in set(text) - set(" /"):
            m = font.metrics(ch)[0]
            assert m is not None and m != missing, f"glyph {ch!r} missing"
            assert m[1] - m[0] > 0 and m[3] > 0, f"glyph {ch!r} empty"
        assert th.text(text, 24).get_width() > 0
    # diacritics actually rise above the base letter and change the bitmap
    assert font.metrics("á")[0][3] > font.metrics("a")[0][3]
    assert font.metrics("ử")[0][3] > font.metrics("u")[0][3]
    a = pygame.surfarray.array_alpha(font.render("a", True, (255, 255, 255)))
    aa = pygame.surfarray.array_alpha(font.render("á", True, (255, 255, 255)))
    assert a.shape == aa.shape and not np.array_equal(a, aa)


def test_vibration_state_machine():
    pat = (800, 200, 800, 200, 800)
    assert vibration_state(pat, 0).on
    assert vibration_state(pat, 799).on and vibration_state(pat, 799).segment == 0
    s = vibration_state(pat, 900)
    assert s.active and not s.on and s.segment == 1
    assert vibration_state(pat, 1000).on and vibration_state(pat, 1000).segment == 2
    assert vibration_state(pat, 2500).on and vibration_state(pat, 2500).segment == 4
    assert vibration_state(pat, 2799).active
    assert not vibration_state(pat, 2800).active  # total = 2800 ms
    assert not vibration_state((), 0).active
    p = VibrationPlayer()
    assert not p.state(5.0).active
    p.start([150, 100, 150], now=10.0)
    assert p.state(10.1).on and not p.state(10.2).on and p.state(10.3).on and not p.state(10.41).active
    p.start([150, 100, 150], now=20.0)  # restart (repeat)
    assert p.state(20.05).on


def test_vibration_jitters_face(ui, cfg):
    cid = _id(cfg, Level.DANGER)
    ui.handle_message(AlertEvent(cid, 0.9, 0.0, time.time()), T0)
    assert ui.vib.state(T0 + 0.1).on
    ui.update(T0 + 0.1)
    ui.draw(ui.window)


def test_demo_runs_headless_fast_enough():
    env = dict(os.environ, SDL_VIDEODRIVER="dummy", SDL_AUDIODRIVER="dummy")
    r = subprocess.run([sys.executable, "-m", "soundwatch.ui.app", "--demo", "--seconds", "3"],
                       capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 0, r.stderr
    line = next(ln for ln in r.stdout.splitlines() if ln.startswith("frames="))
    fps = float(line.split("fps=")[1].split()[0])
    assert fps >= 25, line


def test_run_app_contract(cfg, tmp_path):
    from soundwatch.ui.app import run_app

    out_q: queue.Queue = queue.Queue()
    in_q: queue.Queue = queue.Queue()
    out_q.put(_status(cfg, 0.0))
    out_q.put(AlertEvent(_id(cfg, Level.WARN), 0.9, 0.0, time.time()))
    out_q.put(EngineStopped("eof"))
    t = time.monotonic()
    run_app(cfg, out_q, in_q, show_debug=False, max_seconds=5, screenshot_dir=tmp_path, close_on_stop=True)
    assert time.monotonic() - t < 4
    shots = sorted(p.name for p in tmp_path.glob("*.png"))
    assert any("alert" in s for s in shots) and any("exit" in s for s in shots)
