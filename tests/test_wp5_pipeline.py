import json
import queue
import time

import numpy as np
import pytest
import soundfile as sf

from _legacy_cfg import legacy_config
from soundwatch.audio import features
from soundwatch.audio.sources import ArraySource
from soundwatch.engine.classifier import DummyClassifier
from soundwatch.engine.pipeline import Engine, replay_postprocess, run_offline
from soundwatch.types import AckCommand, AlertEvent, EngineStopped, SettingsCommand, StatusUpdate

SR = 16000
STATS = {"mean": 0.0, "std": 1.0}


@pytest.fixture(autouse=True)
def _features_available(monkeypatch):
    """Use the real WP2 features; fall back to simple stand-ins if they are not implemented yet."""
    try:
        features.rms_dbfs(np.ones(10, np.float32))
        features.compute_logmel(np.zeros(SR, np.float32))
    except NotImplementedError:
        def rms(x):
            ms = float(np.mean(np.square(x, dtype=np.float64)))
            return -120.0 if ms <= 1e-12 else max(-120.0, 10 * np.log10(ms))

        monkeypatch.setattr(features, "rms_dbfs", rms)
        monkeypatch.setattr(features, "compute_logmel", lambda w, cfg=None: np.zeros((49, 40), np.float32))
        monkeypatch.setattr(features, "normalize", lambda lm, st: (lm - st["mean"]) / st["std"])


@pytest.fixture(scope="module")
def cfg():
    return legacy_config()


def scenario(seconds_quiet=10.0, seconds_loud=20.0, seed=0):
    rng = np.random.default_rng(seed)
    quiet = 1e-4 * rng.standard_normal(int(SR * seconds_quiet))  # ~ -80 dBFS
    loud = 0.1 * rng.standard_normal(int(SR * seconds_loud))     # ~ -20 dBFS
    return np.concatenate([quiet, loud]).astype(np.float32)


def collect(q):
    items = []
    while True:
        try:
            items.append(q.get_nowait())
        except queue.Empty:
            return items


def test_engine_end_to_end(cfg, tmp_path):
    siren = cfg.class_by_key("siren").id
    dog = cfg.class_by_key("dog_bark").id
    baby = cfg.class_by_key("baby_cry").id
    sched = [(5.0, 7.0, baby), (12.0, 14.0, siren), (15.0, 17.0, dog)]
    clf = DummyClassifier(cfg.num_classes, sched)
    out_q: queue.Queue = queue.Queue()
    log = tmp_path / "ev.jsonl"
    eng = Engine(cfg, ArraySource(scenario(), blocksize=1234), clf, STATS, out_q, event_log_path=log)
    eng.run()
    items = collect(out_q)

    assert isinstance(items[-1], EngineStopped) and items[-1].reason == "eof"
    statuses = [i for i in items if isinstance(i, StatusUpdate)]
    events = [i for i in items if isinstance(i, AlertEvent)]
    # hops at t = 1.0, 1.5, ..., 30.0
    assert [s.t_audio for s in statuses] == pytest.approx(np.arange(1.0, 30.01, 0.5))
    # every AlertEvent comes right after the StatusUpdate of the same hop
    for i, it in enumerate(items):
        if isinstance(it, AlertEvent):
            j = i - 1
            while isinstance(items[j], AlertEvent):
                j -= 1
            assert isinstance(items[j], StatusUpdate) and items[j].t_audio == it.t_audio

    got = [(e.class_id, e.t_audio, e.repeat) for e in events]
    # baby_cry falls in the gated quiet part -> nothing. siren: fresh, then repeats every 5 s until
    # cleared 10 s after last above-threshold hop (14.0) -> repeats 18, 23. dog_bark (thr .75): 16.5
    expected = [(siren, 13.0, False), (dog, 16.5, False), (siren, 18.0, True), (siren, 23.0, True)]
    assert len(got) == len(expected)
    for (c, t, r), (ce, te, re_) in zip(got, expected, strict=True):
        assert c == ce and r == re_ and abs(t - te) <= 0.5

    gated = [s.gated for s in statuses]
    assert all(gated[:19])           # windows ending at 1.0..10.0 are entirely quiet
    assert not any(gated[19:])
    assert all(s.logmel is None for s in statuses[:19])
    assert statuses[25].logmel.shape == (49, 40)
    assert statuses[0].probs[cfg.background_id] == 1.0

    st = eng.stats
    assert st["hops"] == 59 and st["gated_hops"] == 19
    assert st["gated_fraction"] == pytest.approx(19 / 59)
    assert st["events"] == 4 and st["inferences"] == 40
    for k in ("infer_ms_mean", "infer_ms_p95", "feature_ms_mean", "feature_ms_p95", "realtime_factor"):
        assert st[k] >= 0
    assert st["audio_s"] == pytest.approx(30.0)

    lines = [json.loads(s) for s in log.read_text(encoding="utf-8").splitlines()]
    assert [ln["key"] for ln in lines] == ["siren", "dog_bark", "siren", "siren"]
    assert lines[0]["label_vi"] == "Còi cấp cứu" and lines[0]["level"] == "DANGER"
    assert set(lines[0]) == {"class_id", "key", "label_vi", "level", "confidence", "t_audio", "t_wall",
                             "repeat"}


class _AckingSource(ArraySource):
    """Injects commands into the engine's in_queue once a given stream time has been delivered."""

    def __init__(self, samples, in_q, at_s, cmds):
        super().__init__(samples, blocksize=1600)
        self.in_q, self.at, self.cmds = in_q, int(at_s * SR), cmds

    def blocks(self):
        sent, pos = False, 0
        for b in super().blocks():
            pos += b.size
            yield b
            if not sent and pos >= self.at:
                for c in self.cmds:
                    self.in_q.put(c)
                sent = True


def test_ack_via_in_queue_stops_repeats(cfg):
    siren = cfg.class_by_key("siren").id
    clf = DummyClassifier(cfg.num_classes, [(12.0, 14.0, siren)])
    in_q: queue.Queue = queue.Queue()
    out_q: queue.Queue = queue.Queue()
    src = _AckingSource(scenario(), in_q, 15.0, [AckCommand(siren)])
    Engine(cfg, src, clf, STATS, out_q, in_q).run()
    events = [i for i in collect(out_q) if isinstance(i, AlertEvent)]
    assert [(e.class_id, e.repeat) for e in events] == [(siren, False)]


def test_settings_command_disables_class(cfg):
    dog = cfg.class_by_key("dog_bark").id
    clf = DummyClassifier(cfg.num_classes, [(15.0, 17.0, dog)])
    in_q: queue.Queue = queue.Queue()
    in_q.put(SettingsCommand(dog, False))
    out_q: queue.Queue = queue.Queue()
    Engine(cfg, ArraySource(scenario()), clf, STATS, out_q, in_q).run()
    assert not any(isinstance(i, AlertEvent) for i in collect(out_q))


def test_bounded_queue_drops_status_not_events(cfg):
    siren = cfg.class_by_key("siren").id
    clf = DummyClassifier(cfg.num_classes, [(12.0, 14.0, siren)])
    out_q: queue.Queue = queue.Queue(maxsize=3)
    eng = Engine(cfg, ArraySource(scenario()), clf, STATS, out_q)
    eng.start()
    got = []
    time.sleep(0.3)  # slow consumer: the queue fills up
    while True:
        time.sleep(0.005)
        it = out_q.get(timeout=10)
        got.append(it)
        if isinstance(it, EngineStopped):
            break
    eng.join(5)
    assert not eng.running
    assert sum(isinstance(i, AlertEvent) for i in got) == 3  # fresh + 2 repeats, none dropped
    assert eng.stats["dropped_status"] > 0


def test_stop_from_other_thread(cfg):
    clf = DummyClassifier(cfg.num_classes, [])
    out_q: queue.Queue = queue.Queue()
    src = ArraySource(scenario(0, 60), realtime=True)
    eng = Engine(cfg, src, clf, STATS, out_q)
    eng.start()
    out_q.get(timeout=5)  # first StatusUpdate after ~1 s
    eng.stop()
    eng.join(3)
    assert not eng.running
    items = collect(out_q)
    assert isinstance(items[-1], EngineStopped) and items[-1].reason == "stopped"


def test_run_offline_and_replay(cfg):
    siren = cfg.class_by_key("siren").id
    clf = DummyClassifier(cfg.num_classes, [(12.0, 14.0, siren)])
    events, summary, st = run_offline(cfg, ArraySource(scenario()), clf, STATS)
    assert len(summary) == 59 and summary.probs.shape == (59, cfg.num_classes)
    assert summary.gated.sum() == 19 and summary.stop_reason == "eof"
    assert [e.t_audio for e in events] == [13.0, 18.0, 23.0]
    assert st["hops"] == 59
    replayed = replay_postprocess(cfg, summary)
    assert [(e.class_id, e.t_audio, e.repeat) for e in replayed] == \
        [(e.class_id, e.t_audio, e.repeat) for e in events]


def test_max_audio_s(cfg):
    clf = DummyClassifier(cfg.num_classes, [])
    out_q: queue.Queue = queue.Queue()
    eng = Engine(cfg, ArraySource(scenario()), clf, STATS, out_q, max_audio_s=5.0)
    eng.run()
    assert eng.stats["hops"] == 9 and eng.stats["stop_reason"] == "max_seconds"


def test_cli_headless_fast_dummy(tmp_path, capsys):
    from soundwatch.__main__ import main

    wav = tmp_path / "s.wav"
    rng = np.random.default_rng(1)
    sf.write(wav, (0.1 * rng.standard_normal(SR * 12)).astype(np.float32), SR)
    log = tmp_path / "ev.jsonl"
    rc = main(["run", "--source", str(wav), "--headless", "--fast", "--dummy", "--log", str(log)])
    assert rc == 0
    out = capsys.readouterr().out
    # dummy schedule: smoke_alarm 2-5 s, car_horn 10-13 s; confidence 1.0; config (WP8):
    # smooth_n=1, k_of_n=2/2 -> alert on the 2nd hop inside the interval, repeat 5 s later
    assert "[t=2.5s] DANGER Báo cháy (1.00)" in out
    assert "[t=10.5s] DANGER Còi xe" in out
    assert "[t=7.5s] DANGER Báo cháy" in out and "[repeat]" in out
    assert "alerts" in out and "gated" in out
    assert log.exists()


def test_cli_fast_with_mic_rejected(capsys):
    from soundwatch.__main__ import main

    assert main(["run", "--source", "mic", "--fast", "--headless"]) == 2
