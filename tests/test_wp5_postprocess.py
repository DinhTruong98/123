import dataclasses

import numpy as np
import pytest

from soundwatch.config import load_config
from soundwatch.engine.postprocess import PostProcessor

HOP = 0.5


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def probs_for(cfg, cid, p):
    n = cfg.num_classes
    v = np.full(n, (1 - p) / (n - 1), np.float32)
    v[cid] = p
    return v


def run_seq(pp, seq, t0=1.0):
    """seq: list of prob vectors, one per hop. Returns list of (t, event)."""
    out = []
    for i, p in enumerate(seq):
        t = t0 + i * HOP
        out.extend((t, e) for e in pp.update(p, t, t_wall=0.0))
    return out


def test_k_of_n_and_smoothing(cfg):
    dog = cfg.class_by_key("dog_bark").id  # INFO, thr 0.75, cooldown 15
    pp = PostProcessor(cfg.classes, cfg.postprocess)
    bg = probs_for(cfg, cfg.background_id, 1.0)
    hi = probs_for(cfg, dog, 1.0)
    # one isolated spike: smoothed 1/3 -> never triggers
    assert run_seq(pp, [bg] * 3 + [hi] + [bg] * 5) == []
    pp.reset()
    ev = run_seq(pp, [bg] * 3 + [hi] * 6)
    # smoothed after 1,2,3,4 hi hops: .33, .67, 1.0, 1.0 (thr .75) -> 2nd hit (k=2) at the 4th hi hop
    assert [(t, e.class_id, e.repeat) for t, e in ev] == [(1.0 + 6 * HOP, dog, False)]
    assert ev[0][1].confidence == pytest.approx(1.0)


def test_cooldown(cfg):
    dog = cfg.class_by_key("dog_bark").id
    pp = PostProcessor(cfg.classes, cfg.postprocess)
    hi = probs_for(cfg, dog, 1.0)
    ev = run_seq(pp, [hi] * 70)  # 35 s continuous
    ts = [t for t, _ in ev]
    assert len(ts) == 3
    assert np.allclose(np.diff(ts), 15.0)  # cooldown_s
    assert all(not e.repeat for _, e in ev)


def test_danger_repeat_ack_and_clear(cfg):
    fire = cfg.class_by_key("smoke_alarm").id
    pcfg = cfg.postprocess
    bg = probs_for(cfg, cfg.background_id, 1.0)
    hi = probs_for(cfg, fire, 1.0)

    # continuous alarm, never acked: fresh then repeat every danger_repeat_s
    pp = PostProcessor(cfg.classes, pcfg)
    ev = run_seq(pp, [hi] * 40)
    assert [e.repeat for _, e in ev] == [False, True, True, True]
    assert np.allclose(np.diff([t for t, _ in ev]), pcfg.danger_repeat_s)

    # ack after the first alert: no repeats, no fresh alert while the alarm continues
    pp = PostProcessor(cfg.classes, pcfg)
    first = run_seq(pp, [hi] * 3)
    assert len(first) == 1 and pp.is_pending(fire)
    pp.ack(fire)
    assert not pp.is_pending(fire)
    assert run_seq(pp, [hi] * 30, t0=2.5) == []

    # repeats continue after the sound stops, until it has been below threshold for danger_clear_s
    pp = PostProcessor(cfg.classes, pcfg)
    ev = run_seq(pp, [hi] * 3 + [bg] * 40)
    ts = [t for t, _ in ev]
    # first hop: smoothing uses 1 vector -> above at t=1.0, 1.5 -> alert at 1.5. After the sound
    # stops, smoothed is still 2/3 >= thr at t=2.5 (last above) -> cleared at 2.5 + danger_clear_s
    last_above = 2.5
    assert all(e.repeat for _, e in ev[1:])
    assert all(t < last_above + pcfg.danger_clear_s for t in ts)
    assert ts == [1.5, 6.5, 11.5]
    assert not pp.is_pending(fire)

    # after the occurrence cleared, a new occurrence alerts again (fresh)
    ev2 = run_seq(pp, [hi] * 3, t0=30.0)
    assert len(ev2) == 1 and not ev2[0][1].repeat


def test_ack_then_new_occurrence_after_clear(cfg):
    fire = cfg.class_by_key("smoke_alarm").id
    pp = PostProcessor(cfg.classes, cfg.postprocess)
    bg = probs_for(cfg, cfg.background_id, 1.0)
    hi = probs_for(cfg, fire, 1.0)
    run_seq(pp, [hi] * 3)
    pp.ack(fire)
    quiet = int(cfg.postprocess.danger_clear_s / HOP) + 2
    assert run_seq(pp, [bg] * quiet, t0=2.5) == []
    ev = run_seq(pp, [hi] * 3, t0=2.5 + quiet * HOP)
    assert len(ev) == 1 and not ev[0][1].repeat


def test_disabled_and_background_silent(cfg):
    fire = cfg.class_by_key("smoke_alarm").id
    pp = PostProcessor(cfg.classes, cfg.postprocess)
    pp.set_enabled(fire, False)
    assert run_seq(pp, [probs_for(cfg, fire, 1.0)] * 20) == []
    assert run_seq(pp, [probs_for(cfg, cfg.background_id, 1.0)] * 20, t0=20) == []
    # disabled in config
    classes = list(cfg.classes)
    classes[fire] = dataclasses.replace(classes[fire], enabled=False)
    pp2 = PostProcessor(classes, cfg.postprocess)
    assert run_seq(pp2, [probs_for(cfg, fire, 1.0)] * 20) == []
    # disabling while pending stops repeats
    pp3 = PostProcessor(cfg.classes, cfg.postprocess)
    assert len(run_seq(pp3, [probs_for(cfg, fire, 1.0)] * 3)) == 1
    pp3.set_enabled(fire, False)
    assert run_seq(pp3, [probs_for(cfg, fire, 1.0)] * 20, t0=2.5) == []


def test_multiple_classes_same_hop_ordered(cfg):
    fire = cfg.class_by_key("smoke_alarm").id
    dog = cfg.class_by_key("dog_bark").id
    horn = cfg.class_by_key("car_horn").id
    p = np.zeros(cfg.num_classes, np.float32)
    p[dog], p[fire], p[horn] = 0.9, 0.62, 0.8
    pp = PostProcessor(cfg.classes, cfg.postprocess)
    ev = run_seq(pp, [p] * 2)
    assert [e.class_id for _, e in ev] == [horn, fire, dog]


def test_repeat_confidence_is_occurrence_peak(cfg):
    fire = cfg.class_by_key("smoke_alarm").id
    pp = PostProcessor(cfg.classes, cfg.postprocess)
    bg = probs_for(cfg, cfg.background_id, 1.0)
    ev = run_seq(pp, [probs_for(cfg, fire, 0.7)] * 2 + [probs_for(cfg, fire, 0.95)] * 4 + [bg] * 10)
    assert [e.repeat for _, e in ev] == [False, True]
    assert ev[0][1].confidence == pytest.approx(0.7)
    assert ev[1][1].confidence == pytest.approx(0.95)
