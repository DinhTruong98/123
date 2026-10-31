# ruff: noqa: E501
"""Evaluate the context-mode `auto` rule on labelled indoor / street background sessions (vn-v3).

Sessions: alternating 180 s blocks of indoor and street background (6 blocks = 18 min), concatenated from
the background noise bank of one split (artifacts/data/noise/<split>, never train):
  indoor  = ESC-50 domestic categories (folds 4/5) + VN background clips whose title is clearly indoor
  street  = `vn`:   VN background clips whose title is street / traffic / market
            `orig`: UrbanSound8K street classes (engine_idling, street_music, jackhammer, drilling,
                    children_playing) + ESC-50 engine
Levels: `rand` = every clip normalised to a random -45..-30 dBFS (as make-scenarios: the level carries no
information), `plaus` = an ASSUMED plausible microphone level (indoor -50..-38, street -38..-26 dBFS; must
be calibrated on the real watch / phone).
The model runs once per session (engine, no modes); the PostProcessor is then replayed with each auto rule.
Metrics: hop accuracy (mode == truth), accuracy after 60 s of each block, switch latency after each
transition (censored at the block length), wrong switches per hour.

usage: auto_mode_eval.py OUT.json NAME=MODEL:STATS ...   (selection on val, report on test)
"""

from __future__ import annotations

import dataclasses
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from soundwatch import paths
from soundwatch.audio.sources import ArraySource
from soundwatch.config import AutoModeConfig, load_config
from soundwatch.engine.classifier import load_model
from soundwatch.engine.pipeline import run_offline
from soundwatch.engine.postprocess import PostProcessor

SR = 16000
BLOCK_S = 180.0
N_BLOCKS = 6
N_SESS = 6
SETTLE_S = 60.0
CFG_PATH = paths.ROOT / "config" / "soundwatch_vn_v2.yaml"  # modes indoor/street (+auto replaced below)
CACHE = paths.ARTIFACTS / "vn_v3" / "auto_cache"

ESC_INDOOR = {"breathing", "brushing_teeth", "can_opening", "cat", "clock_tick", "coughing", "door_wood_creaks",
              "drinking_sipping", "footsteps", "keyboard_typing", "laughing", "mouse_click", "sneezing", "snoring",
              "vacuum_cleaner", "washing_machine", "glass_breaking", "clapping"}
ESC_STREET = {"engine"}
US8K_STREET = {"engine_idling", "street_music", "jackhammer", "drilling", "children_playing"}
VN_STREET_RE = re.compile(r"street|phố|traffic|market|chợ|walking|xe|đường|night|hanoi|hà nội|saigon|sài gòn|"
                          r"city|court|rao|cà rem|loa phát thanh|hoi an|da nang|ben thanh|bùi viện|bung|long xuy|"
                          r"trà mơn|rach bang", re.I)
VN_INDOOR_RE = re.compile(r"karaoke tân cổ|quán|cafe|coffee|máy giặt|quạt", re.I)


def pools(split: str) -> dict[str, list[Path]]:
    nd = paths.DATA / "noise" / split
    esc = pd.read_csv(paths.DATA / "raw" / "esc50" / "meta" / "esc50.csv").set_index("filename")["category"]
    us = pd.read_csv(paths.DATA / "raw" / "urbansound8k" / "UrbanSound8K" / "metadata" / "UrbanSound8K.csv")
    us = us.set_index("slice_file_name")["class"]
    man = pd.read_csv(paths.ROOT / "data" / "vn" / "manifest.csv").set_index("file")["title"]
    out: dict[str, list[Path]] = {"indoor": [], "street_vn": [], "street_orig": []}
    for p in sorted(nd.glob("*.wav")):
        n = p.name
        if n.startswith("esc50_"):
            cat = esc.get(n[len("esc50_"):])
            if cat in ESC_INDOOR:
                out["indoor"].append(p)
            elif cat in ESC_STREET:
                out["street_orig"].append(p)
        elif n.startswith("us8k_"):
            if us.get(n.split("_", 2)[2]) in US8K_STREET:
                out["street_orig"].append(p)
        elif n.startswith("vn_"):
            f = n[len("vn_"):].replace("background_background", "background/background", 1)
            title = str(man.get(f, ""))
            if VN_INDOOR_RE.search(title):
                out["indoor"].append(p)
            elif VN_STREET_RE.search(title):
                out["street_vn"].append(p)
    return out


def _rms_db(x: np.ndarray) -> float:
    return 10 * np.log10(np.mean(np.square(x, dtype=np.float64)) + 1e-12)


def make_session(pool_in, pool_st, levels: str, rng, start_street: bool):
    audio, truth = [], []
    for b in range(N_BLOCKS):
        street = (b % 2 == 0) == start_street
        pool = pool_st if street else pool_in
        lo, hi = ((-45, -30) if levels == "rand" else ((-38, -26) if street else (-50, -38)))
        n, buf = 0, []
        while n < BLOCK_S * SR:
            x, _ = sf.read(str(pool[rng.integers(len(pool))]), dtype="float32")
            if x.ndim > 1:
                x = x.mean(axis=1)
            if _rms_db(x) < -90:
                continue
            x = x * 10 ** ((rng.uniform(lo, hi) - _rms_db(x)) / 20)
            buf.append(np.clip(x, -1, 1))
            n += len(x)
        blk = np.concatenate(buf)[: int(BLOCK_S * SR)]
        audio.append(blk)
        truth.append("street" if street else "indoor")
    return np.concatenate(audio).astype(np.float32), truth


def session_hops(model, stats, split, street_kind, levels):
    """Per-hop (t, probs, rms, truth) of every session; cached per model/split/kind/levels."""
    CACHE.mkdir(parents=True, exist_ok=True)
    key = f"{Path(model).parent.name}_{Path(model).stem}_{split}_{street_kind}_{levels}"
    cp = CACHE / f"{key}.npz"
    if cp.exists():
        z = np.load(cp, allow_pickle=True)
        return list(z["sessions"])
    cfg = load_config(paths.ROOT / "config" / "soundwatch.yaml")
    clf, st, _ = load_model(model, stats)
    pl = pools(split)
    rng = np.random.default_rng(2026 + (split == "test"))
    sess = []
    for i in range(N_SESS):
        audio, truth = make_session(pl["indoor"], pl[f"street_{street_kind}"], levels, rng, start_street=i % 2 == 1)
        _, summ, _ = run_offline(cfg, ArraySource(audio), clf, st)
        tr = np.array([truth[min(int(t // BLOCK_S), N_BLOCKS - 1)] for t in summ.t_audio])
        sess.append({"t": summ.t_audio, "probs": summ.probs, "rms": summ.rms_dbfs, "truth": tr})
    np.savez(cp, sessions=np.array(sess, dtype=object))
    return sess


def replay(sess, auto: AutoModeConfig):
    base = load_config(CFG_PATH)
    modes = dataclasses.replace(base.modes, default="auto", auto=auto)
    pp = PostProcessor(base.classes, base.postprocess, modes)
    out = []
    for t, p, r in zip(sess["t"], sess["probs"], sess["rms"], strict=True):
        pp.update(p, float(t), t_wall=0.0, rms_dbfs=float(r))
        out.append(pp.mode)
    return np.array(out)


def score(sessions, auto):
    acc_n = acc_d = set_n = set_d = 0
    lat, wrong, hours = [], 0, 0.0
    for s in sessions:
        m = replay(s, auto)
        ok = m == s["truth"]
        acc_n += ok.sum()
        acc_d += len(ok)
        tin = s["t"] % BLOCK_S
        settled = tin >= SETTLE_S
        set_n += ok[settled].sum()
        set_d += settled.sum()
        hours += s["t"][-1] / 3600
        for b in range(1, N_BLOCKS):
            sel = (s["t"] >= b * BLOCK_S) & (s["t"] < (b + 1) * BLOCK_S)
            hit = np.flatnonzero(ok[sel])
            lat.append(float(s["t"][sel][hit[0]] - b * BLOCK_S) if len(hit) else BLOCK_S)
        sw = np.flatnonzero(m[1:] != m[:-1]) + 1
        wrong += int(sum(m[i] != s["truth"][i] for i in sw))
    lat = np.array(lat)
    return {"acc": acc_n / acc_d, "acc_settled": set_n / set_d, "lat_median_s": float(np.median(lat)),
            "lat_p90_s": float(np.percentile(lat, 90)), "never_switched": int((lat >= BLOCK_S).sum()),
            "n_transitions": len(lat), "wrong_switches_per_h": wrong / hours}


RMS_V2 = AutoModeConfig(street_above_dbfs=-35.0, window_s=30.0, hysteresis_db=3.0, min_s=5.0)


EVIDENCE = {"horn-household": (("car_horn", 1.0), ("baby_cry", -1.0), ("door_knock", -1.0), ("water_running", -1.0)),
            "horn": (("car_horn", 1.0),)}


def grid(evidence):
    for thr in (-0.02, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02, 0.03):
        for hyst in (0.0, 0.01, 0.02):
            for tau in (30.0, 60.0):
                for dwell in (0.0, 30.0, 60.0):
                    yield AutoModeConfig(street_above_dbfs=None, street_evidence=evidence, street_score_above=thr,
                                         score_hysteresis=hyst, evidence_window_s=tau, min_dwell_s=dwell, min_s=10.0)


def select(val, evidence):
    """Best sound-only rule on val (both street kinds, random levels): accuracy - 0.002 * wrong switches/h."""
    best = None
    for a in grid(evidence):
        sc = [score(val[k], a) for k in ("vn", "orig")]
        obj = np.mean([x["acc"] for x in sc]) - 0.002 * np.mean([x["wrong_switches_per_h"] for x in sc])
        if best is None or obj > best[0]:
            best = (obj, a)
    return best[1]


def main():
    outp = sys.argv[1]
    models = [a.split("=", 1) for a in sys.argv[2:]]
    res: dict = {"pools": {s: {k: len(v) for k, v in pools(s).items()} for s in ("val", "test")}, "models": {}}
    print("pools", res["pools"], flush=True)
    for name, ms in models:
        model, stats = ms.split(":")
        r: dict = {}
        val = {k: session_hops(model, stats, "val", k, "rand") for k in ("vn", "orig")}
        # selection on val only (both street kinds, random levels -> sound evidence only)
        sound = select(val, EVIDENCE["horn-household"])
        horn = select(val, EVIDENCE["horn"])
        r["selected_sound_rule"] = dataclasses.asdict(sound)
        r["selected_horn_only_rule"] = dataclasses.asdict(horn)
        comb = dataclasses.replace(sound, street_above_dbfs=-35.0, window_s=30.0, hysteresis_db=3.0)
        rules = {"rms_v2": RMS_V2, "horn_only": horn, "sound": sound, "rms+sound": comb}
        for split in ("val", "test"):
            for kind in ("vn", "orig"):
                for lv in ("rand", "plaus"):
                    sess = session_hops(model, stats, split, kind, lv)
                    for rn, a in rules.items():
                        r[f"{split}/{kind}/{lv}/{rn}"] = x = score(sess, a)
                        print(f"{name} {split} {kind:<4} {lv:<5} {rn:<9} acc {x['acc']:.3f} settled {x['acc_settled']:.3f} "
                              f"lat med {x['lat_median_s']:.0f}s p90 {x['lat_p90_s']:.0f}s never {x['never_switched']}/{x['n_transitions']} "
                              f"wrong/h {x['wrong_switches_per_h']:.1f}", flush=True)
        res["models"][name] = r
        print(name, "selected", r["selected_sound_rule"], "horn only", r["selected_horn_only_rule"], flush=True)
    json.dump(res, open(outp, "w"), indent=1)


if __name__ == "__main__":
    main()
