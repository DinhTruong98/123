"""WP7: synthesize long streaming test scenarios (plan section 6 / WP7).

Outputs (default ``paths.SCENARIOS``):
  scenario_XX.wav / scenario_XX.csv   N x 180 s: continuous background + 3-6 alert events
  bg_XX.wav                           M x 360 s background only (false-alarm measurement)
  index.json                          parameters + per-file metadata
  assets/demo/demo_scenario.wav/.csv  ~60 s demo, one event of every alert class at 10 dB SNR

Construction
  * Background: random noise-bank clips (``noise/<split>/*.wav``) with leading/trailing silence
    trimmed, each RMS-normalized, concatenated with 0.5 s equal-power crossfades, then the whole
    track scaled to a random RMS level in [-45, -30] dBFS.
  * Events: clips from ``events/<split>/<class_key>/*.wav`` trimmed to their *active region*
    (frames of 20 ms whose RMS is within ``ACTIVE_REL_DB`` dB of the clip's loudest frame, and above
    ``ACTIVE_ABS_DBFS``), 10 ms fade in/out, scaled so that
    ``20*log10(rms(event) / rms(background over the same span)) == snr_db`` with snr_db drawn
    from {0, 5, 10, 20}. GT onset/offset = start/end of the inserted (trimmed) clip.
  * Placement: events never overlap and are separated by >= ``MIN_GAP_S`` (4 s) of pure
    background; the first ``HEAD_S`` s and last ``TAIL_S`` s stay event-free.
  * If the mixture would clip, the *whole* file is scaled down (SNRs are preserved).
  * Classes and SNRs are drawn from shuffled cycles so every class / SNR is roughly balanced.

CSV columns: class_key, class_id, onset_s, offset_s, snr_db, source_clip
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from soundwatch import paths
from soundwatch.config import Config, load_config

SNRS_DB = (0.0, 5.0, 10.0, 20.0)
BG_LEVEL_DBFS = (-45.0, -30.0)
MIN_GAP_S = 4.0
HEAD_S = 2.0
TAIL_S = 3.0
FADE_S = 0.010
XFADE_S = 0.5
ACTIVE_REL_DB = 30.0
ACTIVE_ABS_DBFS = -65.0
ACTIVE_FRAME_S = 0.020
MAX_EVENT_S = 8.0
PEAK_LIMIT = 0.98
CSV_FIELDS = ("class_key", "class_id", "onset_s", "offset_s", "snr_db", "source_clip")


# ---------------------------------------------------------------- audio helpers
def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if len(x) else 0.0


def db(x: float) -> float:
    return 20.0 * np.log10(max(x, 1e-12))


def load_wav(path: str | Path, sr: int = 16000) -> np.ndarray:
    """Any soundfile-readable file -> mono float32 at `sr`."""
    import soundfile as sf

    from soundwatch.audio.sources import resample, to_mono_float32

    data, sr_file = sf.read(str(path), dtype="float32", always_2d=True)
    return resample(to_mono_float32(data), int(sr_file), sr)


def write_wav(path: Path, x: np.ndarray, sr: int = 16000) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(x, -1.0, 1.0), sr, subtype="PCM_16")


def active_region(
    x: np.ndarray,
    sr: int = 16000,
    rel_db: float = ACTIVE_REL_DB,
    abs_dbfs: float = ACTIVE_ABS_DBFS,
    frame_s: float = ACTIVE_FRAME_S,
) -> tuple[int, int]:
    """(start, end) sample indices of the non-silent part: first..last frame whose RMS is within
    `rel_db` of the loudest frame and above `abs_dbfs`. (0, 0) if the clip is silent."""
    n = max(1, int(round(frame_s * sr)))
    nf = len(x) // n
    if nf == 0:
        return (0, len(x)) if db(rms(x)) > abs_dbfs else (0, 0)
    fr = x[: nf * n].reshape(nf, n).astype(np.float64)
    lv = 10.0 * np.log10(np.mean(fr * fr, axis=1) + 1e-20)
    thr = max(lv.max() - rel_db, abs_dbfs)
    on = np.flatnonzero(lv >= thr)
    if on.size == 0:
        return 0, 0
    end = len(x) if on[-1] == nf - 1 else (on[-1] + 1) * n
    return int(on[0] * n), int(end)


def apply_fades(x: np.ndarray, sr: int = 16000, fade_s: float = FADE_S) -> np.ndarray:
    y = np.array(x, dtype=np.float32, copy=True)
    n = min(int(round(fade_s * sr)), len(y) // 2)
    if n > 0:
        ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
        y[:n] *= ramp
        y[-n:] *= ramp[::-1]
    return y


def trim_event(x: np.ndarray, sr: int = 16000, max_s: float = MAX_EVENT_S) -> np.ndarray:
    """Active region of an event clip (capped at `max_s`), with 10 ms fades."""
    s, e = active_region(x, sr)
    y = x[s:e][: int(max_s * sr)]
    return apply_fades(y, sr) if len(y) else y


# ---------------------------------------------------------------- sources
@dataclass
class EventClip:
    class_key: str
    class_id: int
    wave: np.ndarray  # trimmed, faded
    source: str


class Library:
    """Noise clips + per-class event clips (loaded lazily, trimmed, cached)."""

    def __init__(
        self,
        cfg: Config,
        noise: list[tuple[str, np.ndarray]],
        events: dict[str, list[tuple[str, np.ndarray]]],
    ):
        self.cfg = cfg
        min_len = int((2 * XFADE_S + 0.2) * cfg.audio.sample_rate)
        self.noise = [(n, w) for n, w in noise if len(w) >= min_len and rms(w) > 1e-6]
        self.events = {k: v for k, v in events.items() if v}
        if not self.noise:
            raise FileNotFoundError("no usable background/noise clips")
        if not self.events:
            raise FileNotFoundError("no usable event clips")

    @property
    def class_keys(self) -> list[str]:
        return [k for k in self.cfg.class_keys[:-1] if k in self.events]

    @classmethod
    def from_dirs(
        cls, cfg: Config, noise_dir: Path, events_dir: Path, noise: list[tuple[str, np.ndarray]] | None = None
    ) -> Library:
        """`noise`: optional pre-loaded noise clips (then `noise_dir` is not read)."""
        sr = cfg.audio.sample_rate
        if noise is None:
            nfiles = sorted(Path(noise_dir).glob("*.wav"))
            if not nfiles:
                raise FileNotFoundError(f"no noise wavs in {noise_dir}")
            noise = []
            for f in nfiles:
                x = load_wav(f, sr)
                s, e = active_region(x, sr)
                noise.append((f.name, x[s:e]))
        events: dict[str, list[tuple[str, np.ndarray]]] = {}
        for key in cfg.class_keys[:-1]:
            clips = []
            for f in sorted((Path(events_dir) / key).glob("*.wav")):
                y = trim_event(load_wav(f, sr), sr)
                if len(y) >= int(0.1 * sr):
                    clips.append((f"{key}/{f.name}", y))
            events[key] = clips
        if not any(events.values()):
            raise FileNotFoundError(f"no event wavs in {events_dir}/<class_key>/")
        return cls(cfg, noise, events)

    @classmethod
    def from_windows(cls, cfg: Config, npz_path: Path) -> Library:
        """Fallback: 1 s windows of windows/<split>.npz (first window of every source clip for
        events; all background windows grouped by clip for noise)."""
        noise, events = windows_clips(cfg, npz_path)
        return cls(cfg, noise, events)


def windows_clips(
    cfg: Config, npz_path: Path
) -> tuple[list[tuple[str, np.ndarray]], dict[str, list[tuple[str, np.ndarray]]]]:
    """(noise clips, event clips) from windows/<split>.npz, see Library.from_windows."""
    if not Path(npz_path).exists():
        raise FileNotFoundError(f"{npz_path} not found")
    sr = cfg.audio.sample_rate
    with np.load(npz_path, allow_pickle=False) as d:
        wave, label, src = d["wave"], d["label"].astype(int), d["source_clip"]
    noise: dict[str, list[np.ndarray]] = {}
    events: dict[str, list[tuple[str, np.ndarray]]] = {k: [] for k in cfg.class_keys[:-1]}
    seen: set[str] = set()
    for w, lab, s in zip(wave, label, src, strict=True):
        s = str(s)
        if lab == cfg.background_id:
            noise.setdefault(s, []).append(w.astype(np.float32)[: sr // 2])  # hop-sized chunks
        elif s not in seen:
            seen.add(s)
            y = trim_event(w.astype(np.float32), sr)
            if len(y) >= int(0.1 * sr):
                events[cfg.class_keys[lab]].append((s, y))
    return [(k, np.concatenate(v)) for k, v in noise.items()], events


def load_library(
    cfg: Config,
    split: str = "test",
    noise_dir: Path | None = None,
    events_dir: Path | None = None,
    windows_dir: Path | None = None,
    verbose: bool = True,
) -> Library:
    noise_dir = Path(noise_dir) if noise_dir else paths.NOISE / split
    events_dir = Path(events_dir) if events_dir else paths.DATA / "events" / split
    npz = (Path(windows_dir) if windows_dir else paths.WINDOWS) / f"{split}.npz"
    if not any(Path(noise_dir).glob("*.wav")) and Path(events_dir).is_dir() and Path(npz).exists():
        # e.g. the val split: build-dataset writes no noise bank for it, but full event clips exist.
        # Use full event clips + background reconstructed from the 1 s windows of <split>.npz.
        try:
            noise, _ = windows_clips(cfg, npz)
            full = Library.from_dirs(cfg, noise_dir, events_dir, noise=noise)
            if verbose:
                print(f"[scenarios] no noise bank in {noise_dir}: background from {npz}, "
                      f"events from {events_dir}")
            return full
        except FileNotFoundError:
            pass
    try:
        return Library.from_dirs(cfg, noise_dir, events_dir)
    except FileNotFoundError as e:
        if verbose:
            print(f"[scenarios] {e}; falling back to 1 s windows from {npz}")
        try:
            return Library.from_windows(cfg, npz)
        except FileNotFoundError as e2:
            raise FileNotFoundError(
                f"cannot build scenarios: {e}; and {e2}. Run `python -m soundwatch build-dataset` first."
            ) from e2


# ---------------------------------------------------------------- synthesis
def make_background(
    lib: Library,
    n_samples: int,
    rng: np.random.Generator,
    level_dbfs: float,
    sr: int = 16000,
    xfade_s: float = XFADE_S,
    segments: list | None = None,
) -> np.ndarray:
    """Continuous background of `n_samples` at RMS `level_dbfs` from crossfaded noise clips.

    If `segments` is a list, (start_s, end_s, clip_name) of every clip used is appended to it
    (does not change the random stream, so outputs are identical with or without it)."""
    xf = int(xfade_s * sr)
    out = np.zeros(n_samples + xf, np.float64)
    pos = 0
    ramp = np.linspace(0.0, np.pi / 2, xf) if xf > 0 else np.zeros(0)
    fade_in, fade_out = np.sin(ramp), np.cos(ramp)
    order: list[int] = []
    skipped = 0
    while pos < n_samples:
        if not order:
            order = list(rng.permutation(len(lib.noise)))
        name, clip = lib.noise[order.pop()]
        r = rms(clip)
        if r < 1e-6 or len(clip) <= 2 * xf:
            skipped += 1
            if skipped > 10 * len(lib.noise):
                raise ValueError("noise clips are too short or silent")
            continue
        seg = clip.astype(np.float64) / r
        if xf > 0:
            seg[-xf:] *= fade_out
        start = pos
        if pos > 0 and xf > 0:
            seg[:xf] *= fade_in
            start = pos - xf
        end = min(start + len(seg), len(out))
        out[start:end] += seg[: end - start]
        if segments is not None and start < n_samples:
            segments.append((round(start / sr, 3), round(min(end, n_samples) / sr, 3), name))
        pos = start + len(seg)
    bg = out[:n_samples]
    return (bg * (10.0 ** (level_dbfs / 20.0)) / max(rms(bg), 1e-12)).astype(np.float32)


def place_events(
    durations: list[int],
    n_samples: int,
    rng: np.random.Generator,
    sr: int = 16000,
    min_gap_s: float = MIN_GAP_S,
    head_s: float = HEAD_S,
    tail_s: float = TAIL_S,
) -> list[int]:
    """Random non-overlapping onsets (samples), >= min_gap between consecutive events, in the
    given order. Raises ValueError if they do not fit."""
    k = len(durations)
    head, tail, gap = int(head_s * sr), int(tail_s * sr), int(min_gap_s * sr)
    slack = n_samples - head - tail - sum(durations) - max(0, k - 1) * gap
    if slack < 0:
        raise ValueError("events do not fit into the scenario")
    cuts = np.sort(rng.integers(0, slack + 1, size=k)) if k else np.zeros(0, int)
    onsets, pos, prev = [], head, 0
    for i, d in enumerate(durations):
        pos += int(cuts[i]) - prev
        prev = int(cuts[i])
        onsets.append(pos)
        pos += d + gap
    return onsets


def insert_event(
    mix: np.ndarray, bg: np.ndarray, ev: np.ndarray, onset: int, snr_db: float, floor_rms: float = 0.0
) -> float:
    """Add `ev` into `mix` at `onset`, scaled to `snr_db` vs the background RMS over the same span
    (at least `floor_rms`). Returns the gain applied."""
    seg = bg[onset : onset + len(ev)]
    ref = max(rms(seg), floor_rms, 1e-7)
    g = ref * 10.0 ** (snr_db / 20.0) / max(rms(ev), 1e-9)
    mix[onset : onset + len(ev)] += (g * ev).astype(np.float32)
    return g


class _Cycle:
    """Endless shuffled cycle over `items`."""

    def __init__(self, items, rng: np.random.Generator):
        self.items, self.rng, self.buf = list(items), rng, []

    def next(self):
        if not self.buf:
            self.buf = [self.items[i] for i in self.rng.permutation(len(self.items))]
        return self.buf.pop()


def _limit_peak(x: np.ndarray, limit: float = PEAK_LIMIT, min_gain: float = 0.0) -> tuple[float, int]:
    """Scale `x` in place so its peak is <= `limit`, but never by less than `min_gain` (keeps the
    background level inside BG_LEVEL_DBFS); samples still above `limit` are hard-clipped.
    Returns (gain, number of clipped samples)."""
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    g = 1.0
    if peak > limit:
        g = max(limit / peak, min(1.0, min_gain))
        x *= g
    n_clip = int(np.count_nonzero(np.abs(x) > limit))
    if n_clip:
        np.clip(x, -limit, limit, out=x)
    return g, n_clip


def synth_event_scenario(
    lib: Library,
    duration_s: float,
    rng: np.random.Generator,
    classes: _Cycle,
    snrs: _Cycle,
    clip_cycles: dict[str, _Cycle],
    n_events: tuple[int, int] = (3, 6),
    sr: int = 16000,
) -> tuple[np.ndarray, list[dict], dict]:
    n = int(round(duration_s * sr))
    level = float(rng.uniform(*BG_LEVEL_DBFS))
    segs: list = []
    bg = make_background(lib, n, rng, level, sr, segments=segs)
    k = int(rng.integers(n_events[0], n_events[1] + 1))
    picks = []
    for _ in range(k):
        key = classes.next()
        name, w = clip_cycles[key].next()
        picks.append((key, name, w, float(snrs.next())))
    # drop events from the end until they fit
    while picks:
        try:
            onsets = place_events([len(p[2]) for p in picks], n, rng, sr)
            break
        except ValueError:
            picks.pop()
    else:
        onsets = []
    mix = bg.copy()
    floor = 10.0 ** ((level - 10.0) / 20.0)
    events = []
    for (key, name, w, snr), on in zip(picks, onsets, strict=True):
        insert_event(mix, bg, w, on, snr, floor)
        events.append(
            {
                "class_key": key,
                "class_id": lib.cfg.class_keys.index(key),
                "onset_s": round(on / sr, 4),
                "offset_s": round((on + len(w)) / sr, 4),
                "snr_db": snr,
                "source_clip": name,
            }
        )
    g, n_clip = _limit_peak(mix, min_gain=10.0 ** ((BG_LEVEL_DBFS[0] - level) / 20.0))
    return (
        mix,
        events,
        {
            "bg_level_dbfs": round(level + db(g), 2),
            "peak_gain": round(g, 4),
            "clipped_samples": n_clip,
            "bg_segments": segs,
        },
    )


def synth_background(
    lib: Library, duration_s: float, rng: np.random.Generator, sr: int = 16000
) -> tuple[np.ndarray, dict]:
    level = float(rng.uniform(*BG_LEVEL_DBFS))
    segs: list = []
    x = make_background(lib, int(round(duration_s * sr)), rng, level, sr, segments=segs)
    g, _ = _limit_peak(x)
    return x, {"bg_level_dbfs": round(level + db(g), 2), "peak_gain": round(g, 4), "bg_segments": segs}


def write_csv(path: Path, events: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        wr.writeheader()
        for e in events:
            wr.writerow({k: e.get(k, "") for k in CSV_FIELDS})


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [
        {
            "class_key": r["class_key"],
            "class_id": int(r["class_id"]),
            "onset_s": float(r["onset_s"]),
            "offset_s": float(r["offset_s"]),
            "snr_db": float(r["snr_db"]),
            "source_clip": r.get("source_clip", ""),
        }
        for r in rows
    ]


def generate(
    cfg: Config,
    lib: Library,
    out_dir: Path,
    n_events: int = 20,
    n_background: int = 10,
    seed: int = 0,
    event_duration_s: float = 180.0,
    bg_duration_s: float = 360.0,
    events_per_scenario: tuple[int, int] = (3, 6),
    split: str = "test",
    verbose: bool = True,
) -> dict:
    """Write scenario_XX.wav/.csv, bg_XX.wav and index.json into `out_dir`; returns the index."""
    sr = cfg.audio.sample_rate
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in list(out_dir.glob("scenario_*.*")) + list(out_dir.glob("bg_*.wav")):
        old.unlink()
    rng = np.random.default_rng(seed)
    classes = _Cycle(lib.class_keys, rng)
    snrs = _Cycle(SNRS_DB, rng)
    clip_cycles = {k: _Cycle(v, rng) for k, v in lib.events.items()}
    index: dict = {
        "seed": seed,
        "split": split,
        "sample_rate": sr,
        "snrs_db": list(SNRS_DB),
        "bg_level_dbfs_range": list(BG_LEVEL_DBFS),
        "min_gap_s": MIN_GAP_S,
        "active_rel_db": ACTIVE_REL_DB,
        "fade_s": FADE_S,
        "classes": lib.class_keys,
        "event_scenarios": [],
        "background_scenarios": [],
    }
    for i in range(n_events):
        x, evs, meta = synth_event_scenario(
            lib, event_duration_s, rng, classes, snrs, clip_cycles, events_per_scenario, sr
        )
        wav, csvp = out_dir / f"scenario_{i:02d}.wav", out_dir / f"scenario_{i:02d}.csv"
        write_wav(wav, x, sr)
        write_csv(csvp, evs)
        index["event_scenarios"].append(
            {"wav": wav.name, "csv": csvp.name, "duration_s": len(x) / sr, "n_events": len(evs), **meta}
        )
        if verbose:
            print(f"[scenarios] {wav.name}: {len(evs)} events, bg {meta['bg_level_dbfs']:.1f} dBFS")
    for i in range(n_background):
        x, meta = synth_background(lib, bg_duration_s, rng, sr)
        wav = out_dir / f"bg_{i:02d}.wav"
        write_wav(wav, x, sr)
        index["background_scenarios"].append({"wav": wav.name, "duration_s": len(x) / sr, **meta})
        if verbose:
            print(f"[scenarios] {wav.name}: background only, {meta['bg_level_dbfs']:.1f} dBFS")
    counts: dict[str, int] = {}
    snr_counts: dict[str, int] = {}
    for s in index["event_scenarios"]:
        for e in read_csv(out_dir / s["csv"]):
            counts[e["class_key"]] = counts.get(e["class_key"], 0) + 1
            snr_counts[f"{e['snr_db']:g}"] = snr_counts.get(f"{e['snr_db']:g}", 0) + 1
    index["event_counts"] = counts
    index["snr_counts"] = snr_counts
    with open(out_dir / "index.json", "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
    return index


def make_demo(
    cfg: Config,
    lib: Library,
    out_wav: Path,
    seed: int = 0,
    duration_s: float = 60.0,
    snr_db: float = 10.0,
    max_event_s: float = 3.5,
) -> list[dict]:
    """One event of every alert class at `snr_db`, evenly spread over ~`duration_s` seconds."""
    sr = cfg.audio.sample_rate
    rng = np.random.default_rng(seed + 1000)
    keys = lib.class_keys
    clips = []
    for key in keys:
        name, w = lib.events[key][int(rng.integers(len(lib.events[key])))]
        clips.append((key, name, apply_fades(w[: int(max_event_s * sr)], sr)))
    n = int(duration_s * sr)
    total = sum(len(c[2]) for c in clips)
    head, tail = int(HEAD_S * sr), int(TAIL_S * sr)
    gap = max(int(MIN_GAP_S * sr), (n - head - tail - total) // max(1, len(clips) - 1))
    need = head + tail + total + gap * (len(clips) - 1)
    n = max(n, need)
    level = -38.0
    bg = make_background(lib, n, rng, level, sr)
    mix = bg.copy()
    pos, events = head, []
    for key, name, w in clips:
        insert_event(mix, bg, w, pos, snr_db, 10.0 ** ((level - 10.0) / 20.0))
        events.append(
            {
                "class_key": key,
                "class_id": cfg.class_keys.index(key),
                "onset_s": round(pos / sr, 4),
                "offset_s": round((pos + len(w)) / sr, 4),
                "snr_db": snr_db,
                "source_clip": name,
            }
        )
        pos += len(w) + gap
    _limit_peak(mix)
    out_wav = Path(out_wav)
    write_wav(out_wav, mix, sr)
    write_csv(out_wav.with_suffix(".csv"), events)
    return events


# ---------------------------------------------------------------- CLI
def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--n-events", type=int, default=20, help="number of event scenarios")
    parser.add_argument("--n-background", type=int, default=10, help="number of background-only scenarios")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", default="test", help="noise/events split to draw from")
    parser.add_argument("--duration", type=float, default=180.0, help="event scenario length (s)")
    parser.add_argument("--bg-duration", type=float, default=360.0, help="background scenario length (s)")
    parser.add_argument("--out", default=None, help="output directory (default artifacts/data/scenarios)")
    parser.add_argument(
        "--no-demo", action="store_true", help="do not (re)write assets/demo/demo_scenario.wav"
    )


def main(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    lib = load_library(cfg, args.split)
    print(
        f"[scenarios] library: {len(lib.noise)} noise clips, "
        + ", ".join(f"{k}={len(v)}" for k, v in lib.events.items())
    )
    out = Path(args.out) if args.out else paths.SCENARIOS
    idx = generate(
        cfg,
        lib,
        out,
        args.n_events,
        args.n_background,
        args.seed,
        args.duration,
        args.bg_duration,
        split=args.split,
    )
    print(f"[scenarios] events per class: {idx['event_counts']}; per SNR: {idx['snr_counts']} -> {out}")
    if not args.no_demo:
        demo = paths.ASSETS / "demo" / "demo_scenario.wav"
        evs = make_demo(cfg, lib, demo, seed=args.seed)
        print(f"[scenarios] demo: {len(evs)} events -> {demo}")
    return 0
