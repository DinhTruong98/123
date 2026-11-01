"""Load and validate config/soundwatch.yaml."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from soundwatch import paths
from soundwatch.types import Level, SoundClass


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int
    window_s: float
    hop_s: float
    n_fft: int
    frame_step: int
    n_mels: int
    fmin: float
    fmax: float
    log_eps: float
    gate_dbfs: float

    @property
    def window_samples(self) -> int:
        return int(round(self.sample_rate * self.window_s))

    @property
    def hop_samples(self) -> int:
        return int(round(self.sample_rate * self.hop_s))

    @property
    def n_frames(self) -> int:
        return 1 + (self.window_samples - self.n_fft) // self.frame_step


@dataclass(frozen=True)
class PostprocessConfig:
    smooth_n: int
    k_of_n: tuple[int, int]
    danger_repeat_s: float
    danger_clear_s: float
    # optional (WP8): a class only counts as "above" if smoothed[c] - smoothed[background] >= bg_margin;
    # None = rule off (default, backward compatible)
    bg_margin: float | None = None
    # optional: a DANGER class whose smoothed prob reaches this in a single hop alerts at once,
    # without waiting for k-of-n (see postprocess.py rule 4b); None = rule off (default)
    danger_fast_threshold: float | None = None
    # optional (vn-v3, postprocess.py rule 10): while an occurrence of one class of `flip_guard_pair` is
    # active, the other class only alerts after its smoothed prob exceeded the active one's by more than
    # `flip_guard_margin` for `flip_guard_hops` consecutive hops; None = rule off (default)
    flip_guard_hops: int | None = None
    flip_guard_margin: float = 0.0
    flip_guard_pair: tuple[str, str] = ("siren", "smoke_alarm")


@dataclass(frozen=True)
class ModelConfig:
    tflite_path: Path
    stats_path: Path
    bundled_tflite_path: Path
    bundled_stats_path: Path

    def resolved_tflite(self) -> Path:
        """Trained model if present, else the bundled demo model."""
        return self.tflite_path if self.tflite_path.exists() else self.bundled_tflite_path

    def resolved_stats(self) -> Path:
        return self.stats_path if self.stats_path.exists() else self.bundled_stats_path


@dataclass(frozen=True)
class UIConfig:
    watch_size: int
    scale: int
    window: tuple[int, int]
    fps: int
    alert_timeout_s: float
    history_len: int
    font: Path
    font_regular: Path


@dataclass(frozen=True)
class ModeSpec:
    """One context mode: classes switched off, per-class level overrides (key -> Level) and (vn-v3)
    per-class display overrides: `label` (key -> text shown instead of label_vi) and `icon`
    (key -> icon file in assets/icons). Absent overrides = the class's own label_vi / icon."""
    disable: tuple[str, ...] = ()
    level: tuple[tuple[str, int], ...] = ()
    label: tuple[tuple[str, str], ...] = ()
    icon: tuple[tuple[str, str], ...] = ()

    @property
    def level_map(self) -> dict[str, int]:
        return dict(self.level)

    @property
    def label_map(self) -> dict[str, str]:
        return dict(self.label)

    @property
    def icon_map(self) -> dict[str, str]:
        return dict(self.icon)


@dataclass(frozen=True)
class AutoModeConfig:
    """`auto`: picks `indoor` / `street` from the background sound (postprocess.py rule 9).

    Level evidence (vn-v2): running median of the per-hop RMS (dBFS) over the last `window_s` seconds,
    "loud" above `street_above_dbfs` + hysteresis_db / 2, "quiet" below it - hysteresis_db / 2
    (`street_above_dbfs: null` = no level evidence).
    Sound evidence (vn-v3, optional, off while `street_evidence` is empty): the model itself hears the
    street. `street_score` = exponential moving average (time constant `evidence_window_s`) of
    sum_k w_k * p_k over the per-hop probabilities, with weights `street_evidence` (class key -> w; e.g.
    car_horn +1 and the household classes baby_cry / door_knock / water_running -1). "Street" evidence
    when street_score >= `street_score_above`, "indoor" evidence when it is below that minus
    `score_hysteresis`.
    Decision: with one kind of evidence, switch to street / indoor on loud / quiet (or street / indoor
    sound evidence). With both: street needs street evidence and a level that is not "quiet"; indoor
    needs indoor evidence and a level that is not "loud".
    No decision before `min_s` seconds of history (stays in the start-up mode), and no automatic switch
    less than `min_dwell_s` after the previous one. MCU cost: the 60-value RMS ring of vn-v2 plus one
    float (street_score) and two timestamps."""
    street_above_dbfs: float | None = -35.0
    window_s: float = 30.0
    hysteresis_db: float = 0.0
    min_s: float = 5.0
    street_evidence: tuple[tuple[str, float], ...] = ()
    street_score_above: float = 0.0
    score_hysteresis: float = 0.0
    evidence_window_s: float = 60.0
    min_dwell_s: float = 0.0

    @property
    def uses_rms(self) -> bool:
        return self.street_above_dbfs is not None

    @property
    def uses_sound(self) -> bool:
        return bool(self.street_evidence)

    @property
    def evidence_map(self) -> dict[str, float]:
        return dict(self.street_evidence)


AUTO_MODE = "auto"
AUTO_LOW_MODE, AUTO_HIGH_MODE = "indoor", "street"


@dataclass(frozen=True)
class ModesConfig:
    """Context modes (optional `modes:` section). `default` = mode at start-up (a named mode or
    'auto'). Absent section = no modes = historical behaviour."""
    default: str
    modes: tuple[tuple[str, ModeSpec], ...]
    auto: AutoModeConfig | None = None

    @property
    def names(self) -> list[str]:
        return [n for n, _ in self.modes] + ([AUTO_MODE] if self.auto is not None else [])

    def spec(self, name: str) -> ModeSpec:
        for n, m in self.modes:
            if n == name:
                return m
        raise KeyError(name)


@dataclass(frozen=True)
class Config:
    audio: AudioConfig
    postprocess: PostprocessConfig
    model: ModelConfig
    ui: UIConfig
    classes: tuple[SoundClass, ...]
    modes: ModesConfig | None = None  # optional context modes (vn-v2); None = off

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @property
    def background_id(self) -> int:
        return len(self.classes) - 1

    @property
    def class_keys(self) -> list[str]:
        return [c.key for c in self.classes]

    def class_by_key(self, key: str) -> SoundClass:
        for c in self.classes:
            if c.key == key:
                return c
        raise KeyError(key)


def _parse_class(i: int, d: dict, check_icons: bool) -> SoundClass:
    try:
        sc = SoundClass(
            id=i,
            key=str(d["key"]),
            label_vi=str(d["label_vi"]),
            level=Level(int(d["level"])),
            icon=str(d["icon"]),
            color=tuple(int(v) for v in d["color"]),  # type: ignore[arg-type]
            vibration=tuple(int(v) for v in d.get("vibration", [])),
            threshold=float(d["threshold"]),
            cooldown_s=float(d["cooldown_s"]),
            enabled=bool(d.get("enabled", True)),
        )
    except (KeyError, ValueError, TypeError) as e:
        raise ConfigError(f"class #{i}: {e!r}") from e
    if len(sc.color) != 3 or not all(0 <= v <= 255 for v in sc.color):
        raise ConfigError(f"class {sc.key}: color must be 3 ints in 0..255")
    if not (0.0 < sc.threshold <= 1.01):
        raise ConfigError(f"class {sc.key}: threshold must be in (0, 1.01]")
    if sc.vibration and len(sc.vibration) % 2 == 0:
        raise ConfigError(f"class {sc.key}: vibration must have odd length (on/off/.../on) or be empty")
    if any(v <= 0 for v in sc.vibration):
        raise ConfigError(f"class {sc.key}: vibration durations must be > 0")
    if check_icons and not (paths.ICONS / sc.icon).exists():
        raise ConfigError(f"class {sc.key}: icon {paths.ICONS / sc.icon} not found")
    return sc


def _parse_modes(raw: dict | None, keys: list[str], check_icons: bool = False) -> ModesConfig | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ConfigError("modes must be a mapping")
    named: list[tuple[str, ModeSpec]] = []
    auto = None
    for name, body in raw.items():
        if name == "default":
            continue
        body = body or {}
        if name == AUTO_MODE:
            try:
                ev = body.get("street_evidence") or {}
                if not isinstance(ev, dict):
                    raise ValueError("street_evidence must be a mapping class -> weight")
                num = {k: (None if v is None else float(v)) for k, v in body.items()
                       if k != "street_evidence"}
                auto = AutoModeConfig(**num, street_evidence=tuple((str(k), float(v)) for k, v in ev.items()))
            except (TypeError, ValueError) as e:
                raise ConfigError(f"modes.auto: {e}") from e
            if auto.window_s <= 0 or auto.evidence_window_s <= 0:
                raise ConfigError("modes.auto.window_s / evidence_window_s must be > 0")
            if not (auto.uses_rms or auto.uses_sound):
                raise ConfigError("modes.auto: needs street_above_dbfs and/or street_evidence")
            for k, _ in auto.street_evidence:
                if k not in keys:
                    raise ConfigError(f"modes.auto.street_evidence: unknown class {k!r}")
            continue
        unknown = set(body) - {"disable", "level", "label", "icon"}
        if unknown:
            raise ConfigError(f"modes.{name}: unknown keys {sorted(unknown)}")
        disable = tuple(str(k) for k in body.get("disable", []) or [])
        level = tuple((str(k), int(v)) for k, v in (body.get("level") or {}).items())
        label = tuple((str(k), str(v)) for k, v in (body.get("label") or {}).items())
        icon = tuple((str(k), str(v)) for k, v in (body.get("icon") or {}).items())
        for k in list(disable) + [k for k, _ in level + label + icon]:
            if k not in keys[:-1]:
                raise ConfigError(f"modes.{name}: unknown alert class {k!r}")
        for k, v in level:
            if v not in (0, 1, 2):
                raise ConfigError(f"modes.{name}.level.{k}: must be 0, 1 or 2")
        for k, v in label:
            if not v.strip():
                raise ConfigError(f"modes.{name}.label.{k}: empty label")
        for k, v in icon:
            if check_icons and not (paths.ICONS / v).exists():
                raise ConfigError(f"modes.{name}.icon.{k}: icon {paths.ICONS / v} not found")
        named.append((str(name), ModeSpec(disable, level, label, icon)))
    names = [n for n, _ in named]
    if auto is not None:
        for need in (AUTO_LOW_MODE, AUTO_HIGH_MODE):
            if need not in names:
                named.append((need, ModeSpec()))
    default = str(raw.get("default", named[0][0] if named else AUTO_MODE))
    m = ModesConfig(default=default, modes=tuple(named), auto=auto)
    if default not in m.names:
        raise ConfigError(f"modes.default {default!r} not among {m.names}")
    return m


def with_mode(cfg: Config, mode: str | None) -> Config:
    """Config whose start-up mode is `mode` (None: unchanged). Raises ConfigError if the config has
    no modes or does not define that mode."""
    import dataclasses

    if mode is None:
        return cfg
    if cfg.modes is None:
        raise ConfigError(f"mode {mode!r} requested but the config has no `modes:` section")
    if mode not in cfg.modes.names:
        raise ConfigError(f"unknown mode {mode!r} (config defines {cfg.modes.names})")
    return dataclasses.replace(cfg, modes=dataclasses.replace(cfg.modes, default=mode))


def load_config(path: str | Path | None = None, check_icons: bool = False) -> Config:
    path = Path(path) if path else paths.CONFIG
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    a = raw["audio"]
    audio = AudioConfig(
        sample_rate=int(a["sample_rate"]),
        window_s=float(a["window_s"]),
        hop_s=float(a["hop_s"]),
        n_fft=int(a["n_fft"]),
        frame_step=int(a["frame_step"]),
        n_mels=int(a["n_mels"]),
        fmin=float(a["fmin"]),
        fmax=float(a["fmax"]),
        log_eps=float(a["log_eps"]),
        gate_dbfs=float(a["gate_dbfs"]),
    )
    p = raw["postprocess"]
    k, n = (int(v) for v in p["k_of_n"])
    if not (1 <= k <= n):
        raise ConfigError("postprocess.k_of_n must satisfy 1 <= k <= n")
    post = PostprocessConfig(
        smooth_n=int(p["smooth_n"]),
        k_of_n=(k, n),
        danger_repeat_s=float(p["danger_repeat_s"]),
        danger_clear_s=float(p["danger_clear_s"]),
        bg_margin=None if p.get("bg_margin") is None else float(p["bg_margin"]),
        danger_fast_threshold=None
        if p.get("danger_fast_threshold") is None
        else float(p["danger_fast_threshold"]),
        flip_guard_hops=None if p.get("flip_guard_hops") is None else int(p["flip_guard_hops"]),
        flip_guard_margin=float(p.get("flip_guard_margin", 0.0)),
        flip_guard_pair=tuple(str(k) for k in p.get("flip_guard_pair", ("siren", "smoke_alarm"))),
    )
    if post.flip_guard_hops is not None and (post.flip_guard_hops < 1 or len(post.flip_guard_pair) != 2):
        raise ConfigError("postprocess.flip_guard_hops must be >= 1 and flip_guard_pair two class keys")
    if post.danger_fast_threshold is not None and not (0.0 < post.danger_fast_threshold <= 1.0):
        raise ConfigError("postprocess.danger_fast_threshold must be in (0, 1]")
    m = raw["model"]
    model = ModelConfig(
        tflite_path=paths.resolve(m["tflite_path"]),
        stats_path=paths.resolve(m["stats_path"]),
        bundled_tflite_path=paths.resolve(m["bundled_tflite_path"]),
        bundled_stats_path=paths.resolve(m["bundled_stats_path"]),
    )
    u = raw["ui"]
    ui = UIConfig(
        watch_size=int(u["watch_size"]),
        scale=int(u["scale"]),
        window=(int(u["window"][0]), int(u["window"][1])),
        fps=int(u["fps"]),
        alert_timeout_s=float(u["alert_timeout_s"]),
        history_len=int(u["history_len"]),
        font=paths.resolve(u["font"]),
        font_regular=paths.resolve(u["font_regular"]),
    )
    classes = tuple(_parse_class(i, d, check_icons) for i, d in enumerate(raw["classes"]))
    keys = [c.key for c in classes]
    if len(set(keys)) != len(keys):
        raise ConfigError("duplicate class keys")
    if not classes or classes[-1].key != "background":
        raise ConfigError("last class must be 'background'")
    if audio.window_samples < audio.n_fft:
        raise ConfigError("window shorter than n_fft")
    if post.flip_guard_hops is not None and not set(post.flip_guard_pair) <= set(keys[:-1]):
        raise ConfigError(f"postprocess.flip_guard_pair: unknown alert class in {post.flip_guard_pair}")
    modes = _parse_modes(raw.get("modes"), keys, check_icons)
    return Config(audio=audio, postprocess=post, model=model, ui=ui, classes=classes, modes=modes)
