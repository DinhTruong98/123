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
class Config:
    audio: AudioConfig
    postprocess: PostprocessConfig
    model: ModelConfig
    ui: UIConfig
    classes: tuple[SoundClass, ...]

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
    )
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
    return Config(audio=audio, postprocess=post, model=model, ui=ui, classes=classes)
