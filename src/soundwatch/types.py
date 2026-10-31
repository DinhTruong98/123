"""Shared data types (contract, see docs/IMPLEMENTATION_PLAN.md section 4.3)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np


class Level(IntEnum):
    INFO = 0
    WARN = 1
    DANGER = 2


@dataclass(frozen=True)
class SoundClass:
    id: int
    key: str
    label_vi: str
    level: Level
    icon: str
    color: tuple[int, int, int]
    vibration: tuple[int, ...]
    threshold: float
    cooldown_s: float
    enabled: bool = True


@dataclass(frozen=True)
class AlertEvent:
    class_id: int
    confidence: float
    t_audio: float
    t_wall: float
    repeat: bool = False  # True when a DANGER alert is re-emitted while waiting for ack
    # effective level when context modes are configured (e.g. car_horn -> INFO in `street`);
    # None = the class's configured level (always None without a `modes:` config section)
    level: int | None = None
    # display overrides of the active context mode (vn-v3, config `modes.<mode>.label/icon`), e.g. siren
    # "Xe ưu tiên ..." in `street`; None = the class's own label_vi / icon (always None without overrides)
    label: str | None = None
    icon: str | None = None


def event_label(ev: AlertEvent, sc: SoundClass) -> str:
    """Text to show for an alert: the mode's label override if any, else the class's label_vi."""
    return ev.label if getattr(ev, "label", None) else sc.label_vi


def event_icon(ev: AlertEvent, sc: SoundClass) -> str:
    """Icon file for an alert: the mode's icon override if any, else the class's icon."""
    return ev.icon if getattr(ev, "icon", None) else sc.icon


@dataclass(frozen=True)
class StatusUpdate:
    t_audio: float
    rms_dbfs: float
    probs: np.ndarray = field(repr=False)
    gated: bool
    infer_ms: float
    logmel: np.ndarray | None = field(default=None, repr=False)  # (49, n_mels) un-normalized, for debug view
    mode: str | None = None  # active context mode (None without a `modes:` config), vn-v3
    mode_selector: str | None = None  # requested mode ('auto' or a named mode)


@dataclass(frozen=True)
class AckCommand:
    class_id: int


@dataclass(frozen=True)
class SettingsCommand:
    class_id: int
    enabled: bool


@dataclass(frozen=True)
class ModeCommand:      # UI/app -> engine: switch context mode ('indoor', 'street', 'auto', ...)
    mode: str


@dataclass(frozen=True)
class EngineStopped:
    reason: str = "eof"
