"""Time-based vibration simulation (no sleep). Mirrors a motor driven by an on/off ms pattern."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VibrationState:
    active: bool  # pattern still playing
    on: bool  # motor on right now
    segment: int  # index into the pattern (-1 when finished / no pattern)
    elapsed_ms: float  # time since pattern start
    total_ms: float  # pattern length

    @property
    def progress(self) -> float:
        return 0.0 if self.total_ms <= 0 else min(1.0, self.elapsed_ms / self.total_ms)


def vibration_state(pattern: tuple[int, ...] | list[int], elapsed_ms: float) -> VibrationState:
    """State of `pattern` (alternating on/off ms, starting with on) `elapsed_ms` after start."""
    total = float(sum(pattern))
    if not pattern or elapsed_ms < 0 or elapsed_ms >= total:
        return VibrationState(False, False, -1, max(0.0, elapsed_ms), total)
    acc = 0.0
    for i, d in enumerate(pattern):
        acc += d
        if elapsed_ms < acc:
            return VibrationState(True, i % 2 == 0, i, elapsed_ms, total)
    return VibrationState(False, False, -1, elapsed_ms, total)  # pragma: no cover


class VibrationPlayer:
    """Plays one pattern at a time; start() restarts (used for repeated DANGER alerts)."""

    def __init__(self) -> None:
        self.pattern: tuple[int, ...] = ()
        self.started_at: float | None = None
        self.class_id: int | None = None

    def start(self, pattern: tuple[int, ...] | list[int], now: float, class_id: int | None = None) -> None:
        self.pattern = tuple(pattern)
        self.started_at = now
        self.class_id = class_id

    def stop(self) -> None:
        self.started_at = None

    def state(self, now: float) -> VibrationState:
        if self.started_at is None:
            return VibrationState(False, False, -1, 0.0, float(sum(self.pattern)))
        return vibration_state(self.pattern, (now - self.started_at) * 1000.0)
