"""Well-known filesystem locations. All generated files live under artifacts/ (gitignored)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config" / "soundwatch.yaml"
ASSETS = ROOT / "assets"
ICONS = ASSETS / "icons"
FONTS = ASSETS / "fonts"

ARTIFACTS = ROOT / "artifacts"
DATA = ARTIFACTS / "data"
RAW = DATA / "raw"
WINDOWS = DATA / "windows"
NOISE = DATA / "noise"
SCENARIOS = DATA / "scenarios"
CACHE = ARTIFACTS / "cache"
MODELS = ARTIFACTS / "models"
FIRMWARE = ARTIFACTS / "firmware"
GOLDEN = ARTIFACTS / "golden"
REPORTS = ARTIFACTS / "reports"
LOGS = ARTIFACTS / "logs"


def resolve(p: str | Path) -> Path:
    """Resolve a config path relative to the repo root."""
    p = Path(p)
    return p if p.is_absolute() else ROOT / p
