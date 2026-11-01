"""Append AlertEvents to a JSON-lines log (artifacts/logs/events_<timestamp>.jsonl). Owner: WP5."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from soundwatch import paths
from soundwatch.types import AlertEvent, Level, SoundClass, event_label


def default_log_path() -> Path:
    return paths.LOGS / f"events_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"


def event_to_dict(event: AlertEvent, classes: list[SoundClass] | tuple[SoundClass, ...]) -> dict:
    sc = classes[event.class_id]
    d = {
        "class_id": event.class_id,
        "key": sc.key,
        "label_vi": event_label(event, sc),  # effective label (context-mode override, vn-v3)
        "level": (sc.level if getattr(event, "level", None) is None else Level(event.level)).name,
        "confidence": round(float(event.confidence), 4),
        "t_audio": round(float(event.t_audio), 3),
        "t_wall": float(event.t_wall),
        "repeat": bool(event.repeat),
    }
    if getattr(event, "icon", None):  # only with a mode icon override, so old logs keep their keys
        d["icon"] = event.icon
    return d


class EventLogger:
    """Writes one JSON object per AlertEvent; the file is opened lazily on the first event and
    flushed after every line, so it is readable while the engine runs."""

    def __init__(self, classes: list[SoundClass] | tuple[SoundClass, ...], path: str | Path | None = None):
        self.classes = tuple(classes)
        self.path = Path(path) if path is not None else default_log_path()
        self._f = None
        self._lock = threading.Lock()
        self.count = 0

    def log(self, event: AlertEvent) -> None:
        line = json.dumps(event_to_dict(event, self.classes), ensure_ascii=False)
        with self._lock:
            if self._f is None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._f = open(self.path, "a", encoding="utf-8")
            self._f.write(line + "\n")
            self._f.flush()
            self.count += 1

    def close(self) -> None:
        with self._lock:
            if self._f is not None:
                self._f.close()
                self._f = None

    def __enter__(self) -> EventLogger:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def read_log(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
