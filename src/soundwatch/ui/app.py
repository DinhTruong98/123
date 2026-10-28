"""Watch simulator (pygame). Owner: WP6.

Contract used by `soundwatch run`:
    run_app(cfg, out_queue, in_queue, show_debug=True, max_seconds=None, screenshot_dir=None) -> None
consumes StatusUpdate / AlertEvent / EngineStopped from out_queue (engine -> UI) and
puts AckCommand / SettingsCommand on in_queue (UI -> engine). Returns when the user quits,
when max_seconds elapses, or (if close_on_stop) after EngineStopped.

`python -m soundwatch.ui.app --demo` runs with a fake event generator (no engine, no model).
"""

from __future__ import annotations

import queue

from soundwatch.config import Config


def run_app(cfg: Config, out_queue: queue.Queue, in_queue: queue.Queue, show_debug: bool = True,
            max_seconds: float | None = None, screenshot_dir=None, close_on_stop: bool = False) -> None:
    raise NotImplementedError("WP6")


if __name__ == "__main__":
    raise SystemExit("WP6: not implemented")
