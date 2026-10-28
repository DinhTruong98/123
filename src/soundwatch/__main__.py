"""CLI entry point: python -m soundwatch <command> [options].

Each command lives in its own module, which exposes
    add_arguments(parser: argparse.ArgumentParser) -> None
    main(args: argparse.Namespace) -> int
Modules are imported lazily so a command only pays for the libraries it uses.
"""

from __future__ import annotations

import argparse
import importlib
import sys

COMMANDS: dict[str, tuple[str, str]] = {
    "doctor": ("soundwatch.tools.doctor", "check environment (TF, YAMNet weights, mic, headless pygame)"),
    "download": ("soundwatch.data.download", "download raw datasets into artifacts/data/raw"),
    "build-dataset": ("soundwatch.data.build_dataset", "build 1 s windows + splits + noise bank"),
    "train-teacher": ("soundwatch.models.teacher", "YAMNet embeddings + classifier head (teacher)"),
    "train-student": ("soundwatch.models.train", "train DS-CNN student (optionally distilled)"),
    "quantize": ("soundwatch.models.quantize", "convert student to int8 / float TFLite"),
    "export": ("soundwatch.models.export_c", "export model + feature params + labels as C headers"),
    "export-icons": ("soundwatch.tools.export_icons", "export icons as XBM / RGB565 C headers"),
    "export-golden": ("soundwatch.tools.export_golden", "export golden feature vectors for firmware tests"),
    "make-scenarios": ("soundwatch.data.scenarios", "synthesize long streaming test scenarios"),
    "eval": ("soundwatch.eval.offline", "offline metrics on test split"),
    "eval-streaming": ("soundwatch.eval.streaming", "event-level metrics on scenarios"),
    "tune": ("soundwatch.eval.tune", "tune post-processing (smoothing, k-of-n, thresholds) on val scenarios"),
    "fa-analysis": ("soundwatch.eval.fa_analysis", "trace background false alarms to source clips + YAMNet"),
    "eval-fast-path": ("soundwatch.eval.fast_path", "before/after metrics of the DANGER fast path"),
    "benchmark": ("soundwatch.eval.benchmark", "model size / MACs / latency benchmark"),
    "report": ("soundwatch.eval.report", "assemble artifacts/reports/report.md"),
    "run": ("soundwatch.engine.run", "run real-time engine (+ watch simulator UI)"),
    "record": ("soundwatch.tools.record", "record + label custom clips from the microphone"),
}


def _usage() -> str:
    lines = ["usage: python -m soundwatch <command> [options]", "", "commands:"]
    lines += [f"  {name:<16}{desc}" for name, (_, desc) in COMMANDS.items()]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(_usage())
        return 0
    name, rest = argv[0], argv[1:]
    if name not in COMMANDS:
        print(f"unknown command: {name}\n\n{_usage()}", file=sys.stderr)
        return 2
    module = importlib.import_module(COMMANDS[name][0])
    parser = argparse.ArgumentParser(prog=f"python -m soundwatch {name}", description=COMMANDS[name][1])
    parser.add_argument("--config", default=None, help="config YAML (default: config/soundwatch.yaml)")
    module.add_arguments(parser)
    args = parser.parse_args(rest)
    return int(module.main(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
