"""Compatibility entry point for the rebuilt training pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parent / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from wukong_rl.cli import main


if __name__ == "__main__":
    arguments = sys.argv[1:]
    commands = {"calibrate", "record", "pretrain", "train", "eval", "benchmark", "migrate-config"}
    if not arguments:
        arguments = ["train"]
    elif not any(argument in commands for argument in arguments):
        if len(arguments) >= 2 and arguments[0] == "--config":
            arguments = [arguments[0], arguments[1], "train", *arguments[2:]]
        else:
            arguments = ["train", *arguments]
    raise SystemExit(main(arguments))
