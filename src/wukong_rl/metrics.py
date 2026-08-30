from __future__ import annotations

import json
import os
import queue
import time
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np


def _json_default(value):
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


class JsonlMetricWriter:
    def __init__(self, directory: str | Path, stream: str) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"{stream}.jsonl"
        self.latest_path = directory / f"{stream}-latest.json"
        self._lock = Lock()

    def write(self, kind: str, **values: Any) -> None:
        event = {"timestamp": time.time(), "kind": kind, **values}
        encoded = json.dumps(event, ensure_ascii=False, default=_json_default)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(encoded + "\n")
            temporary = self.latest_path.with_suffix(".json.tmp")
            temporary.write_text(encoded, encoding="utf-8")
            os.replace(temporary, self.latest_path)


def emit_metric(metric_queue, stream: str, kind: str, **values: Any) -> bool:
    try:
        metric_queue.put_nowait((stream, kind, values))
        return True
    except queue.Full:
        return False


def metric_worker(directory: str | Path, metric_queue) -> None:
    """Single writer process for live Actor/Learner metrics."""
    writers: dict[str, JsonlMetricWriter] = {}
    while True:
        payload = metric_queue.get()
        if payload is None:
            break
        stream, kind, values = payload
        writer = writers.setdefault(stream, JsonlMetricWriter(directory, stream))
        writer.write(kind, **values)
