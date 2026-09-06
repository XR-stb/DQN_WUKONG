from __future__ import annotations

import json
import os
import queue
import sys
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
        self.latest_snapshot_failures = 0

    def _replace_latest(self, encoded: str) -> None:
        temporary = self.latest_path.with_name(
            f"{self.latest_path.name}.{os.getpid()}.tmp"
        )
        temporary.write_text(encoded, encoding="utf-8")
        last_error: OSError | None = None
        for delay in (0.0, 0.01, 0.03, 0.06, 0.1):
            if delay:
                time.sleep(delay)
            try:
                os.replace(temporary, self.latest_path)
                return
            except OSError as exc:
                # Antivirus, editors and file indexers can briefly hold the
                # destination open on Windows. The append-only JSONL stream is
                # authoritative, so a stale convenience snapshot must never
                # kill the metrics process.
                last_error = exc
        self.latest_snapshot_failures += 1
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        print(
            f"[metrics] latest snapshot update failed for {self.latest_path}: {last_error}",
            file=sys.stderr,
            flush=True,
        )

    def write(self, kind: str, **values: Any) -> None:
        event = {"timestamp": time.time(), "kind": kind, **values}
        encoded = json.dumps(event, ensure_ascii=False, default=_json_default)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(encoded + "\n")
            self._replace_latest(encoded)


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
        try:
            writer.write(kind, **values)
        except Exception as exc:
            # Monitoring is observational. A malformed event or transient disk
            # problem must be visible, but must not permanently stop all later
            # Actor/Learner metrics.
            print(
                f"[metrics] dropped {stream}/{kind}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
