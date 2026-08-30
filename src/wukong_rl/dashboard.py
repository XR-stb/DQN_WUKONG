from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np


def load_events(path: str | Path, limit: int = 5000) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    events: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in deque(handle, maxlen=limit):
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def _series(events: list[dict], x_key: str, y_key: str) -> tuple[np.ndarray, np.ndarray]:
    points = [
        (event[x_key], event[y_key])
        for event in events
        if x_key in event and y_key in event
    ]
    if not points:
        return np.asarray([]), np.asarray([])
    return np.asarray([point[0] for point in points]), np.asarray([point[1] for point in points])


def run_dashboard(metrics_directory: str | Path = "artifacts/metrics", refresh_seconds: float = 2.0) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
    except ImportError as error:
        raise RuntimeError("install the dashboard extra: pip install -e '.[dashboard]'") from error

    metrics_directory = Path(metrics_directory)
    figure, axes = plt.subplots(2, 4, figsize=(18, 9))
    figure.suptitle("Wukong R2D3 training monitor")
    plots = (
        ("actor", "environment_steps", "environment_fps", "Environment FPS"),
        ("actor", "environment_steps", "deadline_miss_rate", "Deadline miss rate"),
        ("actor", "environment_steps", "detection_confidence", "Detection confidence"),
        ("actor", "environment_steps", "actor_latency_ms", "Actor inference ms"),
        ("train", "learner_steps", "td_loss", "TD loss"),
        ("train", "learner_steps", "mean_q", "Mean Q"),
        ("train", "learner_steps", "learner_latency_ms", "Learner update ms"),
        ("train", "learner_steps", "replay_sequences", "Replay sequences"),
    )

    def redraw(_frame) -> None:
        streams = {
            "actor": load_events(metrics_directory / "actor.jsonl"),
            "train": load_events(metrics_directory / "train.jsonl"),
        }
        for axis, (stream, x_key, y_key, title) in zip(axes.flat, plots):
            axis.clear()
            x, y = _series(streams[stream], x_key, y_key)
            if x.size:
                axis.plot(x, y, linewidth=1.2)
            axis.set_title(title)
            axis.grid(alpha=0.25)

    animation = FuncAnimation(
        figure,
        redraw,
        interval=max(int(refresh_seconds * 1000), 250),
        cache_frame_data=False,
    )
    figure._wukong_animation = animation
    plt.tight_layout()
    plt.show()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Live JSONL training dashboard")
    parser.add_argument("--metrics", default="artifacts/metrics")
    parser.add_argument("--refresh-seconds", type=float, default=2.0)
    args = parser.parse_args(argv)
    run_dashboard(args.metrics, args.refresh_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
