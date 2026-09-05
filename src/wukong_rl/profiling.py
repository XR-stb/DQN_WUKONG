"""Low-overhead probes; aggregation, resource APIs and disk I/O live off-process."""
from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import platform
import queue
import signal
import time
import uuid
from collections import deque
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .config import MonitoringConfig, PipelineConfig


class TimingProbe:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.started = time.perf_counter()
        self.cpu_started = time.thread_time()
        self.stages: dict[str, dict[str, float]] = {}

    def call(self, name: str, function, *args, **kwargs):
        if not self.enabled:
            return function(*args, **kwargs)
        wall = time.perf_counter()
        cpu = time.thread_time()
        try:
            return function(*args, **kwargs)
        finally:
            elapsed_cpu = (time.thread_time() - cpu) * 1000
            elapsed_wall = (time.perf_counter() - wall) * 1000
            stage = self.stages.setdefault(name, {"wall_ms": 0.0, "thread_cpu_ms": 0.0})
            stage["wall_ms"] += elapsed_wall
            stage["thread_cpu_ms"] += elapsed_cpu

    def payload(self) -> dict:
        return {
            "loop_wall_ms": (time.perf_counter() - self.started) * 1000,
            "loop_thread_cpu_ms": (time.thread_time() - self.cpu_started) * 1000,
            "stages": self.stages,
        }


class RollingStats:
    def __init__(self, window: int = 4096) -> None:
        self.values: deque[float] = deque(maxlen=window)
        self.count = 0
        self.total = 0.0
        self.maximum = -float("inf")

    def add(self, value) -> None:
        if value is None or not math.isfinite(float(value)):
            return
        value = float(value)
        self.values.append(value)
        self.count += 1
        self.total += value
        self.maximum = max(self.maximum, value)

    def summary(self) -> dict:
        values = sorted(self.values)

        def percentile(fraction: float):
            if not values:
                return None
            index = fraction * (len(values) - 1)
            low = int(index)
            high = min(low + 1, len(values) - 1)
            return values[low] + (values[high] - values[low]) * (index - low)

        return {
            "count": self.count,
            "mean_all": self.total / self.count if self.count else None,
            "max_all": self.maximum if self.count else None,
            "quantile_window_samples": len(values),
            "p50": percentile(0.50),
            "p95": percentile(0.95),
            "p99": percentile(0.99),
        }


class PerformanceAggregator:
    def __init__(self, settings: MonitoringConfig, control_hz: float) -> None:
        self.settings = settings
        self.period_ms = 1000 / control_hz
        self.stats: dict[str, RollingStats] = {}
        self.phases: dict[str, dict] = {}
        self.last_tick: dict = {}
        self.latest_resource: dict = {}
        self.dropped_events = 0
        self.saved_episodes = 0
        self.end_reason = "incomplete"
        self.resource_samples = 0

    def add_metric(self, name: str, value) -> None:
        if value is not None:
            self.stats.setdefault(name, RollingStats(self.settings.quantile_window)).add(value)

    def accept(self, event: dict) -> None:
        self.dropped_events = max(self.dropped_events, event.get("dropped_events", 0))
        if event["kind"] == "session_end":
            self.end_reason = event.get("reason", "unknown")
        if event["kind"] == "save":
            self.add_metric("save.wall_ms", event.get("wall_ms"))
            self.saved_episodes += int(event.get("success", False))
        if event["kind"] != "tick":
            return
        self.last_tick = event
        phase_name = event.get("phase", "unknown")
        phase = self.phases.setdefault(phase_name, {
            "ticks": 0, "deadline_misses": 0, "invalid_hp_ticks": 0, "hp_observed_ticks": 0,
            "recorded_transitions": 0, "first_monotonic": event["monotonic"],
            "last_monotonic": event["monotonic"],
            "interval_count": 0, "interval_total_ms": 0.0,
        })
        phase["ticks"] += 1
        phase["last_monotonic"] = event["monotonic"]
        phase["recorded_transitions"] += int(event.get("recorded", False))
        phase["invalid_hp_ticks"] += int(event.get("invalid_hp", False))
        phase["hp_observed_ticks"] += int("invalid_hp" in event)
        interval = event.get("loop_interval_ms")
        if interval is not None:
            phase["interval_count"] += 1
            phase["interval_total_ms"] += interval
        deadline_limit = self.period_ms + self.settings.deadline_tolerance_ms
        phase["deadline_misses"] += int(max(event["loop_wall_ms"], interval or 0) > deadline_limit)
        for key in ("loop_wall_ms", "loop_thread_cpu_ms", "loop_interval_ms", "producer_emit_ms_previous"):
            self.add_metric(f"{phase_name}.{key}", event.get(key))
        for name, values in event.get("stages", {}).items():
            for key, value in values.items():
                self.add_metric(f"{phase_name}.{name}.{key}", value)
        for name, value in event.get("capture", {}).items():
            if isinstance(value, (int, float)) and (name.endswith(("_ms", "_hz")) or name == "repeated_frame"):
                self.add_metric(f"capture.{name}", value)
        for name, value in event.get("input", {}).items():
            self.add_metric(f"input.{name}", value)
        for name, measurement in event.get("hud", {}).items():
            self.add_metric(f"{phase_name}.hud.{name}.confidence", measurement.get("confidence"))
            self.add_metric(f"{phase_name}.hud.{name}.age", measurement.get("age"))

    def accept_resource(self, event: dict) -> None:
        self.latest_resource = event
        self.resource_samples += 1
        for group in ("system", "recorder", "monitor", "gpu"):
            for name, value in (event.get(group) or {}).items():
                if isinstance(value, (int, float)) and name not in {"pid", "index"}:
                    self.add_metric(f"resource.{group}.{name}", value)
        for process in event.get("game", []):
            for name in ("cpu_percent_one_core", "rss_mb", "threads"):
                self.add_metric(f"resource.game.{process['pid']}.{name}", process.get(name))

    def summary(self) -> dict:
        phases = {}
        for name, values in self.phases.items():
            phases[name] = {
                **values,
                "deadline_miss_rate": values["deadline_misses"] / values["ticks"],
                "invalid_hp_rate": values["invalid_hp_ticks"] / values["hp_observed_ticks"] if values["hp_observed_ticks"] else None,
                "observed_loop_hz": (
                    1000 * values["interval_count"] / values["interval_total_ms"]
                    if values["interval_total_ms"] else None
                ),
            }
        warnings = []
        if any(phase["deadline_miss_rate"] > 0.05 for phase in phases.values()):
            warnings.append("Control deadline miss rate exceeds 5%; do not assume 8Hz samples.")
        waiting = phases.get("waiting")
        minimum_waiting_ticks = max(1, round(5000 / self.period_ms))
        if (
            "fighting" not in phases
            and waiting is not None
            and waiting["ticks"] >= minimum_waiting_ticks
        ):
            warnings.append(
                "No fighting phase was detected; this session recorded no demonstration transitions."
            )
        hud_phases = (
            [phases["fighting"]]
            if "fighting" in phases
            else [
                phase
                for name, phase in phases.items()
                if name not in {"waiting", "loading", "paused"}
            ]
        )
        if any((phase["invalid_hp_rate"] or 0) > 0.1 for phase in hud_phases):
            warnings.append("HUD invalid/low-confidence observations exceed 10% in the active phase.")
        input_ages = [self.stats.get(f"input.{key}") for key in ("oldest_pending_age_ms", "consumed_event_age_ms")]
        if any(metric and metric.maximum > self.period_ms for metric in input_ages):
            warnings.append("Input events are older than one tick; action/frame alignment needs inspection.")
        if self.dropped_events:
            warnings.append("Monitoring queue dropped events; timings are incomplete.")
        return {
            "schema_version": 1,
            "end_reason": self.end_reason,
            "target_control_hz": 1000 / self.period_ms,
            "deadline_tolerance_ms": self.settings.deadline_tolerance_ms,
            "quantiles": f"p50/p95/p99 use last {self.settings.quantile_window} samples per metric; mean/max/count use entire run",
            "phases": phases,
            "stats": {name: values.summary() for name, values in self.stats.items()},
            "dropped_events": self.dropped_events,
            "saved_episodes": self.saved_episodes,
            "resource_samples": self.resource_samples,
            "latest_resource": self.latest_resource,
            "last_tick": self.last_tick,
            "game_fps": None,
            "game_fps_source": "not measured; import a game-filtered PresentMon CSV",
            "warnings": warnings,
        }


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def render_report(summary: dict) -> str:
    lines = [
        "# Recording performance report", "",
        f"End reason: {summary['end_reason']}. Game FPS: NOT MEASURED by recorder probes.", "",
        "Loop Hz, capture callbacks/sec and displayed game FPS are different quantities.",
        "Calling-thread CPU time excludes native worker threads. Wall minus thread CPU is not a GIL measurement.",
        "Deadline miss uses max(tick body time, same-phase start-to-start interval), including save/logging gaps, plus configured tolerance.",
        "", summary["quantiles"], "",
        "| Phase | Ticks | Loop Hz | Deadline miss | Invalid HP | Buffered transitions |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, phase in summary["phases"].items():
        hz = phase["observed_loop_hz"]
        hz_label = f"{hz:.2f}" if hz is not None else "N/A"
        invalid_label = f"{phase['invalid_hp_rate']:.1%}" if phase['invalid_hp_rate'] is not None else "N/A"
        lines.append(f"| {name} | {phase['ticks']} | {hz_label} | {phase['deadline_miss_rate']:.1%} | {invalid_label} | {phase['recorded_transitions']} |")
    lines += ["", "## Timings and resource metrics", "", "| Metric | Mean (all) | p50 | p95 | p99 | Max (all) |", "|---|---:|---:|---:|---:|---:|"]
    for name, metric in sorted(summary["stats"].items()):
        values = ["N/A" if metric[key] is None else f"{metric[key]:.3f}" for key in ("mean_all", "p50", "p95", "p99", "max_all")]
        lines.append(f"| {name} | " + " | ".join(values) + " |")
    lines += ["", "## Warnings", ""] + [f"- {warning}" for warning in summary["warnings"]]
    missing = summary.get("latest_resource", {}).get("unavailable", {})
    lines += [f"- {name}: {reason}" for name, reason in missing.items()]
    lines += ["", "Resource logs include the monitor's own CPU/RSS. GPU numbers are device-wide, not per-game.", ""]
    return "\n".join(lines)


def _performance_worker(directory: str, events, stop, ready, parent_pid: int, settings: MonitoringConfig, metadata: dict) -> None:
    from .resources import ResourceSampler

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    dependencies = {}
    for package in ("numpy", "opencv-python", "windows-capture", "psutil", "nvidia-ml-py"):
        try:
            dependencies[package] = version(package)
        except PackageNotFoundError:
            dependencies[package] = None
    atomic_json(root / "run.json", {
        **metadata, "dependencies": dependencies, "platform": platform.platform(),
        "python": platform.python_version(), "cpu_count": os.cpu_count(),
        "monitoring": asdict(settings), "parent_pid": parent_pid,
    })
    aggregator = PerformanceAggregator(settings, metadata["target_control_hz"])
    sampler = ResourceSampler(parent_pid, settings)
    resource_due = time.perf_counter() + settings.resource_interval_seconds
    summary_due = time.perf_counter() + settings.summary_interval_seconds
    ready.set()
    try:
        with (root / "events.jsonl").open("a", encoding="utf-8") as event_file, (root / "resources.jsonl").open("a", encoding="utf-8") as resource_file:
            while True:
                now = time.perf_counter()
                try:
                    event = events.get(timeout=max(0.001, min(0.1, resource_due - now)))
                except queue.Empty:
                    if stop.is_set():
                        break
                    event = False
                if event is None:
                    break
                if event:
                    # Timestamp was taken in the producer, not after queue delay.
                    event["monitor_queue_delay_ms"] = max(0, (time.perf_counter() - event["monotonic"]) * 1000)
                    event_file.write(json.dumps(event, allow_nan=False) + "\n")
                    aggregator.accept(event)
                    aggregator.add_metric("monitor.queue_delay_ms", event["monitor_queue_delay_ms"])
                now = time.perf_counter()
                if now >= resource_due:
                    try:
                        resource = sampler.sample()
                    except Exception as error:
                        resource = {"timestamp": time.time(), "unavailable": {"resource_sampler": str(error)}}
                    aggregator.accept_resource(resource)
                    resource_file.write(json.dumps(resource, allow_nan=False) + "\n")
                    event_file.flush()
                    resource_file.flush()
                    resource_due = time.perf_counter() + settings.resource_interval_seconds
                if now >= summary_due:
                    summary = aggregator.summary()
                    atomic_json(root / "summary.json", summary)
                    tick = summary["last_tick"]
                    phase = summary["phases"].get(tick.get("phase"), {})
                    hz = phase.get("observed_loop_hz")
                    hz_label = f"{hz:.2f}" if hz is not None else "N/A"
                    print(f"[profile] state={tick.get('state', 'starting')} loop_hz={hz_label} ticks={phase.get('ticks', 0)} recorded={phase.get('recorded_transitions', 0)} dropped={summary['dropped_events']}", flush=True)
                    summary_due = time.perf_counter() + settings.summary_interval_seconds
    finally:
        summary = aggregator.summary()
        atomic_json(root / "summary.json", summary)
        (root / "report.md").write_text(render_report(summary), encoding="utf-8")
        sampler.close()


class PerformanceSession:
    def __init__(self, config: PipelineConfig, mode: str, *, enabled: bool | None = None, directory: str | Path | None = None) -> None:
        self.enabled = config.monitoring.enabled if enabled is None else enabled
        self.config = config
        self.mode = mode
        self.directory = Path(directory or config.monitoring.directory) / f"{time.strftime('%Y%m%d-%H%M%S')}-{mode}-{uuid.uuid4().hex[:8]}"
        self.process = None
        self.events = None
        self.stop = None
        self.dropped = 0
        self.last_emit_ms = 0.0
        self.previous_start: float | None = None
        self.previous_phase: str | None = None
        self.closed = False

    def start(self) -> None:
        if not self.enabled:
            return
        context = mp.get_context("spawn")
        self.events = context.Queue(maxsize=self.config.monitoring.queue_capacity)
        self.stop = context.Event()
        ready = context.Event()
        metadata = {
            "schema_version": 1, "mode": self.mode, "started_at": time.time(),
            "config_hash": self.config.fingerprint(), "config": asdict(self.config),
            "target_control_hz": self.config.environment.control_hz,
        }
        self.process = context.Process(target=_performance_worker, args=(str(self.directory), self.events, self.stop, ready, os.getpid(), self.config.monitoring, metadata), name="wukong-profiler", daemon=True)
        self.process.start()
        if not ready.wait(timeout=10):
            self.close("monitor_start_failed")
            raise RuntimeError("performance monitor failed to start")
        # Start the multiprocessing queue feeder before WGC/input hooks run.
        # Creating that thread on the first battle tick would pollute its timing.
        self.emit("session_start", mode=self.mode)
        print(f"[profile] output: {self.directory.resolve()}", flush=True)

    def emit(self, kind: str, **values) -> None:
        if not self.enabled or self.events is None or self.closed:
            return
        started = time.perf_counter()
        event = {"kind": kind, "timestamp": time.time(), "monotonic": started, "dropped_events": self.dropped, **values}
        try:
            self.events.put_nowait(event)
        except queue.Full:
            self.dropped += 1
        self.last_emit_ms = (time.perf_counter() - started) * 1000

    def tick(self, probe: TimingProbe, phase: str, **values) -> None:
        if not self.enabled:
            return
        interval = None
        if self.previous_start is not None and self.previous_phase == phase:
            interval = (probe.started - self.previous_start) * 1000
        self.previous_start, self.previous_phase = probe.started, phase
        self.emit("tick", phase=phase, loop_interval_ms=interval, producer_emit_ms_previous=self.last_emit_ms, **probe.payload(), **values)

    def close(self, reason: str = "completed") -> None:
        if self.closed:
            return
        self.emit("session_end", reason=reason)
        self.closed = True
        if self.process is None:
            return
        try:
            self.events.put(None, timeout=1)
        except queue.Full:
            self.stop.set()
        self.process.join(timeout=5)
        if self.process.is_alive():
            self.stop.set()
            self.process.join(timeout=1)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=2)
            print("[profile] monitor did not stop cleanly; JSONL may be partial", flush=True)
        self.events.cancel_join_thread()
        self.events.close()
