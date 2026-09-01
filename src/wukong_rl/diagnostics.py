"""Read-only A/B probes and offline reports; never inject input or load a model."""
from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

from .config import PipelineConfig
from .profiling import PerformanceSession, RollingStats, TimingProbe, atomic_json, render_report
from .scheduling import wait_until, WindowsTimerResolution


def diagnose(config: PipelineConfig, mode: str, seconds: float = 30, frame_path=None, directory=None) -> Path:
    if mode not in {"baseline", "capture", "observe", "offline"}:
        raise ValueError("unknown diagnostic mode")
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("diagnostic duration must be finite and positive")
    if frame_path and mode != "offline":
        raise ValueError("--frame is only valid with --mode offline")
    source = builder = frame = None
    # Keep baseline and the monitoring worker free of cv2/numpy/torch imports.
    if mode in {"capture", "observe"}:
        from .capture import create_screen_source

        source = create_screen_source(config.capture, diagnostics=True)
    if mode in {"observe", "offline"}:
        from .perception import ScreenPerception
        from .recording import PassiveObservationBuilder, observation_diagnostics

        builder = PassiveObservationBuilder(config, ScreenPerception(config.perception, config.capture.width, config.capture.height))
    if mode == "offline":
        import cv2
        import numpy as np

        frame = cv2.imread(str(frame_path)) if frame_path else np.random.default_rng(19).integers(0, 256, (config.capture.height, config.capture.width, 3), dtype=np.uint8)
        if frame is None or frame.shape[:2] != (config.capture.height, config.capture.width):
            raise ValueError("offline frame must match the configured client dimensions")
    monitor = PerformanceSession(config, f"diagnose-{mode}", enabled=True, directory=directory)
    monitor.start()
    timer_resolution = WindowsTimerResolution()
    timer_resolution.start()
    reason = "completed"
    try:
        startup = TimingProbe()
        if source is not None:
            startup.call("capture_start", source.start)
        monitor.emit("startup", diagnostic_mode=mode, frame_source=str(frame_path) if frame_path else ("synthetic" if mode == "offline" else mode), **startup.payload())
        started = time.perf_counter()
        period = 1 / config.environment.control_hz
        print(f"[diagnose] mode={mode}, duration={seconds}s; no input injection", flush=True)
        while time.perf_counter() - started < seconds:
            probe = TimingProbe()
            details = {"state": "not_observed", "recorded": False}
            if source is not None:
                frame = probe.call("capture_read", source.read)
                details["capture"] = dict(source.last_frame_metadata)
            if builder is not None:
                observation = builder.build(frame, time.monotonic(), probe)
                details.update(observation_diagnostics(observation, source, config.environment.minimum_confidence))
            remaining = period - (time.perf_counter() - probe.started)
            if remaining > 0:
                probe.call("sleep", wait_until, probe.started + period)
            monitor.tick(probe, mode, **details)
    except KeyboardInterrupt:
        reason = "keyboard_interrupt"
    except BaseException as error:
        reason = f"error:{type(error).__name__}"
        monitor.emit("error", error=type(error).__name__, message=str(error))
        raise
    finally:
        try:
            if source is not None:
                source.close()
        finally:
            try:
                monitor.close(reason)
            finally:
                timer_resolution.close()
    return monitor.directory.resolve() / "report.md"


def read_presentmon(path: str | Path, game_process: str) -> dict:
    """Select one application's busiest PID/swapchain; never merge unrelated streams."""
    groups: dict[tuple, dict] = {}
    matched_rows = 0
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        interval_column = next((key for key in ("MsBetweenAppStart", "CPUFrameTime", "MsBetweenPresents") if key in columns), None)
        display_column = next((key for key in ("DisplayedTime", "MsBetweenDisplayChange") if key in columns), None)
        if "Application" not in columns or "ProcessID" not in columns or "SwapChainAddress" not in columns:
            raise ValueError("PresentMon CSV requires Application, ProcessID and SwapChainAddress columns")
        if interval_column is None and display_column is None:
            raise ValueError("unsupported PresentMon timing columns; export console CSV (v1/v2/current)")
        for row in reader:
            name = row.get("Application", "").replace("\\", "/").split("/")[-1]
            if name.casefold() != game_process.casefold():
                continue
            matched_rows += 1
            key = (name, row["ProcessID"], row["SwapChainAddress"])
            group = groups.setdefault(key, {"rows": 0, "app": RollingStats(100_000), "display": RollingStats(100_000), "app_over_33ms": 0, "app_over_50ms": 0})
            group["rows"] += 1
            for metric, column in (("app", interval_column), ("display", display_column)):
                try:
                    value = float(row.get(column, ""))
                except (ValueError, TypeError):
                    continue
                # NA, zero and non-displayed frames must not inflate FPS.
                if math.isfinite(value) and value > 0:
                    group[metric].add(value)
                    if metric == "app":
                        group["app_over_33ms"] += int(value > 1000 / 30)
                        group["app_over_50ms"] += int(value > 50)
    if not groups:
        raise ValueError(f"no PresentMon rows for {game_process}")
    key, selected = max(groups.items(), key=lambda item: (item[1]["app"].count, item[1]["display"].count, item[1]["rows"]))
    app, display = selected["app"].summary(), selected["display"].summary()
    return {
        "source": str(Path(path).resolve()), "application": key[0], "pid": key[1], "swapchain": key[2],
        "selection": "busiest single PID/swapchain by valid application intervals; inspect selection if game uses multiple chains",
        "matched_rows": matched_rows, "selected_rows": selected["rows"], "stream_count": len(groups),
        "app_interval_column": interval_column, "app_interval_ms": app,
        "display_interval_column": display_column, "display_interval_ms": display,
        "app_interval_rate_hz": 1000 / app["mean_all"] if app["mean_all"] else None,
        "display_interval_rate_hz": 1000 / display["mean_all"] if display["mean_all"] else None,
        "app_intervals_over_33ms": selected["app_over_33ms"],
        "app_intervals_over_50ms": selected["app_over_50ms"],
        "quantiles": "last 100000 valid intervals per stream; mean/max/count use entire stream",
        "caveat": "App pacing and displayed intervals are different; generated frames may be included. CSV is not automatically time-aligned with the recorder. Compare matching capture windows/scenes.",
    }


def write_analysis(run, compare=None, presentmon_csv=None, game_process="b1-Win64-Shipping.exe") -> Path:
    root = Path(run).resolve()
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    analysis = {"summary": summary}
    report = render_report(summary)
    if compare:
        baseline = json.loads((Path(compare) / "summary.json").read_text(encoding="utf-8"))
        comparison = {}
        report += "\n## Comparison (current vs reference)\n\nOnly compare matching scenes/settings; a difference alone does not prove causation.\n\n"
        report += "| Metric (p95) | Current | Reference |\n|---|---:|---:|\n"
        # Stage metrics carry a phase prefix; normalize only that prefix for A/B.
        def normalized(stats):
            phases = {"baseline", "capture", "observe", "offline", "waiting", "fighting"}
            present = {key.split(".")[0] for key in stats} & phases
            return {(key.split(".", 1)[1] if len(present) == 1 and key.split(".")[0] in phases else key): value for key, value in stats.items()}
        current_stats, baseline_stats = normalized(summary["stats"]), normalized(baseline["stats"])
        for name in sorted(current_stats.keys() & baseline_stats.keys()):
            current, reference = current_stats[name]["p95"], baseline_stats[name]["p95"]
            comparison[name] = {"current_p95": current, "reference_p95": reference}
            report += f"| {name} | {current} | {reference} |\n"
        analysis["comparison"] = {"reference": str(Path(compare).resolve()), "metrics": comparison}
    if presentmon_csv:
        presentmon = read_presentmon(presentmon_csv, game_process)
        analysis["presentmon"] = presentmon
        report += "\n## External game frame timings (PresentMon)\n\n```json\n" + json.dumps(presentmon, indent=2, ensure_ascii=False) + "\n```\n"
    atomic_json(root / "analysis.json", analysis)
    destination = root / "analysis.md"
    destination.write_text(report, encoding="utf-8")
    return destination
