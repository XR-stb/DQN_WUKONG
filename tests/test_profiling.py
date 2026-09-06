from __future__ import annotations

import json
import queue
import subprocess
import sys
import time
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest

from wukong_rl.config import MonitoringConfig, load_config
from wukong_rl.diagnostics import read_presentmon, write_analysis
from wukong_rl.profiling import PerformanceAggregator, PerformanceSession, RollingStats, TimingProbe


def test_rolling_quantiles_are_bounded_and_totals_are_not():
    stats = RollingStats(3)
    for number in (1, 2, None, float("nan"), 3, 4):
        stats.add(number)
    result = stats.summary()
    assert result["count"] == 4
    assert result["mean_all"] == 2.5
    assert result["p50"] == 3
    assert result["p95"] == pytest.approx(3.9)
    assert result["quantile_window_samples"] == 3


def test_probe_records_exception_and_disabled_probe_is_transparent():
    probe = TimingProbe()
    with pytest.raises(ValueError):
        probe.call("bad", int, "not a number")
    assert probe.stages["bad"]["wall_ms"] >= 0
    assert probe.stages["bad"]["thread_cpu_ms"] >= 0
    disabled = TimingProbe(False)
    assert disabled.call("sum", sum, [1, 2]) == 3
    assert disabled.stages == {}


def test_aggregator_includes_waiting_save_gaps_and_input_age():
    aggregate = PerformanceAggregator(MonitoringConfig(), 8)
    event = dict(kind="tick", monotonic=1, loop_wall_ms=120, loop_interval_ms=None, phase="waiting", invalid_hp=True,
                 input={"consumed_event_age_ms": 900}, capture={"callback_sequence": 123, "repeated_frame": True})
    aggregate.accept(event)
    aggregate.accept({**event, "monotonic": 2, "phase": "fighting", "recorded": True, "invalid_hp": False, "loop_interval_ms": 800})
    aggregate.accept({"kind": "save", "success": True, "wall_ms": 500})
    aggregate.accept({"kind": "session_end", "reason": "keyboard_interrupt", "dropped_events": 2})
    summary = aggregate.summary()
    assert summary["phases"]["waiting"]["invalid_hp_rate"] == 1
    assert summary["phases"]["fighting"]["deadline_misses"] == 1
    assert summary["phases"]["fighting"]["recorded_transitions"] == 1
    assert summary["game_fps"] is None
    assert summary["saved_episodes"] == 1
    assert summary["dropped_events"] == 2
    assert "capture.callback_sequence" not in summary["stats"]
    assert any("Input events" in warning for warning in summary["warnings"])
    assert not any("HUD invalid" in warning for warning in summary["warnings"])


def test_aggregator_warns_when_recording_never_enters_fighting():
    aggregate = PerformanceAggregator(MonitoringConfig(), 8)
    for index in range(40):
        aggregate.accept(
            dict(
                kind="tick",
                monotonic=index / 8,
                loop_wall_ms=1,
                phase="waiting",
                invalid_hp=True,
                recorded=False,
            )
        )
    assert any("No fighting phase" in warning for warning in aggregate.summary()["warnings"])


def test_full_queue_drops_without_blocking():
    monitor = PerformanceSession(load_config(), "test")
    monitor.events = queue.Queue(maxsize=1)
    monitor.emit("first")
    monitor.emit("second")
    assert monitor.dropped == 1
    assert monitor.events.get_nowait()["kind"] == "first"
    monitor.emit("third")
    assert monitor.events.get_nowait()["dropped_events"] == 1


def test_monitoring_settings_do_not_invalidate_existing_hash():
    import hashlib

    config = load_config()
    payload = asdict(config)
    payload.pop("monitoring")
    old_hash = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]
    config.monitoring.enabled = False
    config.monitoring.directory = "elsewhere"
    assert config.fingerprint() == old_hash
    config.monitoring.resource_interval_seconds = 0.01
    with pytest.raises(ValueError, match="sampling interval"):
        config.validate()


def test_windows_spawn_worker_writes_report_even_without_fighting(tmp_path):
    config = load_config()
    config.monitoring.gpu_enabled = False
    config.monitoring.resource_interval_seconds = 0.5
    config.monitoring.summary_interval_seconds = 1
    session = PerformanceSession(config, "test", directory=tmp_path)
    session.start()
    try:
        session.tick(TimingProbe(), "waiting", state="waiting", invalid_hp=True, recorded=False)
        time.sleep(0.65)
    finally:
        session.close("duration_limit")
    assert not session.process.is_alive()
    summary = json.loads((session.directory / "summary.json").read_text())
    assert summary["phases"]["waiting"]["ticks"] == 1
    assert summary["resource_samples"] >= 1
    assert summary["end_reason"] == "duration_limit"
    assert "NOT MEASURED" in (session.directory / "report.md").read_text()
    assert (session.directory / "run.json").is_file()
    assert write_analysis(session.directory).is_file()


def test_cli_and_recording_do_not_import_training_stack():
    result = subprocess.run([sys.executable, "-c", "import sys; import wukong_rl.cli; import wukong_rl.profiling; assert 'numpy' not in sys.modules; assert 'cv2' not in sys.modules; import wukong_rl.recording; assert 'torch' not in sys.modules; assert 'pynput' not in sys.modules; from wukong_rl import Observation; assert Observation.__name__ == 'Observation'"], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("column", ["CPUFrameTime", "MsBetweenPresents", "MsBetweenAppStart"])
def test_presentmon_selects_one_game_swapchain_and_valid_intervals(tmp_path, column):
    path = tmp_path / "frames.csv"
    path.write_text(f"Application,ProcessID,SwapChainAddress,{column},DisplayedTime\n"
                    "other.exe,1,A,1,1\n"
                    "b1-Win64-Shipping.exe,2,A,10,12\n"
                    "b1-Win64-Shipping.exe,2,A,20,NA\n"
                    "b1-Win64-Shipping.exe,2,A,60,70\n"
                    "b1-Win64-Shipping.exe,2,A,0,0\n"
                    "b1-Win64-Shipping.exe,2,B,1,1\n", encoding="utf-8")
    report = read_presentmon(path, "b1-Win64-Shipping.exe")
    assert report["swapchain"] == "A"
    assert report["stream_count"] == 2
    assert report["app_interval_ms"]["count"] == 3
    assert report["app_interval_rate_hz"] == pytest.approx(1000 / 30)
    assert report["display_interval_ms"]["count"] == 2
    assert report["app_intervals_over_50ms"] == 1
    with pytest.raises(ValueError, match="no PresentMon rows"):
        read_presentmon(path, "missing.exe")


def test_capture_metadata_pairs_sequence_and_frame(monkeypatch):
    from wukong_rl.capture import LegacyScreenSource

    config = load_config()
    frame = np.zeros((720, 1280, 4), np.uint8)
    backend = SimpleNamespace(_use_wgc=True, grab_screen=lambda with_metadata: (frame, {"callback_sequence": 3, "callback_timestamp": time.perf_counter()}))
    monkeypatch.setitem(sys.modules, "grabscreen", backend)
    source = LegacyScreenSource(config.capture, diagnostics=True)
    source._started = True
    assert source.read().shape == (720, 1280, 3)
    assert source.last_frame_metadata["repeated_frame"] is False
    source.read()
    assert source.last_frame_metadata["repeated_frame"] is True
    assert source.last_frame_metadata["frame_age_ms"] >= 0


def test_input_diagnostics_coalesce_to_latest_interval_event():
    from wukong_rl.recording import HumanInputObserver
    from wukong_rl.types import CombatToken

    observer = HumanInputObserver()
    observer._latched.append((CombatToken.LIGHT_ATTACK, time.perf_counter() - 1))
    observer._latched.append((CombatToken.DODGE, time.perf_counter() - 0.01))
    assert observer.diagnostics()["oldest_pending_age_ms"] >= 1000
    assert observer.sample()[0].combat is CombatToken.DODGE
    diagnostics = observer.diagnostics()
    assert 0 <= diagnostics["consumed_event_age_ms"] < 250
    assert diagnostics["pending_events"] == 0
    assert diagnostics["coalesced_events_total"] == 1
    observer._latched.append((CombatToken.LIGHT_ATTACK, time.perf_counter()))
    assert observer.discard_pending() == 1
    assert observer.diagnostics()["discarded_events_total"] == 1
    observer._control_requests.extend(("toggle", "toggle", "stop"))
    assert observer.consume_control_requests() == (False, True)


def test_wgc_uses_requested_minimum_update_interval(monkeypatch):
    import grabscreen

    received = {}
    class FakeCapture:
        def __init__(self, **kwargs): received.update(kwargs)
        def event(self, function): return function
        def start(self): return None
    monkeypatch.setattr(grabscreen, "WindowsCapture", FakeCapture)
    monkeypatch.setattr(grabscreen, "_GAME_WINDOW_TITLE", "test")
    grabscreen._wgc_minimum_update_interval_ms = 34
    grabscreen._start_wgc_capture()
    assert received["minimum_update_interval"] == 34
    assert received["cursor_capture"] is False


def test_profile_dashboard_renders_bounded_summary(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.animation
    from wukong_rl.dashboard import run_profile_dashboard

    aggregate = PerformanceAggregator(MonitoringConfig(), 8)
    aggregate.accept(dict(kind="tick", monotonic=1, phase="capture", loop_wall_ms=140, stages={"capture_read": {"wall_ms": 30}}, state="not_observed"))
    (tmp_path / "summary.json").write_text(json.dumps(aggregate.summary()), encoding="utf-8")
    monkeypatch.setattr(matplotlib.animation, "FuncAnimation", lambda figure, callback, **kwargs: callback(0))
    monkeypatch.setattr(plt, "show", lambda: None)
    run_profile_dashboard(tmp_path)
    figure = plt.gcf()
    figure.canvas.draw()
    assert len(figure.axes[0].patches) == 1
    assert "invalid HP=N/A" in figure.axes[1].texts[0].get_text()
    plt.close(figure)
