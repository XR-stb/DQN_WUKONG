from __future__ import annotations

from types import SimpleNamespace

import pytest

from conftest import make_observation
from wukong_rl import recording
from wukong_rl.config import load_config
from wukong_rl.types import ActionCommand, EpisodeState


@pytest.mark.parametrize("state", [EpisodeState.WAITING, EpisodeState.FIGHTING])
def test_recording_profiles_waiting_and_saves_active_episode(tmp_path, monkeypatch, state):
    ticks, events, saved = [], [], []
    class Clock:
        now = 0.0
        def monotonic(self): return self.now
        perf_counter = monotonic
        def sleep(self, seconds): self.now += seconds
    class Monitor:
        enabled = True
        def __init__(self, *args, **kwargs): pass
        def start(self): pass
        def emit(self, kind, **values): events.append((kind, values))
        def tick(self, probe, phase, **values): ticks.append((phase, values))
        def close(self, reason): events.append(("close", reason))
    class Source:
        closed = False
        def start(self): pass
        def read(self): return None
        def close(self): self.closed = True
    class Observer:
        stopped = False
        control_calls = 0
        def start(self): pass
        def sample(self): return ActionCommand(), "{}"
        def diagnostics(self): return {"pending_events": 0}
        def discard_pending(self): return 0
        def consume_control_requests(self):
            self.control_calls += 1
            return False, state is EpisodeState.WAITING and self.control_calls >= 4
        def stop(self): self.stopped = True
    def build(_frame, timestamp, *_args, **_kwargs):
        observation = make_observation(state=state, frame_shape=(90, 160, 3))
        observation.timestamp = timestamp
        # Model capture/perception work after an observation timestamp. The
        # fixed-rate scheduler must absorb this cost after its first tick.
        recording.time.now += 0.01
        return observation
    builder = SimpleNamespace(build=build)
    source, observer = Source(), Observer()
    monkeypatch.setattr(recording, "time", Clock())
    monkeypatch.setattr(recording, "wait_until", lambda deadline, **kwargs: setattr(recording.time, "now", deadline))
    monkeypatch.setattr(recording, "PerformanceSession", Monitor)
    monkeypatch.setattr(recording, "PassiveObservationBuilder", lambda *args: builder)
    def save(*args):
        saved.append(list(args[2]))
        return tmp_path / "episode"
    monkeypatch.setattr(recording, "save_episode", save)
    recording.record_demonstrations(load_config(), "yinhu", tmp_path, duration_seconds=0.3, source=source, observer=observer)
    assert source.closed and observer.stopped
    assert ticks and all(phase == state.value for phase, _ in ticks)
    expected_reason = "duration_limit" if state is EpisodeState.FIGHTING else "hotkey_stop"
    assert events[-1] == ("close", expected_reason)
    if state is EpisodeState.FIGHTING:
        assert len(saved) == 1 and saved[0][-1].truncated
        next_timestamps = [item.next_observation.timestamp for item in saved[0]]
        assert len(next_timestamps) >= 2
        assert all(
            later - earlier == pytest.approx(0.125)
            for earlier, later in zip(next_timestamps, next_timestamps[1:])
        )
        assert all(event["recorded"] for _, event in ticks)
        assert any(kind == "save" and value["success"] for kind, value in events if isinstance(value, dict))
    else:
        assert not saved
        assert all(not event["recorded"] for _, event in ticks)


def test_capture_error_closes_recorder_and_profiles_error(tmp_path, monkeypatch):
    events = []
    class Monitor:
        enabled = True
        def __init__(self, *args, **kwargs): pass
        def start(self): pass
        def emit(self, kind, **values): events.append((kind, values))
        def close(self, reason): events.append(("close", reason))
    def fail(): raise RuntimeError("capture failed")
    source = SimpleNamespace(start=fail, close=lambda: events.append(("source_closed", True)))
    observer = SimpleNamespace(stop=lambda: events.append(("observer_stopped", True)))
    monkeypatch.setattr(recording, "PerformanceSession", Monitor)
    with pytest.raises(RuntimeError, match="capture failed"):
        recording.record_demonstrations(load_config(), "yinhu", tmp_path, source=source, observer=observer)
    assert events[-1] == ("close", "error:RuntimeError")
    assert ("source_closed", True) in events
    assert ("observer_stopped", True) in events


def test_bounded_recording_stops_when_episode_finishes_early(tmp_path, monkeypatch):
    events, saved = [], []

    class Clock:
        now = 0.0
        perf_counter = lambda self: self.now
        monotonic = perf_counter
        def sleep(self, seconds): self.now += seconds

    class Monitor:
        enabled = True
        def __init__(self, *args, **kwargs): pass
        def start(self): pass
        def emit(self, kind, **values): events.append((kind, values))
        def tick(self, *args, **kwargs): pass
        def close(self, reason): events.append(("close", reason))

    class Observer:
        def start(self): pass
        def sample(self): return ActionCommand(), "{}"
        def diagnostics(self): return {"pending_events": 0}
        def discard_pending(self): return 0
        def consume_control_requests(self): return False, False
        def stop(self): pass

    class Source:
        last_frame_metadata = {}
        def start(self): pass
        def read(self): return None
        def close(self): pass

    calls = 0
    def build(_frame, timestamp, *_args, **_kwargs):
        nonlocal calls
        state = EpisodeState.FIGHTING if calls == 0 else EpisodeState.LOST
        calls += 1
        observation = make_observation(state=state, frame_shape=(90, 160, 3))
        observation.timestamp = timestamp
        return observation

    monkeypatch.setattr(recording, "time", Clock())
    monkeypatch.setattr(
        recording,
        "wait_until",
        lambda deadline, **kwargs: setattr(recording.time, "now", deadline),
    )
    monkeypatch.setattr(recording, "PerformanceSession", Monitor)
    monkeypatch.setattr(
        recording, "PassiveObservationBuilder", lambda *args: SimpleNamespace(build=build)
    )
    monkeypatch.setattr(
        recording,
        "save_episode",
        lambda *args: saved.append(list(args[2])) or tmp_path / "episode",
    )
    recording.record_demonstrations(
        load_config(),
        "yinhu",
        tmp_path,
        duration_seconds=60,
        source=Source(),
        observer=Observer(),
    )
    assert len(saved) == 1
    assert saved[0][-1].terminated
    assert events[-1] == ("close", "episode_complete")
