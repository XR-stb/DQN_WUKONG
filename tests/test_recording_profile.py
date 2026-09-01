from __future__ import annotations

from types import SimpleNamespace

import pytest

from conftest import make_observation
from wukong_rl import recording
from wukong_rl.config import load_config
from wukong_rl.types import ActionToken, EpisodeState


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
        def start(self): pass
        def sample(self): return ActionToken.IDLE, "{}"
        def diagnostics(self): return {"pending_events": 0}
        def discard_pending(self): return 0
        def stop(self): self.stopped = True
    def build(*args, **kwargs):
        return make_observation(state=state, frame_shape=(90, 160, 3))
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
    assert events[-1] == ("close", "duration_limit")
    if state is EpisodeState.FIGHTING:
        assert len(saved) == 1 and saved[0][-1].truncated
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
