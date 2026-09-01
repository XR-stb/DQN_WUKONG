from __future__ import annotations

import json
import math
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from .actions import build_action_mask
from .capture import create_screen_source
from .config import PipelineConfig
from .data import save_episode
from .perception import ScreenPerception, TerminalStateMachine
from .reward import OutcomeReward
from .profiling import PerformanceSession, TimingProbe
from .scheduling import wait_until, WindowsTimerResolution
from .types import ActionToken, EpisodeState, Observation, Transition, measurements_to_arrays


class HumanInputObserver:
    """Observe native controls without injecting or suppressing user input."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._held_keys: set[str] = set()
        self._held_buttons: set[str] = set()
        self._latched: deque[tuple[ActionToken, float]] = deque()
        self._last_event_age_ms: float | None = None
        self._coalesced_events = 0
        self._discarded_events = 0
        self._control_requests: deque[str] = deque()
        self._keyboard_listener = None
        self._mouse_listener = None

    @staticmethod
    def _key_name(key) -> str:
        if hasattr(key, "char") and key.char:
            return str(key.char).lower()
        return str(key).replace("Key.", "").lower()

    def start(self) -> None:
        from pynput import keyboard, mouse

        pulse_keys = {
            "space": ActionToken.DODGE,
            "1": ActionToken.SKILL_1,
            "2": ActionToken.SKILL_2,
            "3": ActionToken.SKILL_3,
            "4": ActionToken.SKILL_4,
            "t": ActionToken.FABAO,
            "f": ActionToken.TISHEN,
            "r": ActionToken.DRINK_POTION,
        }

        def on_press(key) -> None:
            name = self._key_name(key)
            with self._lock:
                if name not in self._held_keys and name == "f8":
                    self._control_requests.append("toggle")
                elif name not in self._held_keys and name == "f9":
                    self._control_requests.append("stop")
                if name not in self._held_keys and name in pulse_keys:
                    self._latched.append((pulse_keys[name], time.perf_counter()))
                self._held_keys.add(name)

        def on_release(key) -> None:
            with self._lock:
                self._held_keys.discard(self._key_name(key))

        def on_click(_x, _y, button, pressed) -> None:
            name = str(button).replace("Button.", "").lower()
            with self._lock:
                if pressed:
                    self._held_buttons.add(name)
                    if name == "left":
                        self._latched.append((ActionToken.LIGHT_ATTACK, time.perf_counter()))
                else:
                    self._held_buttons.discard(name)

        self._keyboard_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self._mouse_listener = mouse.Listener(on_click=on_click)
        self._keyboard_listener.start()
        self._mouse_listener.start()

    def sample(self) -> tuple[ActionToken, str]:
        with self._lock:
            self._last_event_age_ms = None
            if self._latched:
                # One token represents one control interval. Multiple click/key
                # pulses inside it cannot be represented separately, so retain
                # the most recent pulse and never replay stale FIFO events.
                self._coalesced_events += max(0, len(self._latched) - 1)
                action, event_time = self._latched[-1]
                self._latched.clear()
                self._last_event_age_ms = (time.perf_counter() - event_time) * 1000
            elif "right" in self._held_buttons:
                action = ActionToken.HEAVY_HOLD
            else:
                action = ActionToken.IDLE
                for key, movement in (
                    ("w", ActionToken.RUN_FORWARD),
                    ("s", ActionToken.RUN_BACK),
                    ("a", ActionToken.RUN_LEFT),
                    ("d", ActionToken.RUN_RIGHT),
                ):
                    if key in self._held_keys:
                        action = movement
                        break
            raw_input = json.dumps(
                {
                    "keys": sorted(self._held_keys),
                    "buttons": sorted(self._held_buttons),
                    "token": action.name,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            return action, raw_input

    def diagnostics(self) -> dict:
        with self._lock:
            return {
                "pending_events": len(self._latched),
                "oldest_pending_age_ms": (time.perf_counter() - self._latched[0][1]) * 1000 if self._latched else 0.0,
                "consumed_event_age_ms": self._last_event_age_ms,
                "coalesced_events_total": self._coalesced_events,
                "discarded_events_total": self._discarded_events,
                "pending_control_requests": len(self._control_requests),
            }

    def consume_control_requests(self) -> tuple[bool, bool]:
        with self._lock:
            toggle = sum(request == "toggle" for request in self._control_requests) % 2 == 1
            stop = any(request == "stop" for request in self._control_requests)
            self._control_requests.clear()
            return toggle, stop

    def discard_pending(self) -> int:
        """Discard pulses captured outside a fighting transition interval."""
        with self._lock:
            count = len(self._latched)
            self._latched.clear()
            self._discarded_events += count
            self._last_event_age_ms = None
            return count

    def snapshot(self) -> ActionToken:
        """Compatibility helper for callers that need only the action token."""
        return self.sample()[0]

    def stop(self) -> None:
        if self._keyboard_listener is not None:
            self._keyboard_listener.stop()
        if self._mouse_listener is not None:
            self._mouse_listener.stop()


class PassiveObservationBuilder:
    def __init__(self, config: PipelineConfig, perception: ScreenPerception) -> None:
        self.config = config
        self.perception = perception
        self.terminal = TerminalStateMachine(config.environment)
        self.previous_action = ActionToken.IDLE
        self.previous_reward = 0.0

    def reset(self) -> None:
        self.perception.reset()
        self.terminal.reset()
        self.previous_action = ActionToken.IDLE
        self.previous_reward = 0.0

    def build(self, frame: np.ndarray, timestamp: float, probe: TimingProbe | None = None) -> Observation:
        call = probe.call if probe else lambda _name, function, *args, **kwargs: function(*args, **kwargs)
        measurements = call("detect", self.perception.detect, frame)
        state = call("terminal", self.terminal.update, measurements, timestamp)
        features, confidence = call("hud_features", measurements_to_arrays, measurements)
        mask = call("action_mask", build_action_mask, measurements, self.config.environment.minimum_confidence)
        rgb = call("color_convert", cv2.cvtColor, frame[:, :, :3], cv2.COLOR_BGR2RGB)
        rgb = call("resize", cv2.resize,
            rgb,
            (self.config.capture.observation_width, self.config.capture.observation_height),
            interpolation=cv2.INTER_AREA,
        )
        return call("observation_pack", Observation,
            frame=np.ascontiguousarray(rgb, dtype=np.uint8),
            features=features,
            feature_confidence=confidence,
            action_mask=mask,
            timestamp=timestamp,
            episode_state=state,
            previous_action=self.previous_action,
            previous_reward=self.previous_reward,
            measurements=measurements,
        )


def observation_diagnostics(observation: Observation, source, minimum_confidence: float = 0.55) -> dict:
    hud = {name: {"value": value.value, "confidence": value.confidence, "age": value.age, "valid": value.valid} for name, value in observation.measurements.items()}
    capture = dict(getattr(source, "last_frame_metadata", {}))
    arrived = capture.get("callback_timestamp")
    if arrived is not None:
        capture["observation_age_ms"] = (time.perf_counter() - arrived) * 1000
    health = [observation.measurements.get(name) for name in ("self_blood", "boss_blood")]
    return {
        "state": observation.episode_state.value,
        "hud": hud,
        "capture": capture,
        "invalid_hp": any(value is None or not value.valid or value.confidence < minimum_confidence for value in health),
    }


def record_demonstrations(
    config: PipelineConfig,
    boss_id: str,
    output: str | Path,
    *,
    profile_enabled: bool | None = None,
    profile_directory: str | Path | None = None,
    duration_seconds: float | None = None,
    start_paused: bool = False,
    source=None,
    observer=None,
) -> None:
    if duration_seconds is not None and (not math.isfinite(duration_seconds) or duration_seconds <= 0):
        raise ValueError("recording duration must be finite and positive")
    monitor = PerformanceSession(config, "record", enabled=profile_enabled, directory=profile_directory)
    source = source or create_screen_source(config.capture, diagnostics=monitor.enabled)
    perception = ScreenPerception(
        config.perception, config.capture.width, config.capture.height
    )
    builder = PassiveObservationBuilder(config, perception)
    reward = OutcomeReward(config.reward, config.environment.minimum_confidence)
    observer = observer or HumanInputObserver()
    period = 1.0 / config.environment.control_hz
    episode: list[Transition] = []
    episode_number = 0
    reason = "completed"
    recording_active = not start_paused
    recorded_elapsed = 0.0
    monitor.start()
    timer_resolution = WindowsTimerResolution()
    timer_resolution.start()

    def observe(probe: TimingProbe) -> Observation:
        frame = probe.call("capture_read", source.read)
        # time.monotonic() is quantized to the Windows scheduler tick on the
        # supported Python 3.10 runtime. QPC-backed perf_counter() is required
        # for 8 Hz trajectory timestamps and deadline accounting.
        return builder.build(frame, time.perf_counter(), probe)

    def save_current() -> None:
        nonlocal reason
        started = time.perf_counter()
        try:
            saved = save_episode(output, boss_id, episode, config.fingerprint())
        except BaseException as error:
            reason = f"save_error:{type(error).__name__}"
            monitor.emit("save", success=False, wall_ms=(time.perf_counter() - started) * 1000, error=type(error).__name__)
            raise
        monitor.emit("save", success=True, wall_ms=(time.perf_counter() - started) * 1000, transitions=len(episode), result=episode[-1].next_observation.episode_state.value, path=str(saved))
        print(f"saved demonstration episode: {saved} ({len(episode)} steps)", flush=True)

    try:
        startup = TimingProbe(monitor.enabled)
        startup.call("capture_start", source.start)
        startup.call("input_start", observer.start)
        current = observe(startup)
        monitor.emit("startup", **startup.payload())
        previous_state = None
        next_fighting_tick: float | None = None
        print(
            "[record] " + (
                "ARMED — press F8 to start. " if start_paused else "recording enabled. "
            ) + "F8 pauses/resumes; F9 saves and stops; Ctrl+C also saves.",
            flush=True,
        )
        while True:
            consume_controls = getattr(observer, "consume_control_requests", lambda: (False, False))
            toggle_requested, stop_requested = consume_controls()
            if stop_requested:
                reason = "hotkey_stop"
                break
            if toggle_requested:
                if recording_active:
                    recording_active = False
                    next_fighting_tick = None
                    if episode:
                        episode[-1].truncated = True
                        episode[-1].next_observation.episode_state = EpisodeState.TRUNCATED
                        save_current()
                        episode_number += 1
                        episode = []
                        builder.reset()
                    monitor.emit("recording_control", state="paused")
                    print("[record] PAUSED — press F8 to resume, F9 to stop", flush=True)
                else:
                    recording_active = True
                    next_fighting_tick = None
                    observer.discard_pending()
                    monitor.emit("recording_control", state="recording")
                    print("[record] RECORDING", flush=True)
            if duration_seconds is not None and recorded_elapsed >= duration_seconds:
                reason = "duration_limit"
                break
            probe = TimingProbe(monitor.enabled)
            if not recording_active:
                observer.discard_pending()
                probe.call("sleep", wait_until, time.perf_counter() + period)
                current = observe(probe)
                if monitor.enabled:
                    monitor.tick(probe, "paused", recorded=False,
                                 input=observer.diagnostics(),
                                 **observation_diagnostics(current, source, config.environment.minimum_confidence))
                continue
            if current.episode_state != previous_state:
                monitor.emit("state_change", previous=previous_state.value if previous_state else None, state=current.episode_state.value)
                print(f"[record] state={current.episode_state.value} buffered_steps={len(episode)}", flush=True)
                if current.episode_state is EpisodeState.FIGHTING:
                    discarded = observer.discard_pending()
                    monitor.emit("input_reset", discarded=discarded, reason="fight_started")
                previous_state = current.episode_state
            if current.episode_state is not EpisodeState.FIGHTING:
                next_fighting_tick = None
                observer.discard_pending()
                probe.call("sleep", time.sleep, min(period, 0.1))
                current = observe(probe)
                if monitor.enabled:
                    monitor.tick(probe, "waiting", recorded=False, input=observer.diagnostics(), **observation_diagnostics(current, source, config.environment.minimum_confidence))
                continue
            if next_fighting_tick is None:
                next_fighting_tick = time.perf_counter() + period
            probe.call("sleep", wait_until, next_fighting_tick)
            # The action label covers input observed during [current, next].
            # Sampling before this wait labels pulse actions one frame late.
            requested_action, raw_input = probe.call("input_sample", observer.sample)
            action = (
                requested_action
                if current.action_mask[int(requested_action)]
                else ActionToken.IDLE
            )
            next_observation = observe(probe)
            breakdown = probe.call("reward", reward.calculate,
                dict(current.measurements), dict(next_observation.measurements), next_observation.episode_state
            )
            next_observation.previous_action = action
            next_observation.previous_reward = breakdown.total
            terminated = next_observation.episode_state in {EpisodeState.WON, EpisodeState.LOST}
            truncated = next_observation.episode_state in {EpisodeState.TRUNCATED, EpisodeState.INVALID}
            transition = probe.call("transition_pack", Transition,
                observation=current,
                action=action,
                reward=breakdown.total,
                next_observation=next_observation,
                terminated=terminated,
                truncated=truncated,
                timestamp=next_observation.timestamp,
                episode_id=episode_number,
                step_id=len(episode),
                demonstration=True,
                raw_input=raw_input,
            )
            episode.append(transition)
            recorded_elapsed += max(0.0, next_observation.timestamp - current.timestamp)
            builder.previous_action = action
            builder.previous_reward = breakdown.total
            current = next_observation
            next_fighting_tick += period
            # Retain the absolute cadence so observation work is included in
            # the 125 ms budget. If the process stalls for a whole tick, skip
            # missed slots instead of emitting catch-up transitions.
            while next_fighting_tick <= time.perf_counter():
                next_fighting_tick += period
            if monitor.enabled:
                monitor.tick(
                    probe,
                    "fighting",
                    recorded=True,
                    action=action.name,
                    requested_action=requested_action.name,
                    action_masked=action != requested_action,
                    buffered_steps=len(episode),
                    input=observer.diagnostics(),
                    **observation_diagnostics(current, source, config.environment.minimum_confidence),
                )
            if transition.done:
                save_current()
                episode_number += 1
                episode = []
                builder.reset()
                current = observe(TimingProbe(monitor.enabled))
    except KeyboardInterrupt:
        reason = "keyboard_interrupt"
    except BaseException as error:
        reason = f"error:{type(error).__name__}"
        monitor.emit("error", error=type(error).__name__, message=str(error))
        raise
    finally:
        try:
            # Do not erase an already known end state when Ctrl+C interrupts a save.
            if episode and reason in {"keyboard_interrupt", "duration_limit", "hotkey_stop"}:
                if not episode[-1].done:
                    episode[-1].truncated = True
                    episode[-1].next_observation.episode_state = EpisodeState.TRUNCATED
                save_current()
            elif not episode and reason in {"keyboard_interrupt", "duration_limit", "hotkey_stop"}:
                print("[record] No unsaved transitions. Check the profile state/HUD if no episode was saved.", flush=True)
        finally:
            try:
                observer.stop()
            finally:
                try:
                    source.close()
                finally:
                    try:
                        monitor.close(reason)
                    finally:
                        timer_resolution.close()
