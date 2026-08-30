from __future__ import annotations

import json
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from .actions import build_action_mask
from .capture import LegacyScreenSource
from .config import PipelineConfig
from .data import save_episode
from .perception import ScreenPerception, TerminalStateMachine
from .reward import OutcomeReward
from .types import ActionToken, EpisodeState, Observation, Transition, measurements_to_arrays


class HumanInputObserver:
    """Observe native controls without injecting or suppressing user input."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._held_keys: set[str] = set()
        self._held_buttons: set[str] = set()
        self._latched: deque[ActionToken] = deque()
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
                if name not in self._held_keys and name in pulse_keys:
                    self._latched.append(pulse_keys[name])
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
                        self._latched.append(ActionToken.LIGHT_ATTACK)
                else:
                    self._held_buttons.discard(name)

        self._keyboard_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self._mouse_listener = mouse.Listener(on_click=on_click)
        self._keyboard_listener.start()
        self._mouse_listener.start()

    def sample(self) -> tuple[ActionToken, str]:
        with self._lock:
            if self._latched:
                action = self._latched.popleft()
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

    def build(self, frame: np.ndarray, timestamp: float) -> Observation:
        measurements = self.perception.detect(frame)
        state = self.terminal.update(measurements, timestamp)
        features, confidence = measurements_to_arrays(measurements)
        mask = build_action_mask(measurements, self.config.environment.minimum_confidence)
        rgb = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(
            rgb,
            (self.config.capture.observation_width, self.config.capture.observation_height),
            interpolation=cv2.INTER_AREA,
        )
        return Observation(
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


def record_demonstrations(config: PipelineConfig, boss_id: str, output: str | Path) -> None:
    source = LegacyScreenSource(config.capture)
    perception = ScreenPerception(
        config.perception, config.capture.width, config.capture.height
    )
    builder = PassiveObservationBuilder(config, perception)
    reward = OutcomeReward(config.reward, config.environment.minimum_confidence)
    observer = HumanInputObserver()
    period = 1.0 / config.environment.control_hz
    episode: list[Transition] = []
    episode_number = 0
    source.start()
    observer.start()
    try:
        current = builder.build(source.read(), time.monotonic())
        while True:
            tick = time.monotonic()
            if current.episode_state is not EpisodeState.FIGHTING:
                time.sleep(min(period, 0.1))
                current = builder.build(source.read(), time.monotonic())
                continue
            action, raw_input = observer.sample()
            remaining = period - (time.monotonic() - tick)
            if remaining > 0:
                time.sleep(remaining)
            next_observation = builder.build(source.read(), time.monotonic())
            breakdown = reward.calculate(
                dict(current.measurements), dict(next_observation.measurements), next_observation.episode_state
            )
            next_observation.previous_action = action
            next_observation.previous_reward = breakdown.total
            terminated = next_observation.episode_state in {EpisodeState.WON, EpisodeState.LOST}
            truncated = next_observation.episode_state in {EpisodeState.TRUNCATED, EpisodeState.INVALID}
            transition = Transition(
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
            builder.previous_action = action
            builder.previous_reward = breakdown.total
            current = next_observation
            if transition.done:
                saved = save_episode(output, boss_id, episode, config.fingerprint())
                print(f"saved demonstration episode: {saved}")
                episode_number += 1
                episode = []
                builder.reset()
                current = builder.build(source.read(), time.monotonic())
    except KeyboardInterrupt:
        if episode:
            episode[-1].truncated = True
            episode[-1].next_observation.episode_state = EpisodeState.TRUNCATED
            saved = save_episode(output, boss_id, episode, config.fingerprint())
            print(f"saved truncated demonstration episode: {saved}")
    finally:
        observer.stop()
        source.close()
