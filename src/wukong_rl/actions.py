from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .types import ActionToken, FieldMeasurement


class InputBackend(Protocol):
    def press_key(self, key: str) -> None: ...
    def release_key(self, key: str) -> None: ...
    def press_mouse(self, button: str) -> None: ...
    def release_mouse(self, button: str) -> None: ...
    def release_all(self) -> None: ...


class NullInputBackend:
    """Deterministic backend for tests and dry-runs."""

    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []
        self.keys: set[str] = set()
        self.buttons: set[str] = set()

    def press_key(self, key: str) -> None:
        self.keys.add(key)
        self.events.append(("press_key", key))

    def release_key(self, key: str) -> None:
        self.keys.discard(key)
        self.events.append(("release_key", key))

    def press_mouse(self, button: str) -> None:
        self.buttons.add(button)
        self.events.append(("press_mouse", button))

    def release_mouse(self, button: str) -> None:
        self.buttons.discard(button)
        self.events.append(("release_mouse", button))

    def release_all(self) -> None:
        for key in tuple(self.keys):
            self.release_key(key)
        for button in tuple(self.buttons):
            self.release_mouse(button)


class PynputInputBackend:
    def __init__(self) -> None:
        from pynput.keyboard import Controller as KeyboardController, Key
        from pynput.mouse import Button, Controller as MouseController

        self._keyboard = KeyboardController()
        self._mouse = MouseController()
        self._key_enum = Key
        self._button_enum = Button
        self._keys: set[str] = set()
        self._buttons: set[str] = set()

    def _key(self, key: str):
        return getattr(self._key_enum, key, key)

    def _button(self, button: str):
        return getattr(self._button_enum, button)

    def press_key(self, key: str) -> None:
        if key not in self._keys:
            self._keyboard.press(self._key(key))
            self._keys.add(key)

    def release_key(self, key: str) -> None:
        if key in self._keys:
            self._keyboard.release(self._key(key))
            self._keys.discard(key)

    def press_mouse(self, button: str) -> None:
        if button not in self._buttons:
            self._mouse.press(self._button(button))
            self._buttons.add(button)

    def release_mouse(self, button: str) -> None:
        if button in self._buttons:
            self._mouse.release(self._button(button))
            self._buttons.discard(button)

    def release_all(self) -> None:
        for key in tuple(self._keys):
            self.release_key(key)
        for button in tuple(self._buttons):
            self.release_mouse(button)


@dataclass(slots=True)
class FixedRateActionController:
    backend: InputBackend
    pulse_seconds: float = 0.04
    _lock: threading.RLock = field(init=False, repr=False)
    _held_movement: str | None = field(init=False, default=None, repr=False)
    _heavy_held: bool = field(init=False, default=False, repr=False)
    _closed: bool = field(init=False, default=False, repr=False)
    _paused: bool = field(init=False, default=False, repr=False)
    _pulse_key: str | None = field(init=False, default=None, repr=False)
    _pulse_button: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        self._lock = threading.RLock()
        self._held_movement = None
        self._heavy_held = False
        self._closed = False
        self._paused = False

    _MOVEMENT = {
        ActionToken.RUN_FORWARD: "w",
        ActionToken.RUN_BACK: "s",
        ActionToken.RUN_LEFT: "a",
        ActionToken.RUN_RIGHT: "d",
    }
    _KEY_PULSES = {
        ActionToken.DODGE: "space",
        ActionToken.SKILL_1: "1",
        ActionToken.SKILL_2: "2",
        ActionToken.SKILL_3: "3",
        ActionToken.SKILL_4: "4",
        ActionToken.FABAO: "t",
        ActionToken.TISHEN: "f",
        ActionToken.DRINK_POTION: "r",
    }

    def _release_stateful(self, next_action: ActionToken) -> None:
        if self._pulse_key is not None:
            self.backend.release_key(self._pulse_key)
            self._pulse_key = None
        if self._pulse_button is not None:
            self.backend.release_mouse(self._pulse_button)
            self._pulse_button = None
        next_movement = self._MOVEMENT.get(next_action)
        if self._held_movement and self._held_movement != next_movement:
            self.backend.release_key(self._held_movement)
            self.backend.release_key("shift")
            self._held_movement = None
        if self._heavy_held and next_action is not ActionToken.HEAVY_HOLD:
            self.backend.release_mouse("right")
            self._heavy_held = False

    def apply(self, action: ActionToken) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("action controller is closed")
            if self._paused:
                self.backend.release_all()
                return
            action = ActionToken(action)
            self._release_stateful(action)
            movement = self._MOVEMENT.get(action)
            if movement:
                self.backend.press_key("shift")
                self.backend.press_key(movement)
                self._held_movement = movement
            elif action is ActionToken.HEAVY_HOLD:
                self.backend.press_mouse("right")
                self._heavy_held = True
            elif action is ActionToken.LIGHT_ATTACK:
                self.backend.press_mouse("left")
                self._pulse_button = "left"
            elif action in self._KEY_PULSES:
                key = self._KEY_PULSES[action]
                self.backend.press_key(key)
                self._pulse_key = key

    def reset(self) -> None:
        with self._lock:
            self.backend.release_all()
            self._held_movement = None
            self._heavy_held = False
            self._pulse_key = None
            self._pulse_button = None

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self.reset()
                self._closed = True

    def pause(self) -> None:
        with self._lock:
            if not self._closed:
                self.reset()
                self._paused = True

    def resume(self) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("action controller is closed")
            self._paused = False


def build_action_mask(
    measurements: dict[str, FieldMeasurement], minimum_confidence: float = 0.55
) -> np.ndarray:
    mask = np.ones(ActionToken.size(), dtype=np.bool_)

    def confidently_ready(field: str, threshold: float = 0.5) -> bool:
        value = measurements.get(field)
        return bool(value and value.valid and value.confidence >= minimum_confidence and value.value > threshold)

    for action, field in {
        ActionToken.SKILL_1: "skill_1",
        ActionToken.SKILL_2: "skill_2",
        ActionToken.SKILL_3: "skill_3",
        ActionToken.SKILL_4: "skill_4",
        ActionToken.FABAO: "skill_fb",
        ActionToken.TISHEN: "skill_ts",
    }.items():
        mask[int(action)] = confidently_ready(field)
    mask[int(ActionToken.DRINK_POTION)] = confidently_ready("hulu", threshold=2.0)

    energy = measurements.get("self_energy")
    if energy and energy.valid and energy.confidence >= minimum_confidence and energy.value <= 2.0:
        mask[int(ActionToken.DODGE)] = False
        mask[int(ActionToken.HEAVY_HOLD)] = False
    mask[int(ActionToken.IDLE)] = True
    return mask
