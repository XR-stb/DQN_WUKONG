from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .types import (
    ACTION_MASK_SIZE,
    COMBAT_MASK_SLICE,
    ActionCommand,
    ActionToken,
    CombatToken,
    FieldMeasurement,
    MovementToken,
)


KEY_PULSE_BINDINGS = {
    CombatToken.DODGE: "space",
    CombatToken.SKILL_1: "1",
    CombatToken.SKILL_2: "2",
    CombatToken.SKILL_3: "3",
    CombatToken.SKILL_4: "4",
    CombatToken.FABAO: "t",
    CombatToken.TISHEN: "f",
    CombatToken.DRINK_POTION: "q",
}


class InputBackend(Protocol):
    def press_key(self, key: str) -> None: ...
    def release_key(self, key: str) -> None: ...
    def press_mouse(self, button: str) -> None: ...
    def release_mouse(self, button: str) -> None: ...
    def release_all(self) -> None: ...


class WindowsWindowActivator:
    """Bring the configured game window forward before a control phase starts."""

    def __init__(self, window_title: str) -> None:
        self.window_title = window_title

    def __call__(self) -> bool:
        import time

        import win32api
        import win32con
        import win32gui

        hwnd = win32gui.FindWindow(None, self.window_title)
        if not hwnd:
            raise RuntimeError(f"cannot locate game window: {self.window_title!r}")
        if win32gui.GetForegroundWindow() == hwnd:
            return False
        if win32gui.IsIconic(hwnd):
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        try:
            win32gui.BringWindowToTop(hwnd)
            win32gui.SetForegroundWindow(hwnd)
        except Exception:
            pass
        if win32gui.GetForegroundWindow() != hwnd:
            # Windows may reject SetForegroundWindow without raising when this
            # process did not receive the most recent user input. A momentary
            # Alt pulse grants foreground eligibility without opening another UI.
            win32api.keybd_event(win32con.VK_MENU, 0, 0, 0)
            try:
                win32gui.BringWindowToTop(hwnd)
                win32gui.SetForegroundWindow(hwnd)
            finally:
                win32api.keybd_event(
                    win32con.VK_MENU, 0, win32con.KEYEVENTF_KEYUP, 0
                )
        time.sleep(0.05)
        if win32gui.GetForegroundWindow() != hwnd:
            raise RuntimeError(f"failed to activate game window: {self.window_title!r}")
        print(f"[input] 已激活游戏窗口: {self.window_title!r}", flush=True)
        return True


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
    _held_movement: set[str] = field(init=False, default_factory=set, repr=False)
    _heavy_held: bool = field(init=False, default=False, repr=False)
    _closed: bool = field(init=False, default=False, repr=False)
    _paused: bool = field(init=False, default=False, repr=False)
    _pulse_key: str | None = field(init=False, default=None, repr=False)
    _pulse_button: str | None = field(init=False, default=None, repr=False)
    _pulse_timer: threading.Timer | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        self._lock = threading.RLock()
        self._held_movement = set()
        self._heavy_held = False
        self._closed = False
        self._paused = False

    _MOVEMENT = {
        MovementToken.NONE: (),
        MovementToken.FORWARD: ("w",),
        MovementToken.BACK: ("s",),
        MovementToken.LEFT: ("a",),
        MovementToken.RIGHT: ("d",),
        MovementToken.FORWARD_LEFT: ("w", "a"),
        MovementToken.FORWARD_RIGHT: ("w", "d"),
        MovementToken.BACK_LEFT: ("s", "a"),
        MovementToken.BACK_RIGHT: ("s", "d"),
    }
    _KEY_PULSES = KEY_PULSE_BINDINGS

    def _release_pulse(self) -> None:
        if self._pulse_key is not None:
            self.backend.release_key(self._pulse_key)
            self._pulse_key = None
        if self._pulse_button is not None:
            self.backend.release_mouse(self._pulse_button)
            self._pulse_button = None
        if self._pulse_timer is not None:
            self._pulse_timer.cancel()
            self._pulse_timer = None

    def _finish_pulse(self) -> None:
        with self._lock:
            if not self._closed:
                self._release_pulse()

    def _schedule_pulse_release(self) -> None:
        timer = threading.Timer(self.pulse_seconds, self._finish_pulse)
        timer.daemon = True
        self._pulse_timer = timer
        timer.start()

    def _set_movement(self, movement: MovementToken) -> None:
        desired = set(self._MOVEMENT[movement])
        if desired == self._held_movement:
            return
        for key in self._held_movement - desired:
            self.backend.release_key(key)
        if desired and not self._held_movement:
            self.backend.press_key("shift")
        for key in desired - self._held_movement:
            self.backend.press_key(key)
        if self._held_movement and not desired:
            self.backend.release_key("shift")
        self._held_movement = desired

    def _set_combat(self, combat: CombatToken) -> None:
        self._release_pulse()
        if self._heavy_held and combat is not CombatToken.HEAVY_HOLD:
            self.backend.release_mouse("right")
            self._heavy_held = False
        if combat is CombatToken.HEAVY_HOLD:
            if not self._heavy_held:
                self.backend.press_mouse("right")
                self._heavy_held = True
        elif combat is CombatToken.LIGHT_ATTACK:
            self.backend.press_mouse("left")
            self._pulse_button = "left"
            self._schedule_pulse_release()
        elif combat in self._KEY_PULSES:
            key = self._KEY_PULSES[combat]
            self.backend.press_key(key)
            self._pulse_key = key
            self._schedule_pulse_release()

    def apply(self, action: ActionCommand | ActionToken) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("action controller is closed")
            if self._paused:
                self.backend.release_all()
                return
            command = action if isinstance(action, ActionCommand) else ActionCommand.from_legacy(action)
            self._set_movement(command.movement)
            self._set_combat(command.combat)

    def reset(self) -> None:
        with self._lock:
            self.backend.release_all()
            if self._pulse_timer is not None:
                self._pulse_timer.cancel()
                self._pulse_timer = None
            self._held_movement.clear()
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
    mask = np.ones(ACTION_MASK_SIZE, dtype=np.bool_)
    combat = mask[COMBAT_MASK_SLICE]

    def confidently_ready(field: str, threshold: float = 0.5) -> bool:
        value = measurements.get(field)
        return bool(value and value.valid and value.confidence >= minimum_confidence and value.value > threshold)

    for action, field in {
        CombatToken.SKILL_1: "skill_1",
        CombatToken.SKILL_2: "skill_2",
        CombatToken.SKILL_3: "skill_3",
        CombatToken.SKILL_4: "skill_4",
        CombatToken.FABAO: "skill_fb",
        CombatToken.TISHEN: "skill_ts",
    }.items():
        combat[int(action)] = confidently_ready(field)
    combat[int(CombatToken.DRINK_POTION)] = confidently_ready("hulu", threshold=2.0)

    energy = measurements.get("self_energy")
    if energy and energy.valid and energy.confidence >= minimum_confidence and energy.value <= 2.0:
        combat[int(CombatToken.DODGE)] = False
        combat[int(CombatToken.HEAVY_HOLD)] = False
    combat[int(CombatToken.NONE)] = True
    return mask
