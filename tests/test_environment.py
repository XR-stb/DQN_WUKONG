from __future__ import annotations

import numpy as np
import pytest

from wukong_rl.actions import FixedRateActionController, NullInputBackend
from wukong_rl.capture import ArrayFrameSource
from wukong_rl.config import CaptureConfig, EnvironmentConfig, PipelineConfig
from wukong_rl.environment import WukongEnvironment
from wukong_rl.types import ActionToken

from conftest import make_measurements


class FakePerception:
    def __init__(self, values):
        self.values = list(values)
        self.index = 0

    def reset(self):
        self.index = 0

    def detect(self, _frame):
        value = self.values[min(self.index, len(self.values) - 1)]
        self.index += 1
        return value


class FakeClock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value

    def sleep(self, duration):
        self.value += max(duration, 0.0)


def build_environment():
    config = PipelineConfig(
        capture=CaptureConfig(
            backend="array",
            width=8,
            height=8,
            observation_width=8,
            observation_height=8,
        ),
        environment=EnvironmentConfig(
            control_hz=8,
            terminal_confirm_frames=1,
            minimum_confidence=0.5,
        ),
    )
    frames = [np.zeros((8, 8, 3), np.uint8) for _ in range(3)]
    perception = FakePerception(
        [make_measurements(), make_measurements(boss_hp=90), make_measurements(boss_hp=90)]
    )
    clock = FakeClock()
    backend = NullInputBackend()
    environment = WukongEnvironment(
        config,
        ArrayFrameSource(frames),
        perception,
        FixedRateActionController(backend),
        clock=clock,
        sleeper=clock.sleep,
    )
    return environment, backend


def test_deterministic_environment_replay_and_fixed_tick() -> None:
    first, backend = build_environment()
    second, _ = build_environment()
    first.reset()
    second.reset()
    transition_a = first.step(ActionToken.LIGHT_ATTACK)
    transition_b = second.step(ActionToken.LIGHT_ATTACK)
    assert transition_a.reward == pytest.approx(0.999)
    assert transition_a.reward == transition_b.reward
    assert transition_a.timestamp == pytest.approx(0.125)
    assert first.last_raw_frame is not None
    first.close()
    second.close()
    assert not backend.keys and not backend.buttons
