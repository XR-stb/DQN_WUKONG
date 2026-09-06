from __future__ import annotations

import numpy as np
import pytest

from wukong_rl.actions import FixedRateActionController, NullInputBackend
from wukong_rl.capture import ArrayFrameSource
from wukong_rl.config import CaptureConfig, EnvironmentConfig, PipelineConfig
from wukong_rl.environment import LegacyRestartHook, WukongEnvironment
from wukong_rl.types import ActionCommand, ActionToken, CombatToken, MovementToken

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


def test_restart_waits_for_death_loading_before_sending_input() -> None:
    events: list[tuple[str, object]] = []

    class FakeExecutor:
        def take_action(self, action_name: str) -> None:
            events.append(("take_action", action_name))

        def wait_for_finish(self) -> None:
            events.append(("wait_for_finish", None))

        def stop(self) -> None:
            events.append(("stop", None))

    hook = LegacyRestartHook(
        "FUZHAN_STAND_RESTART",
        death_load_seconds=18.0,
        sleeper=lambda seconds: events.append(("sleep", seconds)),
        executor=FakeExecutor(),
    )

    hook()
    hook.close()

    assert events == [
        ("sleep", 18.0),
        ("take_action", "FUZHAN_STAND_RESTART"),
        ("wait_for_finish", None),
        ("stop", None),
    ]


def test_restart_retry_repeats_same_action_without_second_death_wait() -> None:
    events: list[tuple[str, object]] = []

    class FakeExecutor:
        def take_action(self, action_name: str) -> None:
            events.append(("take_action", action_name))

        def wait_for_finish(self) -> None:
            events.append(("wait_for_finish", None))

        def stop(self) -> None:
            events.append(("stop", None))

    hook = LegacyRestartHook(
        "FUZHAN_STAND_RESTART",
        death_load_seconds=18.0,
        sleeper=lambda seconds: events.append(("sleep", seconds)),
        executor=FakeExecutor(),
    )

    hook()
    hook.retry(1)
    hook.close()

    assert events == [
        ("sleep", 18.0),
        ("take_action", "FUZHAN_STAND_RESTART"),
        ("wait_for_finish", None),
        ("take_action", "FUZHAN_STAND_RESTART"),
        ("wait_for_finish", None),
        ("stop", None),
    ]


def test_environment_retries_restart_until_fighting(tmp_path) -> None:
    class RetryPerception:
        def __init__(self) -> None:
            self.phase = -1

        def reset(self) -> None:
            self.phase += 1

        def detect(self, _frame):
            if self.phase in {0, 2}:
                return make_measurements()
            return make_measurements(confidence=0.0)

    class RetryHook:
        retry_interval_seconds = 0.25

        def __init__(self) -> None:
            self.calls: list[object] = []

        def __call__(self) -> None:
            self.calls.append("restart")

        def retry(self, attempt: int) -> None:
            self.calls.append(("retry", attempt))

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
    config.training.metrics_directory = str(tmp_path)
    clock = FakeClock()
    hook = RetryHook()
    environment = WukongEnvironment(
        config,
        ArrayFrameSource([np.zeros((8, 8, 3), np.uint8)]),
        RetryPerception(),
        FixedRateActionController(NullInputBackend()),
        restart_hook=hook,
        clock=clock,
        sleeper=clock.sleep,
    )

    environment.reset()
    recovered = environment.reset()

    assert recovered.episode_state.value == "fighting"
    assert hook.calls == ["restart", ("retry", 1)]
    assert len(list((tmp_path / "restart_failures").glob("*.jpg"))) == 1
    environment.close()


def test_restart_timeout_logs_hud_details_and_saves_frame(tmp_path, capsys) -> None:
    class TimeoutPerception:
        def __init__(self) -> None:
            self.phase = -1

        def reset(self) -> None:
            self.phase += 1

        def detect(self, _frame):
            if self.phase == 0:
                return make_measurements()
            return make_measurements(confidence=0.0)

    class RestartHook:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self) -> None:
            self.calls += 1

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
            ready_timeout_seconds=0.25,
        ),
    )
    config.training.metrics_directory = str(tmp_path)
    clock = FakeClock()
    hook = RestartHook()
    environment = WukongEnvironment(
        config,
        ArrayFrameSource([np.zeros((8, 8, 3), np.uint8)]),
        TimeoutPerception(),
        FixedRateActionController(NullInputBackend()),
        restart_hook=hook,
        clock=clock,
        sleeper=clock.sleep,
    )

    environment.reset()
    with pytest.raises(TimeoutError, match="observations="):
        environment.reset()

    output = capsys.readouterr().out
    assert "[restart] 等待战斗 HUD" in output
    assert "self_blood=" in output
    assert "boss_blood=" in output
    assert hook.calls == 1
    assert len(list((tmp_path / "restart_failures").glob("*.jpg"))) == 1
    log_text = (tmp_path / "restart-events.jsonl").read_text(encoding="utf-8")
    assert "等待战斗 HUD" in log_text
    assert "停止自动操作" in log_text
    environment.close()


def test_idle_escape_intervention_runs_for_multiple_ticks() -> None:
    environment, _ = build_environment()
    environment.config.environment.maximum_idle_ticks = 1
    environment.config.environment.idle_escape_ticks = 3
    environment.reset()

    transitions = [environment.step(ActionCommand()) for _ in range(5)]

    assert transitions[0].action == ActionCommand()
    assert [item.action.movement for item in transitions[1:4]] == [
        MovementToken.FORWARD,
        MovementToken.FORWARD,
        MovementToken.FORWARD,
    ]
    assert transitions[4].action == ActionCommand()
    environment.close()


def test_pulse_cooldown_masks_repeated_dodge() -> None:
    environment, _ = build_environment()
    environment.reset()

    transitions = [
        environment.step(ActionCommand(combat=CombatToken.DODGE)) for _ in range(4)
    ]

    assert [item.action.combat for item in transitions] == [
        CombatToken.DODGE,
        CombatToken.NONE,
        CombatToken.NONE,
        CombatToken.DODGE,
    ]
    environment.close()


def test_transformation_press_is_latched_for_the_rest_of_unconfirmed_episode() -> None:
    environment, _ = build_environment()
    observation = environment.reset()
    skill4_index = MovementToken.size() + int(CombatToken.SKILL_4)
    assert observation.action_mask[skill4_index]

    first = environment.step(ActionCommand(combat=CombatToken.SKILL_4))
    repeated = environment.step(ActionCommand(combat=CombatToken.SKILL_4))

    assert first.action.combat is CombatToken.SKILL_4
    assert not first.next_observation.action_mask[skill4_index]
    assert repeated.action.combat is CombatToken.NONE
    environment.close()


def test_attack_probe_breaks_movement_only_policy() -> None:
    environment, _ = build_environment()
    environment.reset()

    transitions = [
        environment.step(ActionCommand(movement=MovementToken.FORWARD))
        for _ in range(9)
    ]

    assert transitions[-1].action.combat is CombatToken.LIGHT_ATTACK
    assert environment.last_policy_intervention == "attack_probe_light"
    environment.close()
