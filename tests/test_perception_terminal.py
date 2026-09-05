from __future__ import annotations

from pathlib import Path

import cv2

from wukong_rl.config import EnvironmentConfig, load_config
from wukong_rl.perception import RobustScalarFilter, ScreenPerception, TerminalStateMachine
from wukong_rl.types import EpisodeState

from conftest import make_measurements


def test_filter_requires_persistent_large_jump_and_blocks_boss_heal() -> None:
    signal = RobustScalarFilter(
        confirm_frames=3,
        minimum_confidence=0.5,
        maximum_jump=35.0,
        monotonic_decrease=True,
        increase_tolerance=1.5,
    )
    for _ in range(3):
        result = signal.update(98.0, 1.0)
    assert result.value == 98.0
    assert signal.update(1.0, 1.0).value == 98.0
    assert signal.update(1.0, 1.0).value == 98.0
    assert signal.update(1.0, 1.0).value == 1.0
    assert signal.update(90.0, 1.0).value == 1.0


def test_invalid_frames_break_large_jump_confirmation() -> None:
    signal = RobustScalarFilter(
        confirm_frames=3,
        minimum_confidence=0.5,
        maximum_jump=35.0,
        monotonic_decrease=True,
    )
    for _ in range(3):
        signal.update(98.0, 1.0)

    for _ in range(3):
        assert signal.update(1.0, 1.0).value == 98.0
        assert signal.update(0.0, 0.0, valid=False).value == 98.0

    assert signal.update(1.0, 1.0).value == 98.0


def test_monotonic_boss_damage_cannot_exceed_one_hundred_percent() -> None:
    signal = RobustScalarFilter(
        confirm_frames=1,
        minimum_confidence=0.5,
        maximum_jump=35.0,
        monotonic_decrease=True,
        increase_tolerance=1.5,
    )
    values = [signal.update(value, 1.0).value for value in (100, 80, 95, 30, 1, 60)]
    damage = sum(max(previous - current, 0.0) for previous, current in zip(values, values[1:]))
    assert values == sorted(values, reverse=True)
    assert damage <= 100.0


def test_boss_filter_can_rebase_before_combat_then_blocks_healing() -> None:
    signal = RobustScalarFilter(
        confirm_frames=3,
        minimum_confidence=0.5,
        maximum_jump=35.0,
        monotonic_decrease=True,
        increase_tolerance=1.5,
    )
    signal.set_monotonic_locked(False)
    for _ in range(3):
        signal.update(1.0, 1.0)
    for _ in range(3):
        result = signal.update(90.0, 1.0)
    assert result.value == 90.0

    signal.set_monotonic_locked(True)
    for _ in range(3):
        result = signal.update(100.0, 1.0)
    assert result.value == 90.0


def test_real_screen_without_boss_bar_is_not_interpreted_as_zero_health() -> None:
    frame_path = Path(__file__).parents[1] / "images" / "screen.png"
    frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    assert frame is not None
    config = load_config()
    detector = ScreenPerception(config.perception, frame.shape[1], frame.shape[0])
    measurements = detector.detect(frame)
    assert not measurements["boss_blood"].valid
    assert measurements["boss_blood"].confidence < config.perception.minimum_confidence


def test_terminal_state_distinguishes_ready_win_loss_and_invalid() -> None:
    config = EnvironmentConfig(
        terminal_confirm_frames=2,
        terminal_health_percent=1.0,
        terminal_inference_health_percent=3.0,
        recognition_failure_seconds=0.5,
        minimum_confidence=0.5,
    )
    machine = TerminalStateMachine(config)
    assert machine.update(make_measurements(), 0.0) is EpisodeState.WAITING
    assert machine.update(make_measurements(), 0.1) is EpisodeState.FIGHTING
    assert machine.update(make_measurements(boss_hp=0.5), 0.2) is EpisodeState.FIGHTING
    assert machine.update(make_measurements(boss_hp=0.5), 0.3) is EpisodeState.WON

    machine.reset(0.0)
    machine.update(make_measurements(), 0.0)
    machine.update(make_measurements(), 0.1)
    machine.update(make_measurements(self_hp=0.5, boss_hp=50), 0.2)
    assert machine.update(make_measurements(self_hp=0.5, boss_hp=50), 0.3) is EpisodeState.LOST

    machine.reset(0.0)
    invalid = make_measurements(confidence=0.0)
    assert machine.update(invalid, 0.0) is EpisodeState.LOADING
    machine.update(make_measurements(), 0.1)
    machine.update(make_measurements(), 0.2)
    for index in range(8):
        state = machine.update(invalid, 0.3 + index * 0.1)
    assert state is EpisodeState.INVALID


def test_terminal_discards_pre_fight_low_boss_measurement() -> None:
    config = EnvironmentConfig(
        terminal_confirm_frames=2,
        recognition_failure_seconds=0.5,
        minimum_confidence=0.5,
    )
    machine = TerminalStateMachine(config)
    assert machine.update(make_measurements(boss_hp=2.0), 0.0) is EpisodeState.WAITING
    assert machine.update(make_measurements(boss_hp=90.0), 0.1) is EpisodeState.WAITING
    assert machine.update(make_measurements(boss_hp=90.0), 0.2) is EpisodeState.FIGHTING
    assert machine.last_valid_boss == 90.0

    invalid = make_measurements(confidence=0.0)
    assert machine.update(invalid, 0.3) is EpisodeState.FIGHTING
    assert machine.update(invalid, 0.9) is EpisodeState.INVALID


def test_terminal_recovers_short_hud_loss_and_infers_pixel_quantized_death() -> None:
    config = EnvironmentConfig(
        terminal_confirm_frames=2,
        terminal_health_percent=1.0,
        terminal_inference_health_percent=3.0,
        recognition_failure_seconds=3.0,
        minimum_confidence=0.5,
    )
    machine = TerminalStateMachine(config)
    machine.update(make_measurements(), 0.0)
    machine.update(make_measurements(), 0.1)
    invalid = make_measurements(confidence=0.0)
    assert machine.update(invalid, 0.2) is EpisodeState.FIGHTING
    assert machine.update(invalid, 2.9) is EpisodeState.FIGHTING
    assert machine.update(make_measurements(self_hp=50, boss_hp=50), 3.0) is EpisodeState.FIGHTING
    assert machine.update(make_measurements(self_hp=2.88, boss_hp=50), 3.1) is EpisodeState.FIGHTING
    assert machine.update(make_measurements(self_hp=2.88, boss_hp=50), 3.2) is EpisodeState.FIGHTING
    assert machine.update(invalid, 3.3) is EpisodeState.FIGHTING
    assert machine.update(invalid, 6.4) is EpisodeState.LOST
def test_perception_limits_opencv_worker_pool():
    import cv2

    config = load_config()
    config.perception.opencv_threads = 1
    ScreenPerception(config.perception, config.capture.width, config.capture.height)
    assert cv2.getNumThreads() == 1
