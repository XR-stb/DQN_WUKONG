from __future__ import annotations

import pytest

from wukong_rl.actions import (
    KEY_PULSE_BINDINGS,
    FixedRateActionController,
    NullInputBackend,
    build_action_mask,
)
from wukong_rl.config import RewardConfig
from wukong_rl.reward import OutcomeReward
from wukong_rl.types import (
    COMBAT_MASK_SLICE,
    ActionCommand,
    CombatToken,
    EpisodeState,
    FieldMeasurement,
    MovementToken,
)

from conftest import make_measurements


def test_controller_holds_for_one_tick_and_releases_everything() -> None:
    backend = NullInputBackend()
    controller = FixedRateActionController(backend)
    controller.apply(ActionCommand(MovementToken.FORWARD, CombatToken.LIGHT_ATTACK))
    assert "left" in backend.buttons
    assert {"shift", "w"}.issubset(backend.keys)
    controller.apply(ActionCommand(MovementToken.FORWARD, CombatToken.NONE))
    assert "left" not in backend.buttons
    assert {"shift", "w"}.issubset(backend.keys)
    controller.apply(ActionCommand(MovementToken.FORWARD, CombatToken.HEAVY_HOLD))
    assert {"shift", "w"}.issubset(backend.keys)
    assert "right" in backend.buttons
    controller.pause()
    assert not backend.keys and not backend.buttons
    controller.apply(ActionCommand(MovementToken.FORWARD))
    assert not backend.keys and not backend.buttons
    controller.resume()
    controller.apply(ActionCommand(MovementToken.FORWARD))
    assert {"shift", "w"}.issubset(backend.keys)
    controller.close()
    assert not backend.keys
    assert not backend.buttons


def test_drink_potion_uses_the_shared_q_binding() -> None:
    from wukong_rl.recording import HumanInputObserver

    assert KEY_PULSE_BINDINGS[CombatToken.DRINK_POTION] == "q"
    assert HumanInputObserver.PULSE_KEYS["q"] is CombatToken.DRINK_POTION
    assert "r" not in HumanInputObserver.PULSE_KEYS
    backend = NullInputBackend()
    controller = FixedRateActionController(backend)
    controller.apply(ActionCommand(combat=CombatToken.DRINK_POTION))
    assert backend.events[-1] == ("press_key", "q")
    controller.apply(ActionCommand())
    assert backend.events[-1] == ("release_key", "q")


def test_action_mask_uses_confident_resource_state() -> None:
    measurements = make_measurements()
    measurements["skill_1"] = FieldMeasurement(0.0, 1.0)
    measurements["skill_2"] = FieldMeasurement(0.0, 0.1, valid=False)
    measurements["hulu"] = FieldMeasurement(0.0, 1.0)
    mask = build_action_mask(measurements)
    combat = mask[COMBAT_MASK_SLICE]
    assert not combat[int(CombatToken.SKILL_1)]
    assert not combat[int(CombatToken.SKILL_2)]
    assert not combat[int(CombatToken.DRINK_POTION)]
    assert combat[int(CombatToken.NONE)]


def test_potion_requires_reliable_low_health_and_available_gourd() -> None:
    measurements = make_measurements(self_hp=100.0)
    measurements["hulu"] = FieldMeasurement(100.0, 1.0)
    combat = build_action_mask(measurements)[COMBAT_MASK_SLICE]
    assert not combat[int(CombatToken.DRINK_POTION)]

    measurements["self_blood"] = FieldMeasurement(60.0, 1.0)
    combat = build_action_mask(measurements)[COMBAT_MASK_SLICE]
    assert combat[int(CombatToken.DRINK_POTION)]

    measurements["self_blood"] = FieldMeasurement(20.0, 0.1, valid=False)
    combat = build_action_mask(measurements)[COMBAT_MASK_SLICE]
    assert not combat[int(CombatToken.DRINK_POTION)]


def test_skill_four_is_masked_while_transformed_or_latched() -> None:
    measurements = make_measurements()
    measurements["skill_4"] = FieldMeasurement(1.0, 1.0)
    measurements["transformation_active"] = FieldMeasurement(1.0, 1.0)
    combat = build_action_mask(measurements)[COMBAT_MASK_SLICE]
    assert not combat[int(CombatToken.SKILL_4)]

    measurements["transformation_active"] = FieldMeasurement(0.0, 1.0)
    combat = build_action_mask(
        measurements, transformation_latched=True
    )[COMBAT_MASK_SLICE]
    assert not combat[int(CombatToken.SKILL_4)]


def test_reward_is_outcome_only_and_clipped() -> None:
    reward = OutcomeReward(RewardConfig())
    previous = make_measurements(self_hp=100, boss_hp=100)
    current = make_measurements(self_hp=90, boss_hp=95)
    breakdown = reward.calculate(previous, current, EpisodeState.FIGHTING)
    assert breakdown.boss_damage == pytest.approx(0.5)
    assert breakdown.self_damage == pytest.approx(-1.2)
    assert breakdown.clipping == 0.0
    assert breakdown.total == pytest.approx(-0.701)
    win = reward.calculate(current, current, EpisodeState.WON)
    assert win.total == pytest.approx(9.999)


def test_reward_ignores_boss_drop_after_reliable_player_death() -> None:
    reward = OutcomeReward(RewardConfig(), terminal_health_percent=1.0)
    previous = make_measurements(self_hp=10.0, boss_hp=99.6)
    current = make_measurements(self_hp=0.0, boss_hp=1.9)

    breakdown = reward.calculate(previous, current, EpisodeState.FIGHTING)

    assert breakdown.boss_damage == 0.0
    assert breakdown.self_damage == pytest.approx(-1.2)
    assert breakdown.total == pytest.approx(-1.201)


def test_reward_penalizes_only_new_episode_health_lows_after_healing() -> None:
    reward = OutcomeReward(RewardConfig())
    first = reward.calculate(
        make_measurements(self_hp=100.0),
        make_measurements(self_hp=50.0),
        EpisodeState.FIGHTING,
    )
    healed = reward.calculate(
        make_measurements(self_hp=50.0),
        make_measurements(self_hp=100.0),
        EpisodeState.FIGHTING,
    )
    repeated = reward.calculate(
        make_measurements(self_hp=100.0),
        make_measurements(self_hp=60.0),
        EpisodeState.FIGHTING,
    )
    new_low = reward.calculate(
        make_measurements(self_hp=60.0),
        make_measurements(self_hp=40.0),
        EpisodeState.FIGHTING,
    )

    assert first.self_damage == pytest.approx(-6.0)
    assert healed.self_damage == 0.0
    assert repeated.self_damage == 0.0
    assert new_low.self_damage == pytest.approx(-1.2)
