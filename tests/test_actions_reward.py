from __future__ import annotations

import pytest

from wukong_rl.actions import FixedRateActionController, NullInputBackend, build_action_mask
from wukong_rl.config import RewardConfig
from wukong_rl.reward import OutcomeReward
from wukong_rl.types import ActionToken, EpisodeState, FieldMeasurement

from conftest import make_measurements


def test_controller_holds_for_one_tick_and_releases_everything() -> None:
    backend = NullInputBackend()
    controller = FixedRateActionController(backend)
    controller.apply(ActionToken.LIGHT_ATTACK)
    assert "left" in backend.buttons
    controller.apply(ActionToken.RUN_FORWARD)
    assert "left" not in backend.buttons
    assert {"shift", "w"}.issubset(backend.keys)
    controller.apply(ActionToken.HEAVY_HOLD)
    assert not {"shift", "w"}.intersection(backend.keys)
    assert "right" in backend.buttons
    controller.pause()
    assert not backend.keys and not backend.buttons
    controller.apply(ActionToken.RUN_FORWARD)
    assert not backend.keys and not backend.buttons
    controller.resume()
    controller.apply(ActionToken.RUN_FORWARD)
    assert {"shift", "w"}.issubset(backend.keys)
    controller.close()
    assert not backend.keys
    assert not backend.buttons


def test_action_mask_uses_confident_resource_state() -> None:
    measurements = make_measurements()
    measurements["skill_1"] = FieldMeasurement(0.0, 1.0)
    measurements["skill_2"] = FieldMeasurement(0.0, 0.1, valid=False)
    measurements["hulu"] = FieldMeasurement(0.0, 1.0)
    mask = build_action_mask(measurements)
    assert not mask[int(ActionToken.SKILL_1)]
    assert not mask[int(ActionToken.SKILL_2)]
    assert not mask[int(ActionToken.DRINK_POTION)]
    assert mask[int(ActionToken.IDLE)]


def test_reward_is_outcome_only_and_clipped() -> None:
    reward = OutcomeReward(RewardConfig())
    previous = make_measurements(self_hp=100, boss_hp=100)
    current = make_measurements(self_hp=90, boss_hp=95)
    breakdown = reward.calculate(previous, current, EpisodeState.FIGHTING)
    assert breakdown.boss_damage == pytest.approx(0.5)
    assert breakdown.self_damage == pytest.approx(-1.2)
    assert breakdown.total == pytest.approx(-0.701)
    win = reward.calculate(current, current, EpisodeState.WON)
    assert win.total == pytest.approx(9.999)
