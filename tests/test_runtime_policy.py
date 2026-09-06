from __future__ import annotations

import pytest

from wukong_rl.config import PipelineConfig
from wukong_rl.runtime import _relabel_demonstration_episode
from wukong_rl.types import COMBAT_MASK_SLICE, ActionCommand, CombatToken

from conftest import make_transition


def _set_self_health(transition, before: float, after: float) -> None:
    transition.observation.features[0] = before / 100.0
    transition.next_observation.features[0] = after / 100.0


def test_demo_relabel_uses_new_potion_mask_and_health_floor_reward() -> None:
    full_health = make_transition(1, 0)
    _set_self_health(full_health, 100.0, 50.0)
    full_health.action = ActionCommand(combat=CombatToken.DRINK_POTION)

    healed = make_transition(1, 1)
    _set_self_health(healed, 50.0, 100.0)
    healed.action = ActionCommand(combat=CombatToken.DRINK_POTION)

    repeated_damage = make_transition(1, 2, done=True)
    _set_self_health(repeated_damage, 100.0, 60.0)

    relabeled = list(
        _relabel_demonstration_episode(
            [full_health, healed, repeated_damage], PipelineConfig()
        )
    )
    potion = int(CombatToken.DRINK_POTION)

    assert not relabeled[0].observation.action_mask[COMBAT_MASK_SLICE][potion]
    assert relabeled[0].action.combat is CombatToken.NONE
    assert relabeled[0].reward == pytest.approx(-2.0)
    assert relabeled[1].observation.action_mask[COMBAT_MASK_SLICE][potion]
    assert relabeled[1].action.combat is CombatToken.DRINK_POTION
    assert relabeled[1].reward == pytest.approx(-0.001)
    assert relabeled[2].reward == pytest.approx(-10.001)
    assert relabeled[2].observation.previous_reward == pytest.approx(-0.001)
