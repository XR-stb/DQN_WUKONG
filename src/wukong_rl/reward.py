from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import RewardConfig
from .types import EpisodeState, FieldMeasurement


@dataclass(slots=True)
class RewardBreakdown:
    total: float
    boss_damage: float
    self_damage: float
    tick: float
    terminal: float


class OutcomeReward:
    def __init__(self, config: RewardConfig, minimum_confidence: float = 0.55) -> None:
        self.config = config
        self.minimum_confidence = minimum_confidence

    def _reliable_delta(
        self,
        previous: dict[str, FieldMeasurement],
        current: dict[str, FieldMeasurement],
        field: str,
    ) -> float:
        before = previous.get(field)
        after = current.get(field)
        if not before or not after:
            return 0.0
        if not (before.valid and after.valid):
            return 0.0
        if min(before.confidence, after.confidence) < self.minimum_confidence:
            return 0.0
        return float(before.value - after.value)

    def calculate(
        self,
        previous: dict[str, FieldMeasurement],
        current: dict[str, FieldMeasurement],
        state: EpisodeState,
    ) -> RewardBreakdown:
        boss_drop = max(0.0, self._reliable_delta(previous, current, "boss_blood"))
        self_drop = max(0.0, self._reliable_delta(previous, current, "self_blood"))
        boss_reward = boss_drop * self.config.boss_damage_per_percent
        self_reward = self_drop * self.config.self_damage_per_percent
        terminal = 0.0
        if state is EpisodeState.WON:
            terminal = self.config.win_reward
        elif state is EpisodeState.LOST:
            terminal = self.config.loss_reward
        nonterminal = float(
            np.clip(
                boss_reward + self_reward + self.config.tick_penalty,
                -self.config.nonterminal_clip,
                self.config.nonterminal_clip,
            )
        )
        return RewardBreakdown(
            total=nonterminal + terminal,
            boss_damage=boss_reward,
            self_damage=self_reward,
            tick=self.config.tick_penalty,
            terminal=terminal,
        )
