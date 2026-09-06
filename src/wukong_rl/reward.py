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
    clipping: float = 0.0


class OutcomeReward:
    def __init__(
        self,
        config: RewardConfig,
        minimum_confidence: float = 0.55,
        terminal_health_percent: float = 1.0,
    ) -> None:
        self.config = config
        self.minimum_confidence = minimum_confidence
        self.terminal_health_percent = terminal_health_percent
        self.reset()

    def reset(self) -> None:
        self._lowest_boss_health: float | None = None
        self._lowest_self_health: float | None = None

    def _reliable_value(
        self, measurements: dict[str, FieldMeasurement], field: str
    ) -> float | None:
        measurement = measurements.get(field)
        if not measurement or not measurement.valid:
            return None
        if measurement.confidence < self.minimum_confidence:
            return None
        return float(measurement.value)

    def _new_health_floor(
        self,
        previous: dict[str, FieldMeasurement],
        current: dict[str, FieldMeasurement],
        field: str,
        attribute: str,
    ) -> float:
        before = self._reliable_value(previous, field)
        after = self._reliable_value(current, field)
        floor = getattr(self, attribute)
        if floor is None and before is not None:
            floor = before
        if floor is None or after is None:
            setattr(self, attribute, floor)
            return 0.0
        drop = max(0.0, floor - after)
        setattr(self, attribute, min(floor, after))
        return drop

    def calculate(
        self,
        previous: dict[str, FieldMeasurement],
        current: dict[str, FieldMeasurement],
        state: EpisodeState,
    ) -> RewardBreakdown:
        current_self = current.get("self_blood")
        player_is_reliably_dead = bool(
            current_self
            and current_self.valid
            and current_self.confidence >= self.minimum_confidence
            and current_self.value <= self.terminal_health_percent
        )
        boss_drop = 0.0
        if not player_is_reliably_dead and state is not EpisodeState.LOST:
            boss_drop = self._new_health_floor(
                previous,
                current,
                "boss_blood",
                "_lowest_boss_health",
            )
        self_drop = self._new_health_floor(
            previous,
            current,
            "self_blood",
            "_lowest_self_health",
        )
        boss_reward = boss_drop * self.config.boss_damage_per_percent
        self_reward = self_drop * self.config.self_damage_per_percent
        terminal = 0.0
        if state is EpisodeState.WON:
            terminal = self.config.win_reward
        elif state is EpisodeState.LOST:
            terminal = self.config.loss_reward
        raw_nonterminal = boss_reward + self_reward + self.config.tick_penalty
        nonterminal = float(
            np.clip(
                raw_nonterminal,
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
            clipping=nonterminal - raw_nonterminal,
        )
