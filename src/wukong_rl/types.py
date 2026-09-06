from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Mapping, Sequence

import numpy as np


HUD_KEYS = (
    "self_blood",
    "boss_blood",
    "self_energy",
    "self_magic",
    "hulu",
    "skill_1",
    "skill_2",
    "skill_3",
    "skill_4",
    "skill_ts",
    "skill_fb",
    "gunshi1",
    "gunshi2",
    "gunshi3",
)


class ActionToken(IntEnum):
    IDLE = 0
    RUN_FORWARD = 1
    RUN_BACK = 2
    RUN_LEFT = 3
    RUN_RIGHT = 4
    LIGHT_ATTACK = 5
    HEAVY_HOLD = 6
    DODGE = 7
    SKILL_1 = 8
    SKILL_2 = 9
    SKILL_3 = 10
    SKILL_4 = 11
    FABAO = 12
    TISHEN = 13
    DRINK_POTION = 14

    @classmethod
    def size(cls) -> int:
        return len(cls)


class MovementToken(IntEnum):
    NONE = 0
    FORWARD = 1
    BACK = 2
    LEFT = 3
    RIGHT = 4
    FORWARD_LEFT = 5
    FORWARD_RIGHT = 6
    BACK_LEFT = 7
    BACK_RIGHT = 8

    @classmethod
    def size(cls) -> int:
        return len(cls)


class CombatToken(IntEnum):
    NONE = 0
    LIGHT_ATTACK = 1
    HEAVY_HOLD = 2
    DODGE = 3
    SKILL_1 = 4
    SKILL_2 = 5
    SKILL_3 = 6
    SKILL_4 = 7
    FABAO = 8
    TISHEN = 9
    DRINK_POTION = 10

    @classmethod
    def size(cls) -> int:
        return len(cls)


ACTION_MASK_SIZE = MovementToken.size() + CombatToken.size()
MOVEMENT_MASK_SLICE = slice(0, MovementToken.size())
COMBAT_MASK_SLICE = slice(MovementToken.size(), ACTION_MASK_SIZE)


@dataclass(slots=True, frozen=True)
class ActionCommand:
    """One control tick with independent movement and combat branches."""

    movement: MovementToken = MovementToken.NONE
    combat: CombatToken = CombatToken.NONE

    def __post_init__(self) -> None:
        object.__setattr__(self, "movement", MovementToken(self.movement))
        object.__setattr__(self, "combat", CombatToken(self.combat))

    @property
    def name(self) -> str:
        if self.is_idle:
            return "IDLE"
        parts = []
        if self.movement is not MovementToken.NONE:
            parts.append(f"RUN_{self.movement.name}")
        if self.combat is not CombatToken.NONE:
            parts.append(self.combat.name)
        return "+".join(parts)

    @property
    def is_idle(self) -> bool:
        return self.movement is MovementToken.NONE and self.combat is CombatToken.NONE

    def as_array(self) -> np.ndarray:
        return np.asarray([int(self.movement), int(self.combat)], dtype=np.int16)

    @classmethod
    def from_array(cls, value: Sequence[int] | np.ndarray) -> "ActionCommand":
        if len(value) != 2:
            raise ValueError("an action command must contain movement and combat branches")
        return cls(MovementToken(int(value[0])), CombatToken(int(value[1])))

    @classmethod
    def from_legacy(cls, action: ActionToken | int) -> "ActionCommand":
        action = ActionToken(action)
        movement = {
            ActionToken.RUN_FORWARD: MovementToken.FORWARD,
            ActionToken.RUN_BACK: MovementToken.BACK,
            ActionToken.RUN_LEFT: MovementToken.LEFT,
            ActionToken.RUN_RIGHT: MovementToken.RIGHT,
        }.get(action, MovementToken.NONE)
        combat = {
            ActionToken.LIGHT_ATTACK: CombatToken.LIGHT_ATTACK,
            ActionToken.HEAVY_HOLD: CombatToken.HEAVY_HOLD,
            ActionToken.DODGE: CombatToken.DODGE,
            ActionToken.SKILL_1: CombatToken.SKILL_1,
            ActionToken.SKILL_2: CombatToken.SKILL_2,
            ActionToken.SKILL_3: CombatToken.SKILL_3,
            ActionToken.SKILL_4: CombatToken.SKILL_4,
            ActionToken.FABAO: CombatToken.FABAO,
            ActionToken.TISHEN: CombatToken.TISHEN,
            ActionToken.DRINK_POTION: CombatToken.DRINK_POTION,
        }.get(action, CombatToken.NONE)
        return cls(movement, combat)


IDLE_COMMAND = ActionCommand()


def split_action_mask(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    value = np.asarray(mask, dtype=np.bool_)
    if value.shape[-1] != ACTION_MASK_SIZE:
        raise ValueError("action mask has an unexpected branch dimension")
    return value[..., MOVEMENT_MASK_SLICE], value[..., COMBAT_MASK_SLICE]


def canonicalize_command(command: ActionCommand, mask: np.ndarray) -> ActionCommand:
    movement_mask, combat_mask = split_action_mask(mask)
    movement = command.movement if movement_mask[int(command.movement)] else MovementToken.NONE
    combat = command.combat if combat_mask[int(command.combat)] else CombatToken.NONE
    return ActionCommand(movement, combat)


class EpisodeState(str, Enum):
    WAITING = "waiting"
    FIGHTING = "fighting"
    WON = "won"
    LOST = "lost"
    LOADING = "loading"
    INVALID = "invalid"
    TRUNCATED = "truncated"


@dataclass(slots=True, frozen=True)
class FieldMeasurement:
    value: float
    confidence: float
    age: int = 0
    valid: bool = True

    def normalized(self, percent: bool) -> float:
        value = self.value / 100.0 if percent else self.value
        return float(np.clip(value, 0.0, 1.0))


@dataclass(slots=True)
class Observation:
    frame: np.ndarray
    features: np.ndarray
    feature_confidence: np.ndarray
    action_mask: np.ndarray
    timestamp: float
    episode_state: EpisodeState = EpisodeState.WAITING
    previous_action: ActionCommand = field(default_factory=ActionCommand)
    previous_reward: float = 0.0
    measurements: Mapping[str, FieldMeasurement] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.frame.dtype != np.uint8 or self.frame.ndim != 3 or self.frame.shape[-1] != 3:
            raise ValueError("Observation.frame must be HxWx3 uint8")
        self.features = np.asarray(self.features, dtype=np.float32)
        self.feature_confidence = np.asarray(self.feature_confidence, dtype=np.float32)
        self.action_mask = np.asarray(self.action_mask, dtype=np.bool_)
        if self.features.shape != self.feature_confidence.shape:
            raise ValueError("features and feature_confidence must have the same shape")
        if self.action_mask.shape != (ACTION_MASK_SIZE,):
            raise ValueError("action_mask has an unexpected action dimension")
        movement_mask, combat_mask = split_action_mask(self.action_mask)
        if not movement_mask.any() or not combat_mask.any():
            raise ValueError("each action branch must allow at least one action")
        if not isinstance(self.previous_action, ActionCommand):
            self.previous_action = ActionCommand.from_legacy(self.previous_action)


@dataclass(slots=True)
class Transition:
    observation: Observation
    action: ActionCommand
    reward: float
    next_observation: Observation
    terminated: bool
    truncated: bool
    timestamp: float
    episode_id: int = 0
    step_id: int = 0
    demonstration: bool = False
    raw_input: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.action, ActionCommand):
            self.action = ActionCommand.from_legacy(self.action)

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated

    @property
    def detection_confidence(self) -> np.ndarray:
        return self.next_observation.feature_confidence


@dataclass(slots=True, frozen=True)
class EpisodeResult:
    episode_id: int
    state: EpisodeState
    reward: float
    steps: int
    duration: float
    boss_health: float
    self_health: float
    damage_dealt: float
    damage_taken: float


def measurements_to_arrays(
    measurements: Mapping[str, FieldMeasurement],
) -> tuple[np.ndarray, np.ndarray]:
    values: list[float] = []
    confidences: list[float] = []
    percent_fields = {"self_blood", "boss_blood", "self_energy", "self_magic", "hulu"}
    for key in HUD_KEYS:
        measurement = measurements.get(key, FieldMeasurement(0.0, 0.0, valid=False))
        values.append(measurement.normalized(key in percent_fields))
        confidences.append(float(np.clip(measurement.confidence, 0.0, 1.0)))
    return np.asarray(values, dtype=np.float32), np.asarray(confidences, dtype=np.float32)
