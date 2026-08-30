from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Mapping

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
    previous_action: ActionToken = ActionToken.IDLE
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
        if self.action_mask.shape != (ActionToken.size(),):
            raise ValueError("action_mask has an unexpected action dimension")
        if not self.action_mask.any():
            raise ValueError("at least one action must be valid")


@dataclass(slots=True)
class Transition:
    observation: Observation
    action: ActionToken
    reward: float
    next_observation: Observation
    terminated: bool
    truncated: bool
    timestamp: float
    episode_id: int = 0
    step_id: int = 0
    demonstration: bool = False
    raw_input: str = ""

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
