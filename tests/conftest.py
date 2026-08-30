from __future__ import annotations

import numpy as np

from wukong_rl.types import (
    ActionToken,
    EpisodeState,
    HUD_KEYS,
    FieldMeasurement,
    Observation,
    Transition,
)


def make_measurements(self_hp: float = 100.0, boss_hp: float = 100.0, confidence: float = 1.0):
    measurements = {
        key: FieldMeasurement(1.0, confidence)
        for key in HUD_KEYS
        if key not in {"self_blood", "boss_blood", "self_energy", "self_magic", "hulu"}
    }
    measurements.update(
        {
            "self_blood": FieldMeasurement(self_hp, confidence),
            "boss_blood": FieldMeasurement(boss_hp, confidence),
            "self_energy": FieldMeasurement(100.0, confidence),
            "self_magic": FieldMeasurement(100.0, confidence),
            "hulu": FieldMeasurement(100.0, confidence),
        }
    )
    return measurements


def make_observation(
    value: int = 0,
    self_hp: float = 100.0,
    boss_hp: float = 100.0,
    state: EpisodeState = EpisodeState.FIGHTING,
    frame_shape: tuple[int, int, int] = (24, 32, 3),
) -> Observation:
    measurements = make_measurements(self_hp, boss_hp)
    features = np.ones(len(HUD_KEYS), dtype=np.float32)
    return Observation(
        frame=np.full(frame_shape, value, dtype=np.uint8),
        features=features,
        feature_confidence=np.ones_like(features),
        action_mask=np.ones(ActionToken.size(), dtype=np.bool_),
        timestamp=float(value),
        episode_state=state,
        measurements=measurements,
    )


def make_transition(
    episode_id: int,
    step_id: int,
    done: bool = False,
    frame_shape: tuple[int, int, int] = (24, 32, 3),
) -> Transition:
    current = make_observation(step_id % 255, frame_shape=frame_shape)
    state = EpisodeState.LOST if done else EpisodeState.FIGHTING
    following = make_observation((step_id + 1) % 255, state=state, frame_shape=frame_shape)
    return Transition(
        current,
        ActionToken.LIGHT_ATTACK,
        1.0,
        following,
        done,
        False,
        float(step_id + 1),
        episode_id,
        step_id,
    )
