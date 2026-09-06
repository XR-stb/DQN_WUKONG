from __future__ import annotations

from typing import Any, Mapping, Protocol

import numpy as np

from .types import ActionCommand, FieldMeasurement, Observation, Transition


class StateDetector(Protocol):
    def reset(self) -> None: ...

    def detect(self, frame: np.ndarray) -> Mapping[str, FieldMeasurement]: ...


class Policy(Protocol):
    """Online policy boundary shared by R2D3 and future sequence models."""

    def initial_state(self, batch_size: int = 1) -> Any: ...

    def act(
        self,
        observation: Observation,
        state: Any | None,
        epsilon: float,
        rng: np.random.Generator,
    ) -> tuple[ActionCommand, Any, np.ndarray]: ...


class TransitionSink(Protocol):
    def add(self, transition: Transition) -> int: ...

    def flush(self) -> None: ...
