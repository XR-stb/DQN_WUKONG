"""Wukong RL training package.

The public surface intentionally stays small.  Runtime integrations live behind
protocols so recorded traces and fake backends can exercise the whole pipeline
without starting the game.
"""

from .types import ActionToken, EpisodeResult, FieldMeasurement, Observation, Transition
from .actions import InputBackend
from .capture import FrameSource
from .environment import Environment
from .interfaces import Policy, StateDetector, TransitionSink

__all__ = [
    "ActionToken",
    "EpisodeResult",
    "FieldMeasurement",
    "FrameSource",
    "InputBackend",
    "Observation",
    "Environment",
    "Policy",
    "StateDetector",
    "Transition",
    "TransitionSink",
]

__version__ = "0.2.0"
