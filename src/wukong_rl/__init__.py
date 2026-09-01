"""Wukong RL training package.

The public surface intentionally stays small.  Runtime integrations live behind
protocols so recorded traces and fake backends can exercise the whole pipeline
without starting the game.
"""

from importlib import import_module

# Spawned monitoring workers import the package too. Keep optional image/model
# stacks out of startup while preserving the existing public import surface.
_PUBLIC_MODULES = {
    "ActionToken": ".types", "EpisodeResult": ".types", "FieldMeasurement": ".types",
    "Observation": ".types", "Transition": ".types", "InputBackend": ".actions",
    "FrameSource": ".capture", "Environment": ".environment",
    "Policy": ".interfaces", "StateDetector": ".interfaces", "TransitionSink": ".interfaces",
}


def __getattr__(name: str):
    if name not in _PUBLIC_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_PUBLIC_MODULES[name], __name__), name)
    globals()[name] = value
    return value

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
