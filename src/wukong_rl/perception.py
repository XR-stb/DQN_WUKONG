from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np

from .config import EnvironmentConfig, PerceptionConfig
from .types import EpisodeState, FieldMeasurement


PERCENT_FIELDS = {"self_blood", "boss_blood", "self_energy", "self_magic", "hulu"}


class RobustScalarFilter:
    """Confidence-aware temporal filter with persistent jump confirmation."""

    def __init__(
        self,
        *,
        confirm_frames: int,
        minimum_confidence: float,
        maximum_jump: float,
        monotonic_decrease: bool = False,
        increase_tolerance: float = 0.0,
    ) -> None:
        self.confirm_frames = max(1, confirm_frames)
        self.minimum_confidence = minimum_confidence
        self.maximum_jump = maximum_jump
        self.monotonic_decrease = monotonic_decrease
        self.increase_tolerance = increase_tolerance
        self._history: deque[float] = deque(maxlen=self.confirm_frames)
        self._pending: deque[float] = deque(maxlen=self.confirm_frames)
        self._stable: float | None = None
        self._age = 0

    def reset(self) -> None:
        self._history.clear()
        self._pending.clear()
        self._stable = None
        self._age = 0

    def update(self, value: float, confidence: float, valid: bool = True) -> FieldMeasurement:
        value = float(value)
        confidence = float(np.clip(confidence, 0.0, 1.0))
        if not valid or not np.isfinite(value) or confidence < self.minimum_confidence:
            self._age += 1
            fallback = 0.0 if self._stable is None else self._stable
            return FieldMeasurement(fallback, confidence, self._age, valid=False)

        value = float(np.clip(value, 0.0, 100.0))
        if self._stable is None:
            self._history.append(value)
            if len(self._history) >= self.confirm_frames:
                self._stable = float(np.median(self._history))
            fallback = value if self._stable is None else self._stable
            return FieldMeasurement(fallback, confidence, 0, valid=self._stable is not None)

        if self.monotonic_decrease and value > self._stable + self.increase_tolerance:
            self._age += 1
            return FieldMeasurement(self._stable, confidence * 0.5, self._age, valid=True)

        if abs(value - self._stable) > self.maximum_jump:
            self._pending.append(value)
            if len(self._pending) < self.confirm_frames or np.ptp(self._pending) > 4.0:
                self._age += 1
                return FieldMeasurement(self._stable, confidence * 0.5, self._age, valid=True)
            value = float(np.median(self._pending))
            self._pending.clear()
            self._history.clear()
        else:
            self._pending.clear()

        self._history.append(value)
        filtered = float(np.median(self._history))
        if self.monotonic_decrease:
            filtered = min(filtered, self._stable)
        self._stable = filtered
        self._age = 0
        return FieldMeasurement(filtered, confidence, 0, valid=True)


@dataclass(slots=True)
class RegionDetector:
    name: str
    coordinates: tuple[int, int, int, int]
    value_range: tuple[float, float]
    percentage: bool
    filter: RobustScalarFilter

    def detect(self, frame: np.ndarray) -> FieldMeasurement:
        x1, y1, x2, y2 = self.coordinates
        region = frame[y1:y2, x1:x2, :3]
        if region.size == 0 or region.shape[0] < 1 or region.shape[1] < 1:
            return self.filter.update(0.0, 0.0, valid=False)
        hls = cv2.cvtColor(region, cv2.COLOR_BGR2HLS)
        lightness = hls[:, :, 1]
        minimum, maximum = self.value_range
        if self.percentage:
            samples = (
                lightness[:, lightness.shape[1] // 2]
                if self.name == "hulu"
                else lightness[lightness.shape[0] // 2]
            )
            mask = (samples >= minimum) & (samples <= maximum)
            value = float(mask.mean() * 100.0)
            inside = samples[mask]
            outside = samples[~mask]
            separation = 0.0
            if inside.size and outside.size:
                separation = min(1.0, abs(float(inside.mean() - outside.mean())) / 50.0)
            # A disappeared HUD is not a reliable zero. Require at least a small
            # amount of in-range foreground before accepting a percentage value;
            # the temporal filter then decides whether a low value is persistent.
            minimum_support = max(1, int(np.ceil(samples.size * 0.005)))
            support = min(1.0, float(mask.sum()) / minimum_support)
            confidence = float(
                np.clip(
                    (0.55 + 0.35 * separation + 0.1 * min(region.shape[0] / 6.0, 1.0))
                    * support,
                    0,
                    1,
                )
            )
            valid = bool(mask.any())
        else:
            mean = float(lightness.mean())
            midpoint = (minimum + maximum) / 2.0
            half_width = max((maximum - minimum) / 2.0, 1.0)
            ready = minimum <= mean <= maximum
            distance = abs(mean - midpoint) / half_width
            confidence = float(np.clip(1.0 - min(distance, 1.0), 0.0, 1.0))
            if not ready:
                confidence = max(confidence, 0.65)
            value = 1.0 if ready else 0.0
            valid = True
        return self.filter.update(value, confidence, valid=valid)


class ScreenPerception:
    DEFAULT_RANGES = {
        "self_blood": (117.0, 255.0),
        "boss_blood": (117.0, 255.0),
        "self_magic": (80.0, 120.0),
        "self_energy": (135.0, 165.0),
        "hulu": (120.0, 255.0),
        "skill_1": (90.0, 160.0),
        "skill_2": (90.0, 160.0),
        "skill_3": (90.0, 160.0),
        "skill_4": (90.0, 160.0),
        "skill_ts": (145.0, 190.0),
        "skill_fb": (145.0, 190.0),
        "gunshi1": (210.0, 255.0),
        "gunshi2": (210.0, 255.0),
        "gunshi3": (210.0, 255.0),
    }

    def __init__(self, config: PerceptionConfig, frame_width: int, frame_height: int) -> None:
        self.config = config
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.detectors: dict[str, RegionDetector] = {}
        for name, coordinates in config.regions.items():
            if name not in self.DEFAULT_RANGES:
                continue
            x1, y1, x2, y2 = self._scale_coordinates(coordinates)
            value_range = tuple(config.ranges.get(name, self.DEFAULT_RANGES[name]))
            is_percent = name in PERCENT_FIELDS
            scalar_filter = RobustScalarFilter(
                confirm_frames=config.confirm_frames,
                minimum_confidence=config.minimum_confidence,
                maximum_jump=config.maximum_jump_percent if is_percent else 1.0,
                monotonic_decrease=name == "boss_blood",
                increase_tolerance=config.boss_increase_tolerance,
            )
            self.detectors[name] = RegionDetector(
                name, (x1, y1, x2, y2), value_range, is_percent, scalar_filter
            )

    def _scale_coordinates(self, coordinates: Iterable[int]) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = (int(value) for value in coordinates)
        sx = self.frame_width / self.config.base_width
        sy = self.frame_height / self.config.base_height
        return (
            int(round(x1 * sx)),
            int(round(y1 * sy)),
            int(round((x2 + 1) * sx)),
            int(round((y2 + 1) * sy)),
        )

    def reset(self) -> None:
        for detector in self.detectors.values():
            detector.filter.reset()

    def detect(self, frame: np.ndarray) -> dict[str, FieldMeasurement]:
        if frame.shape[:2] != (self.frame_height, self.frame_width):
            raise ValueError(
                f"perception expected {self.frame_width}x{self.frame_height}, "
                f"got {frame.shape[1]}x{frame.shape[0]}"
            )
        return {name: detector.detect(frame) for name, detector in self.detectors.items()}


class TerminalStateMachine:
    def __init__(self, config: EnvironmentConfig) -> None:
        self.config = config
        self.reset()

    def reset(self, timestamp: float | None = None) -> None:
        self.state = EpisodeState.WAITING
        self.started_at = time.monotonic() if timestamp is None else timestamp
        self._ready_count = 0
        self._boss_low_count = 0
        self._self_low_count = 0
        self._invalid_count = 0
        self.last_valid_boss = 100.0
        self.last_valid_self = 100.0

    def _valid(self, measurement: FieldMeasurement | None) -> bool:
        return bool(
            measurement
            and measurement.valid
            and measurement.confidence >= self.config.minimum_confidence
            and measurement.age <= self.config.terminal_confirm_frames
        )

    def update(
        self, measurements: dict[str, FieldMeasurement], timestamp: float | None = None
    ) -> EpisodeState:
        now = time.monotonic() if timestamp is None else timestamp
        self_hp = measurements.get("self_blood")
        boss_hp = measurements.get("boss_blood")
        valid_self = self._valid(self_hp)
        valid_boss = self._valid(boss_hp)
        if valid_self:
            self.last_valid_self = float(self_hp.value)
        if valid_boss:
            self.last_valid_boss = min(self.last_valid_boss, float(boss_hp.value))

        if self.state in {EpisodeState.WON, EpisodeState.LOST, EpisodeState.TRUNCATED, EpisodeState.INVALID}:
            return self.state

        if self.state is not EpisodeState.FIGHTING:
            if valid_self and valid_boss and self_hp.value > 5.0 and boss_hp.value > 5.0:
                self._ready_count += 1
                if self._ready_count >= self.config.terminal_confirm_frames:
                    self.state = EpisodeState.FIGHTING
                    self.started_at = now
            else:
                self._ready_count = 0
                self.state = EpisodeState.LOADING if not (valid_self or valid_boss) else EpisodeState.WAITING
            return self.state

        if now - self.started_at >= self.config.episode_timeout_seconds:
            self.state = EpisodeState.TRUNCATED
            return self.state

        if not (valid_self and valid_boss):
            self._invalid_count += 1
            if self._invalid_count >= self.config.terminal_confirm_frames * 4:
                self.state = EpisodeState.INVALID
            return self.state
        self._invalid_count = 0

        self._boss_low_count = self._boss_low_count + 1 if boss_hp.value <= 1.0 else 0
        self._self_low_count = self._self_low_count + 1 if self_hp.value <= 1.0 else 0
        if self._boss_low_count >= self.config.terminal_confirm_frames and self_hp.value > 1.0:
            self.state = EpisodeState.WON
        elif self._self_low_count >= self.config.terminal_confirm_frames and self.last_valid_boss > 1.0:
            self.state = EpisodeState.LOST
        return self.state
