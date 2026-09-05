from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Protocol

import cv2
import numpy as np

from .actions import FixedRateActionController, build_action_mask
from .capture import FrameSource
from .config import PipelineConfig
from .perception import ScreenPerception, TerminalStateMachine
from .reward import OutcomeReward
from .scheduling import wait_until, WindowsTimerResolution
from .types import (
    ActionToken,
    EpisodeState,
    Observation,
    Transition,
    measurements_to_arrays,
)


class Environment(Protocol):
    def reset(self) -> Observation: ...
    def observe(self) -> Observation: ...
    def step(self, action: ActionToken) -> Transition: ...
    def close(self) -> None: ...


RestartHook = Callable[[], None]


@dataclass(slots=True)
class EnvironmentMetrics:
    observations: int = 0
    deadline_misses: int = 0
    invalid_observations: int = 0
    observation_latency_ms: float = 0.0

    @property
    def deadline_miss_rate(self) -> float:
        return self.deadline_misses / max(self.observations, 1)

    @property
    def invalid_observation_rate(self) -> float:
        return self.invalid_observations / max(self.observations, 1)


class WukongEnvironment:
    def __init__(
        self,
        config: PipelineConfig,
        source: FrameSource,
        perception: ScreenPerception,
        controller: FixedRateActionController,
        restart_hook: RestartHook | None = None,
        clock: Callable[[], float] = time.perf_counter,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self.source = source
        self.perception = perception
        self.controller = controller
        self.restart_hook = restart_hook
        self.clock = clock
        self.sleeper = sleeper
        self.terminal = TerminalStateMachine(config.environment)
        self.reward = OutcomeReward(config.reward, config.environment.minimum_confidence)
        self.metrics = EnvironmentMetrics()
        self._period = 1.0 / config.environment.control_hz
        self._system_scheduler = clock is time.perf_counter and sleeper is time.sleep
        self._timer_resolution = WindowsTimerResolution()
        self._previous_action = ActionToken.IDLE
        self._previous_reward = 0.0
        self._last_observation: Observation | None = None
        self._episode_id = 0
        self._step_id = 0
        self._started = False
        self._next_tick: float | None = None
        self.last_raw_frame: np.ndarray | None = None

    def _frame_to_observation(self, frame: np.ndarray, timestamp: float) -> Observation:
        measurements = self.perception.detect(frame)
        state = self.terminal.update(measurements, timestamp)
        set_episode_active = getattr(self.perception, "set_episode_active", None)
        if set_episode_active is not None:
            set_episode_active(state is EpisodeState.FIGHTING)
        features, confidence = measurements_to_arrays(measurements)
        action_mask = build_action_mask(measurements, self.config.environment.minimum_confidence)
        rgb = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(
            rgb,
            (self.config.capture.observation_width, self.config.capture.observation_height),
            interpolation=cv2.INTER_AREA,
        )
        if state is EpisodeState.INVALID:
            self.metrics.invalid_observations += 1
        return Observation(
            frame=np.ascontiguousarray(rgb, dtype=np.uint8),
            features=features,
            feature_confidence=confidence,
            action_mask=action_mask,
            timestamp=timestamp,
            episode_state=state,
            previous_action=self._previous_action,
            previous_reward=self._previous_reward,
            measurements=measurements,
        )

    def observe(self) -> Observation:
        start = self.clock()
        frame = self.source.read()
        self.last_raw_frame = frame
        timestamp = self.clock()
        observation = self._frame_to_observation(frame, timestamp)
        self.metrics.observations += 1
        self.metrics.observation_latency_ms = (self.clock() - start) * 1000.0
        return observation

    def reset(self) -> Observation:
        if self._system_scheduler:
            self._timer_resolution.start()
        if not self._started:
            self.source.start()
            self._started = True
        elif self.restart_hook is not None:
            self.controller.reset()
            self.restart_hook()
        self.perception.reset()
        self.terminal.reset(self.clock())
        self._previous_action = ActionToken.IDLE
        self._previous_reward = 0.0
        self._step_id = 0
        self._episode_id += 1
        deadline = self.clock() + self.config.environment.ready_timeout_seconds
        observation = self.observe()
        while observation.episode_state is not EpisodeState.FIGHTING:
            if self.clock() >= deadline:
                raise TimeoutError("game did not enter a reliable fighting state")
            self.sleeper(min(self._period, 0.1))
            observation = self.observe()
        self._last_observation = observation
        self._next_tick = self.clock() + self._period
        return observation

    def step(self, action: ActionToken) -> Transition:
        if self._last_observation is None:
            raise RuntimeError("reset must be called before step")
        action = ActionToken(action)
        if not self._last_observation.action_mask[int(action)]:
            action = ActionToken.IDLE
        tick_started = self.clock()
        if self._next_tick is None:
            self._next_tick = tick_started + self._period
        if tick_started > self._next_tick:
            self.metrics.deadline_misses += 1
        self.controller.apply(action)
        remaining = self._next_tick - self.clock()
        if remaining > 0:
            if self._system_scheduler:
                wait_until(self.clock() + remaining, clock=self.clock, sleep=self.sleeper)
            else:
                self.sleeper(remaining)
        next_observation = self.observe()
        self._next_tick += self._period
        while self._next_tick <= self.clock():
            self._next_tick += self._period
        breakdown = self.reward.calculate(
            dict(self._last_observation.measurements),
            dict(next_observation.measurements),
            next_observation.episode_state,
        )
        terminated = next_observation.episode_state in {EpisodeState.WON, EpisodeState.LOST}
        truncated = next_observation.episode_state in {EpisodeState.TRUNCATED, EpisodeState.INVALID}
        transition = Transition(
            observation=self._last_observation,
            action=action,
            reward=breakdown.total,
            next_observation=next_observation,
            terminated=terminated,
            truncated=truncated,
            timestamp=next_observation.timestamp,
            episode_id=self._episode_id,
            step_id=self._step_id,
        )
        self._step_id += 1
        self._previous_action = action
        self._previous_reward = breakdown.total
        next_observation.previous_action = action
        next_observation.previous_reward = breakdown.total
        self._last_observation = next_observation
        return transition

    def close(self) -> None:
        try:
            self.controller.close()
        finally:
            try:
                if self.restart_hook is not None and hasattr(self.restart_hook, "close"):
                    self.restart_hook.close()
            finally:
                try:
                    self.source.close()
                finally:
                    self._timer_resolution.close()
                    self._started = False
                    self._next_tick = None
                    self.last_raw_frame = None


class LegacyRestartHook:
    def __init__(self, action_name: str, config_path: str = "config/actions_conf.yaml") -> None:
        from actions import ActionExecutor

        self.executor = ActionExecutor(config_path)
        self.action_name = action_name

    def __call__(self) -> None:
        self.executor.take_action(self.action_name)
        self.executor.wait_for_finish()

    def close(self) -> None:
        self.executor.stop()
