from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Protocol

import cv2
import numpy as np

from .actions import FixedRateActionController, build_action_mask
from .capture import FrameSource
from .config import PipelineConfig
from .interfaces import StateDetector
from .perception import TerminalStateMachine
from .reward import OutcomeReward
from .scheduling import wait_until, WindowsTimerResolution
from .types import (
    ActionCommand,
    ActionToken,
    CombatToken,
    EpisodeState,
    Observation,
    MovementToken,
    Transition,
    canonicalize_command,
    measurements_to_arrays,
)


class Environment(Protocol):
    def reset(self) -> Observation: ...
    def observe(self) -> Observation: ...
    def step(self, action: ActionCommand) -> Transition: ...
    def close(self) -> None: ...


RestartHook = Callable[[], None]


class RestartExecutor(Protocol):
    def take_action(self, action_name: str) -> None: ...
    def wait_for_finish(self) -> None: ...
    def stop(self) -> None: ...


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
    _MAXIMUM_ATTACKLESS_TICKS = 8
    _PULSE_COOLDOWN_TICKS = {
        CombatToken.LIGHT_ATTACK: 1,
        CombatToken.DODGE: 2,
        CombatToken.SKILL_1: 3,
        CombatToken.SKILL_2: 3,
        CombatToken.SKILL_3: 3,
        CombatToken.SKILL_4: 3,
        CombatToken.FABAO: 3,
        CombatToken.TISHEN: 3,
        CombatToken.DRINK_POTION: 7,
    }
    _ATTACK_ACTIONS = {
        CombatToken.LIGHT_ATTACK,
        CombatToken.HEAVY_HOLD,
        CombatToken.SKILL_1,
        CombatToken.SKILL_2,
        CombatToken.SKILL_3,
        CombatToken.SKILL_4,
        CombatToken.FABAO,
        CombatToken.TISHEN,
    }

    def __init__(
        self,
        config: PipelineConfig,
        source: FrameSource,
        perception: StateDetector,
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
        self.reward = OutcomeReward(
            config.reward,
            config.environment.minimum_confidence,
            config.environment.terminal_health_percent,
        )
        self.metrics = EnvironmentMetrics()
        self._period = 1.0 / config.environment.control_hz
        self._system_scheduler = clock is time.perf_counter and sleeper is time.sleep
        self._timer_resolution = WindowsTimerResolution()
        self._previous_action = ActionCommand()
        self._previous_reward = 0.0
        self._last_observation: Observation | None = None
        # Replay persists across process restarts, so small per-run IDs could
        # make two partial episodes look contiguous after a restart.
        self._episode_id = time.time_ns() // 1_000_000
        self._step_id = 0
        self._idle_streak = 0
        self._idle_escape_remaining = 0
        self._attackless_streak = 0
        self._combat_cooldowns = {
            action: 0 for action in self._PULSE_COOLDOWN_TICKS
        }
        self.last_policy_intervention: str | None = None
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
        for combat, remaining in self._combat_cooldowns.items():
            if remaining > 0:
                action_mask[MovementToken.size() + int(combat)] = False
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
        self._previous_action = ActionCommand()
        self._previous_reward = 0.0
        self._step_id = 0
        self._idle_streak = 0
        self._idle_escape_remaining = 0
        self._attackless_streak = 0
        for combat in self._combat_cooldowns:
            self._combat_cooldowns[combat] = 0
        self.last_policy_intervention = None
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

    def step(self, action: ActionCommand | ActionToken) -> Transition:
        if self._last_observation is None:
            raise RuntimeError("reset must be called before step")
        action = action if isinstance(action, ActionCommand) else ActionCommand.from_legacy(action)
        action = canonicalize_command(action, self._last_observation.action_mask)
        interventions: list[str] = []
        if action.is_idle and self._idle_escape_remaining > 0:
            action = ActionCommand(MovementToken.FORWARD, action.combat)
            interventions.append("idle_escape_forward")
            self._idle_escape_remaining -= 1
        elif action.is_idle and self._idle_streak >= self.config.environment.maximum_idle_ticks:
            action = ActionCommand(MovementToken.FORWARD, action.combat)
            interventions.append("idle_escape_forward")
            self._idle_escape_remaining = self.config.environment.idle_escape_ticks - 1
        if (
            action.combat not in self._ATTACK_ACTIONS
            and self._attackless_streak >= self._MAXIMUM_ATTACKLESS_TICKS
            and self._last_observation.action_mask[
                MovementToken.size() + int(CombatToken.LIGHT_ATTACK)
            ]
        ):
            action = ActionCommand(action.movement, CombatToken.LIGHT_ATTACK)
            interventions.append("attack_probe_light")
        self.last_policy_intervention = "+".join(interventions) or None
        if action.is_idle:
            self._idle_streak += 1
        else:
            self._idle_streak = 0
        if action.combat in self._ATTACK_ACTIONS:
            self._attackless_streak = 0
        else:
            self._attackless_streak += 1
        tick_started = self.clock()
        if self._next_tick is None:
            self._next_tick = tick_started + self._period
        if tick_started > self._next_tick:
            self.metrics.deadline_misses += 1
        self.controller.apply(action)
        for combat, remaining in self._combat_cooldowns.items():
            self._combat_cooldowns[combat] = max(0, remaining - 1)
        if action.combat in self._PULSE_COOLDOWN_TICKS:
            self._combat_cooldowns[action.combat] = self._PULSE_COOLDOWN_TICKS[
                action.combat
            ]
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
                    close_perception = getattr(self.perception, "close", None)
                    if close_perception is not None:
                        close_perception()
                finally:
                    try:
                        self.source.close()
                    finally:
                        self._timer_resolution.close()
                        self._started = False
                        self._next_tick = None
                        self.last_raw_frame = None


class LegacyRestartHook:
    def __init__(
        self,
        action_name: str,
        config_path: str = "config/actions_conf.yaml",
        *,
        death_load_seconds: float = 18.0,
        sleeper: Callable[[float], None] = time.sleep,
        executor: RestartExecutor | None = None,
    ) -> None:
        if death_load_seconds < 0:
            raise ValueError("death_load_seconds cannot be negative")
        if executor is None:
            from actions import ActionExecutor

            executor = ActionExecutor(config_path)
        self.executor = executor
        self.action_name = action_name
        self.death_load_seconds = death_load_seconds
        self.sleeper = sleeper

    def __call__(self) -> None:
        # Terminal loss is confirmed from HP before the death/loading sequence
        # can accept input. The legacy pipeline waited here; omitting that wait
        # caused E to be swallowed and reset() to time out waiting for a boss bar.
        if self.death_load_seconds:
            print(
                f"[restart] 等待死亡加载 {self.death_load_seconds:.1f} 秒...",
                flush=True,
            )
            self.sleeper(self.death_load_seconds)
        print(f"[restart] 执行复战动作: {self.action_name}", flush=True)
        self.executor.take_action(self.action_name)
        self.executor.wait_for_finish()
        print("[restart] 复战输入完成，等待可靠战斗画面...", flush=True)

    def close(self) -> None:
        self.executor.stop()
