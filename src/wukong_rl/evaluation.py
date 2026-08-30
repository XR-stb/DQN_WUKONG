from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from .agent import R2D3Agent
from .checkpoint import load_checkpoint
from .config import PipelineConfig
from .metrics import JsonlMetricWriter
from .runtime import build_live_environment
from .types import ActionToken, EpisodeResult, HUD_KEYS


def evaluate_live(
    config: PipelineConfig,
    checkpoint_path: str | Path,
    episodes: int = 20,
    exploration: float = 0.0,
) -> dict:
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    agent = R2D3Agent(len(HUD_KEYS), ActionToken.size(), config.model)
    load_checkpoint(
        agent,
        checkpoint_path,
        expected_config_hash=config.fingerprint(),
        load_optimizer=False,
    )
    environment = build_live_environment(config)
    rng = np.random.default_rng(config.training.random_seed + 100)
    run_directory = Path("artifacts/evaluations") / time.strftime("%Y%m%d-%H%M%S")
    run_directory.mkdir(parents=True, exist_ok=True)
    metric_writer = JsonlMetricWriter(run_directory, "evaluation")
    results: list[EpisodeResult] = []
    try:
        for episode_index in range(episodes):
            observation = environment.reset()
            state = agent.initial_state()
            reward_sum = 0.0
            started = time.monotonic()
            video_path = run_directory / f"episode-{episode_index + 1:03d}.mp4"
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                config.environment.control_hz,
                (config.capture.width, config.capture.height),
            )
            if not writer.isOpened():
                raise RuntimeError(f"failed to create evaluation video: {video_path}")
            transition = None
            try:
                while transition is None or not transition.done:
                    action, state, q_values = agent.act(observation, state, exploration, rng)
                    transition = environment.step(action)
                    reward_sum += transition.reward
                    if environment.last_raw_frame is None:
                        raise RuntimeError("evaluation environment did not retain the raw frame")
                    writer.write(environment.last_raw_frame)
                    metric_writer.write(
                        "step",
                        episode=episode_index + 1,
                        step=transition.step_id,
                        action=transition.action.name,
                        reward=transition.reward,
                        entropy=agent.action_entropy(q_values, observation.action_mask),
                    )
                    observation = transition.next_observation
            finally:
                writer.release()
            result = EpisodeResult(
                episode_id=episode_index + 1,
                state=observation.episode_state,
                reward=reward_sum,
                steps=transition.step_id + 1,
                duration=time.monotonic() - started,
                boss_health=environment.terminal.last_valid_boss,
                self_health=environment.terminal.last_valid_self,
                damage_dealt=100.0 - environment.terminal.last_valid_boss,
                damage_taken=100.0 - environment.terminal.last_valid_self,
            )
            results.append(result)
            metric_writer.write("episode", **asdict(result))
    finally:
        environment.close()
    wins = sum(result.state.value == "won" for result in results)
    summary = {
        "episodes": episodes,
        "wins": wins,
        "win_rate": wins / episodes,
        "mean_reward": float(np.mean([result.reward for result in results])),
        "median_boss_health": float(np.median([result.boss_health for result in results])),
        "checkpoint": str(checkpoint_path),
        "config_hash": config.fingerprint(),
        "passed": wins / episodes >= 0.5,
        "results": [
            {**asdict(result), "state": result.state.value}
            for result in results
        ],
    }
    (run_directory / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return summary
