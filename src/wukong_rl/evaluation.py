from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from .agent import R2D3Agent
from .checkpoint import load_checkpoint
from .config import PipelineConfig
from .metrics import JsonlMetricWriter
from .runtime import build_live_environment
from .types import ActionToken, EpisodeResult, EpisodeState, HUD_KEYS


def _build_summary(
    results: list[EpisodeResult],
    *,
    requested_episodes: int,
    checkpoint_path: str | Path,
    config_hash: str,
    error: str | None = None,
    interrupted: bool = False,
) -> dict:
    completed_episodes = len(results)
    wins = sum(result.state is EpisodeState.WON for result in results)
    win_rate = wins / completed_episodes if completed_episodes else 0.0
    complete = completed_episodes == requested_episodes and error is None and not interrupted
    return {
        "requested_episodes": requested_episodes,
        "episodes": completed_episodes,
        "wins": wins,
        "win_rate": win_rate,
        "mean_reward": (
            float(np.mean([result.reward for result in results])) if results else None
        ),
        "median_boss_health": (
            float(np.median([result.boss_health for result in results])) if results else None
        ),
        "checkpoint": str(checkpoint_path),
        "config_hash": config_hash,
        "complete": complete,
        "interrupted": interrupted,
        "error": error,
        "passed": complete and win_rate >= 0.5,
        "results": [
            {**asdict(result), "state": result.state.value}
            for result in results
        ],
    }


def _write_summary(run_directory: Path, summary: dict) -> None:
    destination = run_directory / "summary.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    temporary.replace(destination)


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
    run_error: str | None = None
    interrupted = False
    print(
        f"[eval] 冻结策略评测已启动（不会训练或更新权重），输出目录: {run_directory}",
        flush=True,
    )
    metric_writer.write(
        "run_started",
        requested_episodes=episodes,
        checkpoint=str(checkpoint_path),
        exploration=exploration,
    )
    try:
        for episode_index in range(episodes):
            if episode_index:
                print(
                    f"[eval] 第 {episode_index + 1}/{episodes} 局：触发自动复战并等待可靠战斗画面...",
                    flush=True,
                )
            else:
                print(
                    f"[eval] 第 1/{episodes} 局：等待可靠战斗画面...",
                    flush=True,
                )
            observation = environment.reset()
            print(f"[eval] 第 {episode_index + 1}/{episodes} 局已进入战斗。", flush=True)
            state = agent.initial_state()
            reward_sum = 0.0
            started = time.monotonic()
            next_progress = started + 5.0
            action_counts: Counter[str] = Counter()
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
                    action_counts[transition.action.name] += 1
                    if environment.last_raw_frame is None:
                        raise RuntimeError("evaluation environment did not retain the raw frame")
                    writer.write(environment.last_raw_frame)
                    metric_writer.write(
                        "step",
                        episode=episode_index + 1,
                        step=transition.step_id,
                        action=transition.action.name,
                        reward=transition.reward,
                        reward_sum=reward_sum,
                        boss_health=environment.terminal.last_valid_boss,
                        self_health=environment.terminal.last_valid_self,
                        entropy=agent.action_entropy(q_values, observation.action_mask),
                    )
                    observation = transition.next_observation
                    now = time.monotonic()
                    if now >= next_progress:
                        common_actions = ", ".join(
                            f"{name}:{count}" for name, count in action_counts.most_common(3)
                        )
                        print(
                            f"[eval] 局{episode_index + 1} step={transition.step_id + 1} "
                            f"reward={reward_sum:.3f} "
                            f"boss_hp={environment.terminal.last_valid_boss:.1f}% "
                            f"self_hp={environment.terminal.last_valid_self:.1f}% "
                            f"top_actions=[{common_actions}]",
                            flush=True,
                        )
                        next_progress = now + 5.0
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
            metric_writer.write("episode", **asdict(result), actions=dict(action_counts))
            print(
                f"[eval] 第 {episode_index + 1} 局结束: state={result.state.value} "
                f"reward={result.reward:.3f} steps={result.steps} "
                f"boss_hp={result.boss_health:.1f}% self_hp={result.self_health:.1f}%",
                flush=True,
            )
            # INVALID means perception could not prove either victory or death.
            # Pressing a restart key in that state can corrupt the next episode.
            if result.state is EpisodeState.INVALID and episode_index + 1 < episodes:
                run_error = (
                    f"episode {episode_index + 1} ended with invalid perception; "
                    "automatic restart was skipped"
                )
                print(f"[eval] 已停止后续局数: {run_error}", flush=True)
                break
    except KeyboardInterrupt:
        interrupted = True
        run_error = "evaluation interrupted by user"
        print("[eval] 收到中断，正在安全释放按键并保存部分结果...", flush=True)
    except Exception as exc:
        run_error = f"{type(exc).__name__}: {exc}"
        print(f"[eval] 评测提前停止: {run_error}", flush=True)
    finally:
        try:
            environment.close()
        except Exception as exc:
            close_error = f"environment close failed: {type(exc).__name__}: {exc}"
            run_error = f"{run_error}; {close_error}" if run_error else close_error

    summary = _build_summary(
        results,
        requested_episodes=episodes,
        checkpoint_path=checkpoint_path,
        config_hash=config.fingerprint(),
        error=run_error,
        interrupted=interrupted,
    )
    _write_summary(run_directory, summary)
    metric_writer.write("run_finished", **summary)
    print(
        f"[eval] 评测结束，完成 {len(results)}/{episodes} 局；汇总已保存: "
        f"{run_directory / 'summary.json'}",
        flush=True,
    )
    return summary
