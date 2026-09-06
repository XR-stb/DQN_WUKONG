from __future__ import annotations

import multiprocessing as mp
import queue
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .actions import FixedRateActionController, PynputInputBackend, build_action_mask
from .agent import R2D3Agent
from .capture import create_screen_source
from .checkpoint import cpu_state_dict, load_checkpoint, save_checkpoint
from .config import PipelineConfig, load_config, online_replay_directory, replay_semantics_key
from .data import TrajectoryDataset, transitions_from_episode
from .environment import LegacyRestartHook, WukongEnvironment
from .metrics import emit_metric, metric_worker
from .replay import DiskPrioritizedSequenceReplay, ReplayBatch, concatenate_batches
from .reward import OutcomeReward
from .telemetry import build_perception
from .types import (
    ACTION_MASK_SIZE,
    ActionCommand,
    CombatToken,
    EpisodeResult,
    EpisodeState,
    FieldMeasurement,
    HUD_KEYS,
    MovementToken,
    Observation,
    Transition,
    canonicalize_command,
)


_TRANSITION_STREAM_END = "wukong_rl.transition_stream_end"
_PERCENT_HUD_FIELDS = {"self_blood", "boss_blood", "self_energy", "self_magic", "hulu"}


def build_live_environment(config: PipelineConfig) -> WukongEnvironment:
    source = create_screen_source(config.capture)
    perception = build_perception(config)
    controller = FixedRateActionController(PynputInputBackend())
    restart = LegacyRestartHook(
        config.environment.restart_action,
        expected_window_title=config.capture.window_title,
        metrics_directory=config.training.metrics_directory,
    )
    return WukongEnvironment(
        config,
        source,
        perception,
        controller,
        restart_hook=restart,
    )


def _put_latest(weight_queue, payload) -> None:
    try:
        weight_queue.put_nowait(payload)
        return
    except queue.Full:
        pass
    try:
        weight_queue.get_nowait()
    except queue.Empty:
        pass
    try:
        weight_queue.put_nowait(payload)
    except queue.Full:
        pass


def _measurements_from_observation(
    observation: Observation,
) -> dict[str, FieldMeasurement]:
    measurements: dict[str, FieldMeasurement] = {}
    for index, key in enumerate(HUD_KEYS):
        value = float(observation.features[index])
        if key in _PERCENT_HUD_FIELDS:
            value *= 100.0
        measurements[key] = FieldMeasurement(
            value=value,
            confidence=float(observation.feature_confidence[index]),
            age=0,
            valid=True,
        )
    return measurements


def _relabel_demonstration_episode(transitions, config: PipelineConfig):
    """Apply current reward/mask semantics without altering source recordings."""

    reward = OutcomeReward(
        config.reward,
        config.environment.minimum_confidence,
        config.environment.terminal_health_percent,
    )
    previous_action = ActionCommand()
    previous_reward = 0.0
    for transition in transitions:
        current_measurements = _measurements_from_observation(transition.observation)
        next_measurements = _measurements_from_observation(transition.next_observation)
        transition.observation.measurements = current_measurements
        transition.next_observation.measurements = next_measurements
        transition.observation.action_mask = build_action_mask(
            current_measurements,
            config.environment.minimum_confidence,
            potion_health_threshold_percent=(
                config.environment.potion_health_threshold_percent
            ),
        )
        transition.next_observation.action_mask = build_action_mask(
            next_measurements,
            config.environment.minimum_confidence,
            potion_health_threshold_percent=(
                config.environment.potion_health_threshold_percent
            ),
        )
        transition.action = canonicalize_command(
            transition.action, transition.observation.action_mask
        )
        transition.observation.previous_action = previous_action
        transition.observation.previous_reward = previous_reward
        breakdown = reward.calculate(
            current_measurements,
            next_measurements,
            transition.next_observation.episode_state,
        )
        transition.reward = breakdown.total
        transition.next_observation.previous_action = transition.action
        transition.next_observation.previous_reward = breakdown.total
        previous_action = transition.action
        previous_reward = breakdown.total
        yield transition


def _bootstrap_demo_replay(
    config: PipelineConfig,
    dataset_path: str | None,
    sequence_length: int,
    frame_shape: tuple[int, int, int],
) -> DiskPrioritizedSequenceReplay | None:
    if not dataset_path:
        return None
    dataset = TrajectoryDataset(dataset_path, boss_id=config.environment.boss_id)
    total = dataset.total_transitions
    capacity = max(total + sequence_length + 1, sequence_length * 4)
    replay = DiskPrioritizedSequenceReplay(
        # v4 re-labels reward floors and health-aware action masks. Keep it
        # separate so older cached rewards/masks cannot leak into DQfD updates.
        Path(config.replay.directory)
        / f"demonstrations-v4-{replay_semantics_key(config)}-{dataset.version}",
        capacity,
        frame_shape,
        len(HUD_KEYS),
        ACTION_MASK_SIZE,
        sequence_length,
        config.model.burn_in,
        config.replay.priority_alpha,
        demonstration=True,
    )
    if replay.sequence_count == 0:
        print(
            f"[learner] 正在以当前奖励/动作掩码重建示范回放 v4: "
            f"episodes={len(dataset.manifest_paths)} transitions={total}",
            flush=True,
        )
        loaded_transitions = 0
        for episode_id, episode in enumerate(dataset.episodes()):
            try:
                transitions = transitions_from_episode(
                    episode, episode_id, demonstration=True
                )
                for transition in _relabel_demonstration_episode(transitions, config):
                    replay.add(transition)
                    loaded_transitions += 1
            finally:
                episode.close()
        replay.flush()
        print(
            f"[learner] 示范回放 v4 已就绪: transitions={loaded_transitions} "
            f"sequences={replay.sequence_count}",
            flush=True,
        )
    else:
        print(
            f"[learner] 已加载示范回放 v4: sequences={replay.sequence_count}",
            flush=True,
        )
    return replay


def _sample_training_batch(
    online: DiskPrioritizedSequenceReplay,
    demonstrations: DiskPrioritizedSequenceReplay | None,
    batch_size: int,
    demo_ratio: float,
    beta: float,
    rng: np.random.Generator,
) -> ReplayBatch:
    demo_count = 0
    if demonstrations is not None and demonstrations.sequence_count:
        demo_count = min(int(round(batch_size * demo_ratio)), demonstrations.sequence_count)
    online_count = batch_size - demo_count
    if online.sequence_count < online_count:
        missing = online_count - online.sequence_count
        demo_count = min(batch_size, demo_count + missing)
        online_count = batch_size - demo_count
    batches: list[ReplayBatch] = []
    if online_count:
        batches.append(online.sample(online_count, beta, rng))
    if demo_count:
        if demonstrations is None or demonstrations.sequence_count < demo_count:
            raise RuntimeError("not enough online or demonstration sequences")
        batches.append(demonstrations.sample(demo_count, beta, rng))
    return concatenate_batches(batches)


def learner_worker(
    config_path: str,
    transition_queue,
    weight_queue,
    stop_event,
    dataset_path: str | None,
    checkpoint_path: str | None,
    boss_id: str | None,
    metric_queue,
) -> None:
    config = load_config(config_path)
    if boss_id:
        config.environment.boss_id = boss_id
    rng = np.random.default_rng(config.training.random_seed)
    data_version = None
    if dataset_path:
        data_version = TrajectoryDataset(
            dataset_path, boss_id=config.environment.boss_id
        ).version
    agent = R2D3Agent(len(HUD_KEYS), ACTION_MASK_SIZE, config.model)
    checkpoint_extra: dict[str, object] = {}
    if checkpoint_path and Path(checkpoint_path).exists():
        checkpoint_extra = load_checkpoint(
            agent,
            checkpoint_path,
            expected_config_hash=config.fingerprint(),
            expected_data_version=data_version,
        )
    environment_steps = (
        max(0, int(checkpoint_extra.get("environment_steps", 0)))
        if checkpoint_extra.get("stage") == "online_r2d3"
        else 0
    )
    sequence_length = config.model.burn_in + config.model.unroll + config.model.n_step
    frame_shape = (
        config.capture.observation_height,
        config.capture.observation_width,
        3,
    )
    online = DiskPrioritizedSequenceReplay(
        online_replay_directory(config),
        config.replay.capacity_frames,
        frame_shape,
        len(HUD_KEYS),
        ACTION_MASK_SIZE,
        sequence_length,
        config.model.burn_in,
        config.replay.priority_alpha,
    )
    demonstrations = _bootstrap_demo_replay(config, dataset_path, sequence_length, frame_shape)
    _put_latest(weight_queue, (cpu_state_dict(agent), environment_steps))
    update_budget = 0.0
    last_sync = time.monotonic()
    last_checkpoint = time.monotonic()
    latest_checkpoint = Path(config.training.checkpoint_directory) / "latest.pt"
    try:
        while not stop_event.is_set():
            try:
                transition = transition_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if transition == _TRANSITION_STREAM_END:
                break
            if not isinstance(transition, Transition):
                raise TypeError(f"unexpected transition payload: {type(transition)!r}")
            online.add(transition)
            environment_steps += 1
            update_budget += config.training.updates_per_environment_step
            while (
                update_budget >= 1.0
                and online.sequence_count + (demonstrations.sequence_count if demonstrations else 0)
                >= config.replay.minimum_sequences
            ):
                beta = min(1.0, config.replay.priority_beta_start + agent.learner_steps / 1_000_000)
                batch = _sample_training_batch(
                    online,
                    demonstrations,
                    config.model.batch_size,
                    config.replay.demo_ratio,
                    beta,
                    rng,
                )
                learner_started = time.perf_counter()
                learner_metrics = agent.learn(batch)
                learner_latency_ms = (time.perf_counter() - learner_started) * 1000.0
                online_mask = ~batch.demonstrations
                if online_mask.any():
                    online.update_priorities(
                        batch.start_ids[online_mask], learner_metrics.priorities[online_mask]
                    )
                if demonstrations is not None and batch.demonstrations.any():
                    demonstrations.update_priorities(
                        batch.start_ids[batch.demonstrations],
                        learner_metrics.priorities[batch.demonstrations],
                    )
                update_budget -= 1.0
                emit_metric(
                    metric_queue,
                    "train",
                    "learner",
                    environment_steps=environment_steps,
                    learner_steps=agent.learner_steps,
                    replay_sequences=online.sequence_count,
                    loss=learner_metrics.loss,
                    td_loss=learner_metrics.td_loss,
                    demo_loss=learner_metrics.demo_loss,
                    mean_q=learner_metrics.mean_q,
                    mean_target=learner_metrics.mean_target,
                    gradient_norm=learner_metrics.gradient_norm,
                    update_skipped=learner_metrics.update_skipped,
                    learner_latency_ms=learner_latency_ms,
                    replay_utilization=agent.learner_steps / max(environment_steps, 1),
                )
            now = time.monotonic()
            if now - last_sync >= config.training.weight_sync_seconds:
                _put_latest(weight_queue, (cpu_state_dict(agent), environment_steps))
                last_sync = now
            if now - last_checkpoint >= config.training.checkpoint_interval_seconds:
                save_checkpoint(
                    agent,
                    latest_checkpoint,
                    config.fingerprint(),
                    {"stage": "online_r2d3", "environment_steps": environment_steps},
                    data_version=data_version,
                )
                online.flush()
                if demonstrations:
                    demonstrations.flush()
                last_checkpoint = now
    finally:
        save_checkpoint(
            agent,
            latest_checkpoint,
            config.fingerprint(),
            {"stage": "online_r2d3", "environment_steps": environment_steps},
            data_version=data_version,
        )
        online.flush()
        if demonstrations:
            demonstrations.flush()
        stop_event.set()


def run_training(
    config_path: str,
    dataset_path: str | None = None,
    checkpoint_path: str | None = None,
    boss_id: str | None = None,
    max_environment_steps: int | None = None,
) -> None:
    if max_environment_steps is not None and max_environment_steps <= 0:
        raise ValueError("max_environment_steps must be positive")
    config = load_config(config_path)
    if boss_id:
        config.environment.boss_id = boss_id
    context = mp.get_context("spawn")
    transition_queue = context.Queue(maxsize=512)
    weight_queue = context.Queue(maxsize=1)
    metric_queue = context.Queue(maxsize=4096)
    stop_event = context.Event()
    metric_process = context.Process(
        target=metric_worker,
        args=(config.training.metrics_directory, metric_queue),
        name="wukong-metrics",
    )
    learner = context.Process(
        target=learner_worker,
        args=(
            config_path,
            transition_queue,
            weight_queue,
            stop_event,
            dataset_path,
            checkpoint_path,
            boss_id,
            metric_queue,
        ),
        name="wukong-learner",
    )
    torch.set_num_threads(config.training.actor_cpu_threads)
    actor_agent = R2D3Agent(len(HUD_KEYS), ACTION_MASK_SIZE, config.model, device="cpu")
    environment = build_live_environment(config)
    rng = np.random.default_rng(config.training.random_seed + 1)
    environment_steps = 0
    run_environment_steps = 0
    episode_reward = 0.0
    episode_reward_boss = 0.0
    episode_reward_self = 0.0
    episode_reward_tick = 0.0
    episode_reward_terminal = 0.0
    episode_reward_clipping = 0.0
    dropped = 0
    dropped_metrics = 0
    state = actor_agent.initial_state()
    metric_process.start()
    learner.start()
    try:
        print(
            "[train] Learner 已启动，正在加载 BC 检查点并准备示范回放；"
            "Actor 在收到首份权重前不会操作游戏...",
            flush=True,
        )
        initial_weight_deadline = time.monotonic() + 180.0
        while True:
            if not learner.is_alive():
                raise RuntimeError("learner process exited before publishing initial weights")
            try:
                weights, environment_steps = weight_queue.get(timeout=1.0)
                break
            except queue.Empty:
                if time.monotonic() >= initial_weight_deadline:
                    raise TimeoutError("learner did not publish initial weights within 180 seconds")
        actor_agent.online.load_state_dict(weights)
        state = actor_agent.initial_state()
        print("[train] BC 权重已加载，等待可靠战斗画面...", flush=True)
        observation = environment.reset()
        print("[train] 已进入在线 R2D3/DQfD 训练。Ctrl+C 可安全保存并退出。", flush=True)
        actor_started = time.monotonic()
        episode_started = actor_started
        next_progress = actor_started + 5.0
        metrics_process_warning_emitted = False
        while not stop_event.is_set():
            if not learner.is_alive():
                raise RuntimeError("learner process exited unexpectedly")
            try:
                while True:
                    weights, _learner_environment_steps = weight_queue.get_nowait()
                    actor_agent.online.load_state_dict(weights)
            except queue.Empty:
                pass
            epsilon = actor_agent.exploration(
                environment_steps,
                config.training.actor_epsilon_start,
                config.training.actor_epsilon_end,
                config.training.epsilon_decay_steps,
            )
            inference_started = time.perf_counter()
            action, state, q_values = actor_agent.act(observation, state, epsilon, rng)
            actor_latency_ms = (time.perf_counter() - inference_started) * 1000.0
            transition = environment.step(action)
            reward_breakdown = environment.last_reward_breakdown
            try:
                transition_queue.put_nowait(transition)
            except queue.Full:
                dropped += 1
            environment_steps += 1
            run_environment_steps += 1
            episode_reward += transition.reward
            episode_reward_boss += reward_breakdown.boss_damage
            episode_reward_self += reward_breakdown.self_damage
            episode_reward_tick += reward_breakdown.tick
            episode_reward_terminal += reward_breakdown.terminal
            episode_reward_clipping += reward_breakdown.clipping
            if not metric_process.is_alive() and not metrics_process_warning_emitted:
                print(
                    "[metrics] 指标进程已退出；训练仍继续，但实时监控数据会不完整。",
                    flush=True,
                )
                metrics_process_warning_emitted = True
            telemetry_status = None
            telemetry_client = getattr(environment.perception, "client", None)
            if telemetry_client is not None:
                telemetry_status = telemetry_client.status()
            metric_ok = emit_metric(
                metric_queue,
                "actor",
                "actor_step",
                environment_steps=environment_steps,
                episode_id=transition.episode_id,
                step_id=transition.step_id,
                action=transition.action.name,
                movement=transition.action.movement.name,
                combat=transition.action.combat.name,
                policy_intervention=environment.last_policy_intervention,
                reward=transition.reward,
                reward_boss_damage=reward_breakdown.boss_damage,
                reward_self_damage=reward_breakdown.self_damage,
                reward_tick=reward_breakdown.tick,
                reward_terminal=reward_breakdown.terminal,
                reward_clipping=reward_breakdown.clipping,
                epsilon=epsilon,
                action_entropy=actor_agent.action_entropy(q_values, observation.action_mask),
                actor_latency_ms=actor_latency_ms,
                environment_fps=run_environment_steps
                / max(time.monotonic() - actor_started, 1.0e-6),
                observation_latency_ms=environment.metrics.observation_latency_ms,
                deadline_miss_rate=environment.metrics.deadline_miss_rate,
                invalid_observation_rate=environment.metrics.invalid_observation_rate,
                dropped_transitions=dropped,
                dropped_metrics=dropped_metrics,
                detection_confidence=float(transition.next_observation.feature_confidence.mean()),
                boss_health=environment.terminal.last_valid_boss,
                self_health=environment.terminal.last_valid_self,
                boss_damage=100.0 - environment.terminal.last_valid_boss,
                potion_allowed=bool(
                    observation.action_mask[
                        MovementToken.size() + int(CombatToken.DRINK_POTION)
                    ]
                ),
                skill4_allowed=bool(
                    observation.action_mask[
                        MovementToken.size() + int(CombatToken.SKILL_4)
                    ]
                ),
                transformation_active=bool(
                    observation.measurements.get("transformation_active")
                    and observation.measurements["transformation_active"].valid
                    and observation.measurements["transformation_active"].value > 0.5
                ),
                telemetry_connected=(
                    None if telemetry_status is None else telemetry_status.connected
                ),
                telemetry_fresh=(
                    None if telemetry_status is None else telemetry_status.fresh
                ),
                telemetry_age_ms=(
                    None if telemetry_status is None else telemetry_status.age_ms
                ),
                **actor_agent.action_diagnostics(q_values, observation.action_mask),
            )
            if not metric_ok:
                dropped_metrics += 1
            observation = transition.next_observation
            now = time.monotonic()
            if now >= next_progress:
                print(
                    f"[train] step={environment_steps} episode={transition.episode_id} "
                    f"epsilon={epsilon:.3f} reward={episode_reward:.3f} "
                    f"boss_hp={environment.terminal.last_valid_boss:.1f}% "
                    f"self_hp={environment.terminal.last_valid_self:.1f}% "
                    f"action={transition.action.name} "
                    f"intervention={environment.last_policy_intervention or '-'} "
                    f"deadline_miss={environment.metrics.deadline_miss_rate:.1%}",
                    flush=True,
                )
                next_progress = now + 5.0
            if (
                max_environment_steps is not None
                and run_environment_steps >= max_environment_steps
            ):
                print(
                    f"[train] 已达到验证上限 {max_environment_steps} steps，正在保存退出...",
                    flush=True,
                )
                break
            if transition.done:
                result = EpisodeResult(
                    episode_id=transition.episode_id,
                    state=observation.episode_state,
                    reward=episode_reward,
                    steps=transition.step_id + 1,
                    duration=time.monotonic() - episode_started,
                    boss_health=environment.terminal.last_valid_boss,
                    self_health=environment.terminal.last_valid_self,
                    damage_dealt=100.0 - environment.terminal.last_valid_boss,
                    damage_taken=100.0 - environment.terminal.last_valid_self,
                )
                episode_metric = {
                    **asdict(result),
                    "reward_boss_damage": episode_reward_boss,
                    "reward_self_damage": episode_reward_self,
                    "reward_tick": episode_reward_tick,
                    "reward_terminal": episode_reward_terminal,
                    "reward_clipping": episode_reward_clipping,
                }
                if not emit_metric(metric_queue, "actor", "episode", **episode_metric):
                    dropped_metrics += 1
                print(
                    f"[train] episode_end state={result.state.value} "
                    f"reward={result.reward:.3f} steps={result.steps} "
                    f"boss_hp={result.boss_health:.1f}% self_hp={result.self_health:.1f}%",
                    flush=True,
                )
                episode_reward = 0.0
                episode_reward_boss = 0.0
                episode_reward_self = 0.0
                episode_reward_tick = 0.0
                episode_reward_terminal = 0.0
                episode_reward_clipping = 0.0
                episode_started = time.monotonic()
                state = actor_agent.initial_state()
                observation = environment.reset()
    except KeyboardInterrupt:
        pass
    finally:
        environment.close()
        if learner.is_alive() and not stop_event.is_set():
            try:
                # The sentinel is queued behind every accepted transition, so
                # bounded and interrupted runs persist all collected experience.
                transition_queue.put(_TRANSITION_STREAM_END, timeout=5)
            except queue.Full:
                stop_event.set()
        learner.join(timeout=30)
        if learner.is_alive():
            stop_event.set()
            learner.terminate()
            learner.join(timeout=5)
            print("[train] Learner 未在 30 秒内退出，已强制终止。", flush=True)
        try:
            metric_queue.put(None, timeout=2)
        except queue.Full:
            pass
        metric_process.join(timeout=5)
        if metric_process.is_alive():
            metric_process.terminate()
            metric_process.join(timeout=2)
        for ipc_queue in (transition_queue, weight_queue, metric_queue):
            ipc_queue.cancel_join_thread()
            ipc_queue.close()
