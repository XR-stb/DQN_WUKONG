from __future__ import annotations

import multiprocessing as mp
import queue
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .actions import FixedRateActionController, PynputInputBackend
from .agent import R2D3Agent
from .capture import create_screen_source
from .checkpoint import cpu_state_dict, load_checkpoint, save_checkpoint
from .config import PipelineConfig, load_config
from .data import TrajectoryDataset, transitions_from_episode
from .environment import LegacyRestartHook, WukongEnvironment
from .metrics import emit_metric, metric_worker
from .perception import ScreenPerception
from .replay import DiskPrioritizedSequenceReplay, ReplayBatch, concatenate_batches
from .types import ActionToken, EpisodeResult, EpisodeState, HUD_KEYS, Transition


def build_live_environment(config: PipelineConfig) -> WukongEnvironment:
    source = create_screen_source(config.capture)
    perception = ScreenPerception(
        config.perception, config.capture.width, config.capture.height
    )
    controller = FixedRateActionController(PynputInputBackend())
    restart = LegacyRestartHook(config.environment.restart_action)
    return WukongEnvironment(config, source, perception, controller, restart_hook=restart)


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
        Path(config.replay.directory) / f"demonstrations-{dataset.version}",
        capacity,
        frame_shape,
        len(HUD_KEYS),
        ActionToken.size(),
        sequence_length,
        config.model.burn_in,
        config.replay.priority_alpha,
        demonstration=True,
    )
    if replay.sequence_count == 0:
        for episode_id, episode in enumerate(dataset.episodes()):
            try:
                for transition in transitions_from_episode(episode, episode_id, demonstration=True):
                    replay.add(transition)
            finally:
                episode.close()
        replay.flush()
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
    agent = R2D3Agent(len(HUD_KEYS), ActionToken.size(), config.model)
    if checkpoint_path and Path(checkpoint_path).exists():
        load_checkpoint(
            agent,
            checkpoint_path,
            expected_config_hash=config.fingerprint(),
            expected_data_version=data_version,
        )
    sequence_length = config.model.burn_in + config.model.unroll + config.model.n_step
    frame_shape = (
        config.capture.observation_height,
        config.capture.observation_width,
        3,
    )
    online = DiskPrioritizedSequenceReplay(
        Path(config.replay.directory) / "online",
        config.replay.capacity_frames,
        frame_shape,
        len(HUD_KEYS),
        ActionToken.size(),
        sequence_length,
        config.model.burn_in,
        config.replay.priority_alpha,
    )
    demonstrations = _bootstrap_demo_replay(config, dataset_path, sequence_length, frame_shape)
    _put_latest(weight_queue, cpu_state_dict(agent))
    update_budget = 0.0
    environment_steps = 0
    last_sync = time.monotonic()
    last_checkpoint = time.monotonic()
    latest_checkpoint = Path(config.training.checkpoint_directory) / "latest.pt"
    try:
        while not stop_event.is_set():
            try:
                transition: Transition = transition_queue.get(timeout=0.1)
            except queue.Empty:
                transition = None
            if transition is not None:
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
                    learner_latency_ms=learner_latency_ms,
                    replay_utilization=agent.learner_steps / max(environment_steps, 1),
                )
            now = time.monotonic()
            if now - last_sync >= config.training.weight_sync_seconds:
                _put_latest(weight_queue, cpu_state_dict(agent))
                last_sync = now
            if now - last_checkpoint >= config.training.checkpoint_interval_seconds:
                save_checkpoint(
                    agent,
                    latest_checkpoint,
                    config.fingerprint(),
                    {"environment_steps": environment_steps},
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
            {"environment_steps": environment_steps},
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
) -> None:
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
    actor_agent = R2D3Agent(len(HUD_KEYS), ActionToken.size(), config.model, device="cpu")
    environment = build_live_environment(config)
    rng = np.random.default_rng(config.training.random_seed + 1)
    environment_steps = 0
    actor_started = time.monotonic()
    episode_reward = 0.0
    episode_started = time.monotonic()
    dropped = 0
    dropped_metrics = 0
    state = actor_agent.initial_state()
    metric_process.start()
    learner.start()
    try:
        observation = environment.reset()
        while not stop_event.is_set():
            if not learner.is_alive():
                raise RuntimeError("learner process exited unexpectedly")
            weights_updated = False
            try:
                while True:
                    weights = weight_queue.get_nowait()
                    actor_agent.online.load_state_dict(weights)
                    weights_updated = True
            except queue.Empty:
                pass
            if weights_updated:
                state = actor_agent.initial_state()
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
            try:
                transition_queue.put_nowait(transition)
            except queue.Full:
                dropped += 1
            environment_steps += 1
            episode_reward += transition.reward
            metric_ok = emit_metric(
                metric_queue,
                "actor",
                "actor_step",
                environment_steps=environment_steps,
                episode_id=transition.episode_id,
                step_id=transition.step_id,
                action=transition.action.name,
                reward=transition.reward,
                epsilon=epsilon,
                action_entropy=actor_agent.action_entropy(q_values, observation.action_mask),
                actor_latency_ms=actor_latency_ms,
                environment_fps=environment_steps / max(time.monotonic() - actor_started, 1.0e-6),
                observation_latency_ms=environment.metrics.observation_latency_ms,
                deadline_miss_rate=environment.metrics.deadline_miss_rate,
                invalid_observation_rate=environment.metrics.invalid_observation_rate,
                dropped_transitions=dropped,
                dropped_metrics=dropped_metrics,
                detection_confidence=float(transition.next_observation.feature_confidence.mean()),
                boss_health=environment.terminal.last_valid_boss,
                self_health=environment.terminal.last_valid_self,
                boss_damage=100.0 - environment.terminal.last_valid_boss,
            )
            if not metric_ok:
                dropped_metrics += 1
            observation = transition.next_observation
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
                if not emit_metric(metric_queue, "actor", "episode", **asdict(result)):
                    dropped_metrics += 1
                episode_reward = 0.0
                episode_started = time.monotonic()
                state = actor_agent.initial_state()
                observation = environment.reset()
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        stop_event.set()
        environment.close()
        learner.join(timeout=15)
        if learner.is_alive():
            learner.terminate()
            learner.join(timeout=5)
        try:
            metric_queue.put(None, timeout=2)
        except queue.Full:
            pass
        metric_process.join(timeout=5)
        if metric_process.is_alive():
            metric_process.terminate()
            metric_process.join(timeout=2)
