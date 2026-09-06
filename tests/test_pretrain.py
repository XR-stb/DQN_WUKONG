from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch.nn.functional as torch_functional

from wukong_rl.agent import R2D3Agent
from wukong_rl.config import ModelConfig
from wukong_rl.data import TrajectoryDataset, save_episode
from wukong_rl.pretrain import (
    BcMetrics,
    ClosedLoopMetrics,
    BehaviorCloningTrainer,
    assess_bc_release,
    core_balanced_score,
)
from wukong_rl.types import (
    ACTION_MASK_SIZE,
    COMBAT_MASK_SLICE,
    ActionCommand,
    CombatToken,
    HUD_KEYS,
    MovementToken,
)

from conftest import make_transition


def test_behavior_cloning_can_overfit_tiny_demonstration(tmp_path) -> None:
    transitions = [make_transition(1, index, done=index == 15) for index in range(16)]
    save_episode(tmp_path, "yinhu", transitions, "hash")
    paths = TrajectoryDataset(tmp_path, boss_id="yinhu").manifest_paths
    config = ModelConfig(hidden_size=32, burn_in=1, unroll=4, n_step=1, batch_size=2)
    agent = R2D3Agent(len(HUD_KEYS), ACTION_MASK_SIZE, config, device="cpu")
    trainer = BehaviorCloningTrainer(agent, sequence_length=4, seed=1)
    before = trainer.run_epoch(paths, batch_size=2, steps=1, train=False)
    trainer.run_epoch(paths, batch_size=2, steps=12, train=True)
    after = trainer.run_epoch(paths, batch_size=2, steps=2, train=False)
    assert after.loss < before.loss
    assert after.accuracy >= before.accuracy


def test_behavior_cloning_uses_effective_masked_actions(tmp_path) -> None:
    transitions = [make_transition(1, index, done=index == 7) for index in range(8)]
    for transition in transitions:
        transition.action = ActionCommand(combat=CombatToken.SKILL_1)
        transition.observation.action_mask[
            COMBAT_MASK_SLICE.start + int(CombatToken.SKILL_1)
        ] = False
        transition.next_observation.previous_action = ActionCommand(combat=CombatToken.SKILL_1)
    save_episode(tmp_path, "yinhu", transitions, "hash")
    paths = TrajectoryDataset(tmp_path, boss_id="yinhu").manifest_paths
    config = ModelConfig(hidden_size=32, burn_in=1, unroll=4, n_step=1, batch_size=2)
    agent = R2D3Agent(len(HUD_KEYS), ACTION_MASK_SIZE, config, device="cpu")
    trainer = BehaviorCloningTrainer(agent, sequence_length=4, seed=1)
    episodes = trainer._load_episodes(paths)
    try:
        batch = trainer._sample_batch(episodes, 2)
    finally:
        for episode in episodes:
            episode.close()
    action_masks, previous_actions, actions = batch[3], batch[4], batch[-1]
    assert not action_masks[..., COMBAT_MASK_SLICE.start + int(CombatToken.SKILL_1)].any()
    assert np.all(previous_actions == 0)
    assert np.all(actions == 0)
    metrics = trainer.run_epoch(paths, batch_size=2, steps=1, train=True)
    assert np.isfinite(metrics.loss)


def test_behavior_cloning_weights_rare_effective_actions(tmp_path) -> None:
    transitions = [make_transition(1, index, done=index == 15) for index in range(16)]
    transitions[-1].action = ActionCommand(combat=CombatToken.DODGE)
    save_episode(tmp_path, "yinhu", transitions, "hash")
    paths = TrajectoryDataset(tmp_path, boss_id="yinhu").manifest_paths
    config = ModelConfig(hidden_size=32, burn_in=1, unroll=4, n_step=1, batch_size=2)
    agent = R2D3Agent(len(HUD_KEYS), ACTION_MASK_SIZE, config, device="cpu")
    trainer = BehaviorCloningTrainer(agent, sequence_length=4, seed=1)
    _, combat_weights = trainer.estimate_class_weights(paths)
    assert combat_weights[int(CombatToken.DODGE)] > combat_weights[int(CombatToken.LIGHT_ATTACK)]
    assert combat_weights[int(CombatToken.SKILL_4)] == 1


def test_behavior_cloning_burn_in_only_scores_the_unroll(tmp_path) -> None:
    transitions = [make_transition(1, index, done=index == 11) for index in range(12)]
    save_episode(tmp_path, "yinhu", transitions, "hash")
    paths = TrajectoryDataset(tmp_path, boss_id="yinhu").manifest_paths
    config = ModelConfig(hidden_size=32, burn_in=2, unroll=3, n_step=1, batch_size=2)
    agent = R2D3Agent(len(HUD_KEYS), ACTION_MASK_SIZE, config, device="cpu")
    trainer = BehaviorCloningTrainer(agent, sequence_length=3, burn_in=2, seed=1)
    episodes = trainer._load_episodes(paths)
    try:
        batch = trainer._sample_batch(episodes, 2)
        assert batch[0].shape[1] == 5
        assert batch[-2][0]
        _, predicted, target = trainer._step(episodes, 2, train=False)
    finally:
        for episode in episodes:
            episode.close()
    assert predicted.shape == target.shape == (6, 2)


def test_behavior_cloning_samples_multiple_episode_prefixes(tmp_path) -> None:
    transitions = [make_transition(1, index, done=index == 15) for index in range(16)]
    save_episode(tmp_path, "yinhu", transitions, "hash")
    paths = TrajectoryDataset(tmp_path, boss_id="yinhu").manifest_paths
    config = ModelConfig(hidden_size=32, burn_in=2, unroll=3, n_step=1, batch_size=8)
    agent = R2D3Agent(len(HUD_KEYS), ACTION_MASK_SIZE, config, device="cpu")
    trainer = BehaviorCloningTrainer(
        agent, sequence_length=3, burn_in=2, seed=1, episode_prefix_fraction=0.25
    )
    episodes = trainer._load_episodes(paths)
    try:
        batch = trainer._sample_batch(episodes, 8)
    finally:
        for episode in episodes:
            episode.close()
    assert batch[-2].sum() >= 2


def test_behavior_cloning_supports_fully_autoregressive_action_feedback(tmp_path) -> None:
    transitions = [make_transition(1, index, done=index == 7) for index in range(8)]
    save_episode(tmp_path, "yinhu", transitions, "hash")
    paths = TrajectoryDataset(tmp_path, boss_id="yinhu").manifest_paths
    config = ModelConfig(hidden_size=32, burn_in=2, unroll=3, n_step=1, batch_size=2)
    agent = R2D3Agent(len(HUD_KEYS), ACTION_MASK_SIZE, config, device="cpu")
    trainer = BehaviorCloningTrainer(agent, sequence_length=3, burn_in=2, seed=1)

    metrics = trainer.run_epoch(
        paths,
        batch_size=2,
        steps=1,
        train=True,
        model_feedback_probability=1.0,
    )

    assert np.isfinite(metrics.loss)
    assert metrics.movement_confusion.sum() == 6
    assert metrics.combat_confusion.sum() == 6


def test_behavior_cloning_adds_cold_start_loss_for_episode_prefix(tmp_path, monkeypatch) -> None:
    transitions = [make_transition(1, index, done=index == 7) for index in range(8)]
    transitions[0].action = ActionCommand(combat=CombatToken.DODGE)
    save_episode(tmp_path, "yinhu", transitions, "hash")
    paths = TrajectoryDataset(tmp_path, boss_id="yinhu").manifest_paths
    config = ModelConfig(hidden_size=32, burn_in=2, unroll=3, n_step=1, batch_size=1)
    agent = R2D3Agent(len(HUD_KEYS), ACTION_MASK_SIZE, config, device="cpu")
    trainer = BehaviorCloningTrainer(agent, sequence_length=3, burn_in=2, seed=1)
    calls = []
    original_cross_entropy = torch_functional.cross_entropy

    def track_cross_entropy(*args, **kwargs):
        calls.append(args[0].shape[0])
        return original_cross_entropy(*args, **kwargs)

    monkeypatch.setattr("wukong_rl.pretrain.functional.cross_entropy", track_cross_entropy)
    episodes = trainer._load_episodes(paths)
    try:
        trainer._step(episodes, 1, train=False)
    finally:
        for episode in episodes:
            episode.close()

    assert calls == [3, 3, 2, 2]


def test_core_balanced_score_penalizes_zero_recall_core_action() -> None:
    movement = np.eye(MovementToken.size(), dtype=np.int64) * 60
    combat_balanced = np.eye(CombatToken.size(), dtype=np.int64) * 60
    combat_collapsed = combat_balanced.copy()
    combat_collapsed[int(CombatToken.DODGE)] = 0
    combat_collapsed[int(CombatToken.DODGE), int(CombatToken.NONE)] = 100

    def metrics(combat: np.ndarray, accuracy: float) -> BcMetrics:
        movement_recall = {index: 1.0 for index in range(MovementToken.size())}
        combat_recall = {
            index: float(combat[index, index] / max(combat[index].sum(), 1))
            for index in range(CombatToken.size())
        }
        return BcMetrics(
            1.0,
            accuracy,
            accuracy,
            accuracy,
            movement_recall,
            combat_recall,
            movement,
            combat,
        )

    assert core_balanced_score(metrics(combat_balanced, 0.60)) > core_balanced_score(
        metrics(combat_collapsed, 0.70)
    )


def test_bc_release_gate_rejects_long_idle_or_missing_dodge() -> None:
    healthy = ClosedLoopMetrics(
        joint_accuracy=0.2,
        movement_accuracy=0.4,
        combat_accuracy=0.5,
        static_escape_rate=1.0,
        mean_first_action_step=2.0,
        max_idle_run=32,
        episodes=3,
        movement_active_rate=0.5,
        combat_active_rate=0.2,
        movement_switch_rate=0.1,
        combat_switch_rate=0.1,
        movement_counts={"NONE": 100, "FORWARD": 100},
        combat_counts={"NONE": 160, "LIGHT_ATTACK": 30, "DODGE": 10},
    )
    assert assess_bc_release(healthy) == (True, [])

    collapsed = replace(
        healthy,
        max_idle_run=100,
        combat_counts={"NONE": 190, "LIGHT_ATTACK": 10, "DODGE": 0},
    )
    ready, reasons = assess_bc_release(collapsed)
    assert not ready
    assert "max_idle_run>64" in reasons
    assert "closed_loop_dodge_rate<0.005" in reasons


def test_balanced_sampling_includes_core_action_windows(tmp_path) -> None:
    transitions = [make_transition(1, index, done=index == 31) for index in range(32)]
    for transition in transitions:
        transition.action = ActionCommand()
    transitions[16].action = ActionCommand(combat=CombatToken.DODGE)
    save_episode(tmp_path, "yinhu", transitions, "hash")
    paths = TrajectoryDataset(tmp_path, boss_id="yinhu").manifest_paths
    config = ModelConfig(hidden_size=32, burn_in=2, unroll=4, n_step=1, batch_size=4)
    agent = R2D3Agent(len(HUD_KEYS), ACTION_MASK_SIZE, config, device="cpu")
    trainer = BehaviorCloningTrainer(
        agent,
        sequence_length=4,
        burn_in=2,
        seed=2,
        episode_prefix_fraction=0.0 + 0.25,
        balanced_window_fraction=0.5,
    )
    episodes = trainer._load_episodes(paths)
    try:
        actions = trainer._sample_batch(episodes, 4)[-1]
    finally:
        for episode in episodes:
            episode.close()
    assert np.any(actions[..., 1] == int(CombatToken.DODGE))
