from __future__ import annotations

import numpy as np

from wukong_rl.agent import R2D3Agent
from wukong_rl.config import ModelConfig
from wukong_rl.data import TrajectoryDataset, save_episode
from wukong_rl.pretrain import BehaviorCloningTrainer
from wukong_rl.types import ActionToken, HUD_KEYS

from conftest import make_transition


def test_behavior_cloning_can_overfit_tiny_demonstration(tmp_path) -> None:
    transitions = [make_transition(1, index, done=index == 15) for index in range(16)]
    save_episode(tmp_path, "yinhu", transitions, "hash")
    paths = TrajectoryDataset(tmp_path, boss_id="yinhu").manifest_paths
    config = ModelConfig(hidden_size=32, burn_in=1, unroll=4, n_step=1, batch_size=2)
    agent = R2D3Agent(len(HUD_KEYS), ActionToken.size(), config, device="cpu")
    trainer = BehaviorCloningTrainer(agent, sequence_length=4, seed=1)
    before = trainer.run_epoch(paths, batch_size=2, steps=1, train=False)
    trainer.run_epoch(paths, batch_size=2, steps=12, train=True)
    after = trainer.run_epoch(paths, batch_size=2, steps=2, train=False)
    assert after.loss < before.loss
    assert after.accuracy >= before.accuracy


def test_behavior_cloning_uses_effective_masked_actions(tmp_path) -> None:
    transitions = [make_transition(1, index, done=index == 7) for index in range(8)]
    for transition in transitions:
        transition.action = ActionToken.SKILL_1
        transition.observation.action_mask[int(ActionToken.SKILL_1)] = False
        transition.next_observation.previous_action = ActionToken.SKILL_1
    save_episode(tmp_path, "yinhu", transitions, "hash")
    paths = TrajectoryDataset(tmp_path, boss_id="yinhu").manifest_paths
    config = ModelConfig(hidden_size=32, burn_in=1, unroll=4, n_step=1, batch_size=2)
    agent = R2D3Agent(len(HUD_KEYS), ActionToken.size(), config, device="cpu")
    trainer = BehaviorCloningTrainer(agent, sequence_length=4, seed=1)
    episodes = trainer._load_episodes(paths)
    try:
        batch = trainer._sample_batch(episodes, 2)
    finally:
        for episode in episodes:
            episode.close()
    action_masks, previous_actions, actions = batch[3], batch[4], batch[-1]
    assert not action_masks[..., int(ActionToken.SKILL_1)].any()
    assert np.all(previous_actions == int(ActionToken.IDLE))
    assert np.all(actions == int(ActionToken.IDLE))
    metrics = trainer.run_epoch(paths, batch_size=2, steps=1, train=True)
    assert np.isfinite(metrics.loss)


def test_behavior_cloning_weights_rare_effective_actions(tmp_path) -> None:
    transitions = [make_transition(1, index, done=index == 15) for index in range(16)]
    transitions[-1].action = ActionToken.DODGE
    save_episode(tmp_path, "yinhu", transitions, "hash")
    paths = TrajectoryDataset(tmp_path, boss_id="yinhu").manifest_paths
    config = ModelConfig(hidden_size=32, burn_in=1, unroll=4, n_step=1, batch_size=2)
    agent = R2D3Agent(len(HUD_KEYS), ActionToken.size(), config, device="cpu")
    trainer = BehaviorCloningTrainer(agent, sequence_length=4, seed=1)
    weights = trainer.estimate_class_weights(paths)
    assert weights[int(ActionToken.DODGE)] > weights[int(ActionToken.LIGHT_ATTACK)]
    assert weights[int(ActionToken.SKILL_4)] == 1
