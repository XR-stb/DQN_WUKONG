from __future__ import annotations

import numpy as np
import torch

from wukong_rl.agent import R2D3Agent
from wukong_rl.checkpoint import load_checkpoint, save_checkpoint
from wukong_rl.config import ModelConfig
from wukong_rl.replay import ReplayBatch
from wukong_rl.types import ActionToken, HUD_KEYS


def random_batch(config: ModelConfig, batch_size: int = 2) -> ReplayBatch:
    length = config.burn_in + config.unroll + config.n_step
    rng = np.random.default_rng(3)
    return ReplayBatch(
        frames=rng.integers(0, 256, (batch_size, length + 1, 24, 32, 3), dtype=np.uint8),
        features=rng.random((batch_size, length + 1, len(HUD_KEYS)), dtype=np.float32),
        confidence=np.ones((batch_size, length + 1, len(HUD_KEYS)), dtype=np.float32),
        action_masks=np.ones((batch_size, length + 1, ActionToken.size()), dtype=np.bool_),
        previous_actions=np.zeros((batch_size, length + 1), dtype=np.int64),
        previous_rewards=np.zeros((batch_size, length + 1), dtype=np.float32),
        actions=rng.integers(0, ActionToken.size(), (batch_size, length), dtype=np.int64),
        rewards=rng.standard_normal((batch_size, length), dtype=np.float32),
        terminated=np.zeros((batch_size, length), dtype=np.bool_),
        truncated=np.zeros((batch_size, length), dtype=np.bool_),
        weights=np.ones(batch_size, dtype=np.float32),
        start_ids=np.arange(batch_size, dtype=np.int64),
        demonstrations=np.asarray([False, True], dtype=np.bool_),
    )


def test_visual_encoder_receives_gradient_and_checkpoint_roundtrips(tmp_path) -> None:
    config = ModelConfig(hidden_size=64, burn_in=2, unroll=4, n_step=2, batch_size=2)
    agent = R2D3Agent(len(HUD_KEYS), ActionToken.size(), config, device="cpu")
    before = next(agent.online.visual.parameters()).detach().clone()
    metrics = agent.learn(random_batch(config))
    after = next(agent.online.visual.parameters()).detach()
    assert metrics.loss > 0
    assert not torch.equal(before, after)
    path = save_checkpoint(
        agent,
        tmp_path / "model.pt",
        "config-hash",
        normalization_state={"hud": "identity"},
        data_version="dataset-v2",
    )
    restored = R2D3Agent(len(HUD_KEYS), ActionToken.size(), config, device="cpu")
    load_checkpoint(restored, path, expected_config_hash="config-hash")
    assert restored.learner_steps == agent.learner_steps
    for expected, actual in zip(agent.online.parameters(), restored.online.parameters()):
        assert torch.equal(expected, actual)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["normalization_state"] == {"hud": "identity"}
    assert payload["data_version"] == "dataset-v2"


def test_learner_canonicalizes_masked_demonstration_actions() -> None:
    config = ModelConfig(hidden_size=32, burn_in=1, unroll=3, n_step=1, batch_size=2)
    batch = random_batch(config)
    index = config.burn_in
    batch.actions[1, index] = int(ActionToken.SKILL_1)
    batch.action_masks[1, index, int(ActionToken.SKILL_1)] = False
    agent = R2D3Agent(len(HUD_KEYS), ActionToken.size(), config, device="cpu")
    metrics = agent.learn(batch)
    assert np.isfinite(metrics.loss)
    assert np.isfinite(metrics.demo_loss)


def test_batched_visual_encoding_matches_regular_network_forward() -> None:
    config = ModelConfig(hidden_size=32, burn_in=1, unroll=3, n_step=1, batch_size=2)
    agent = R2D3Agent(len(HUD_KEYS), ActionToken.size(), config, device="cpu")
    batch = random_batch(config)
    frames = torch.from_numpy(batch.frames[:, :4])
    features = torch.from_numpy(batch.features[:, :4])
    confidence = torch.from_numpy(batch.confidence[:, :4])
    previous_actions = torch.from_numpy(batch.previous_actions[:, :4])
    previous_rewards = torch.from_numpy(batch.previous_rewards[:, :4])

    expected, _ = agent.online(
        frames, features, confidence, previous_actions, previous_rewards
    )
    actual, _ = agent.online.forward_from_visual(
        agent.online.encode_visual(frames),
        features,
        confidence,
        previous_actions,
        previous_rewards,
    )

    assert torch.equal(expected, actual)
