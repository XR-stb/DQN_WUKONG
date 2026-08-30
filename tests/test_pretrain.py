from __future__ import annotations

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
