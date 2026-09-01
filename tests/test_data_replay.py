from __future__ import annotations

import numpy as np

from wukong_rl.data import TrajectoryDataset, save_episode, transitions_from_episode
from wukong_rl.replay import DiskPrioritizedSequenceReplay
from wukong_rl.types import ActionToken, HUD_KEYS

from conftest import make_transition


def test_episode_roundtrip_and_schema_validation(tmp_path) -> None:
    transitions = [make_transition(1, index, done=index == 9) for index in range(10)]
    episode_dir = save_episode(tmp_path, "yinhu", transitions, "hash")
    dataset = TrajectoryDataset(tmp_path, boss_id="yinhu")
    episode = next(dataset.episodes())
    assert episode.directory == episode_dir
    assert len(episode) == 10
    rebuilt = list(transitions_from_episode(episode, 7))
    assert rebuilt[-1].terminated
    assert rebuilt[0].observation.frame.dtype == np.uint8
    assert "raw_inputs" in episode.trajectory
    assert dataset.version
    episode.close()


def test_dataset_import_maps_masked_human_intent_to_idle(tmp_path) -> None:
    transition = make_transition(1, 0, done=True)
    transition.action = ActionToken.SKILL_1
    transition.observation.action_mask[int(ActionToken.SKILL_1)] = False
    transition.next_observation.previous_action = ActionToken.SKILL_1
    transition.raw_input = '{"token":"SKILL_1"}'
    save_episode(tmp_path, "yinhu", [transition], "hash")
    episode = next(TrajectoryDataset(tmp_path, boss_id="yinhu").episodes())
    rebuilt = list(transitions_from_episode(episode, 1))
    assert rebuilt[0].action is ActionToken.IDLE
    assert rebuilt[0].next_observation.previous_action is ActionToken.IDLE
    assert "SKILL_1" in rebuilt[0].raw_input
    episode.close()


def test_dataset_rejects_non_final_episode_boundary(tmp_path) -> None:
    transitions = [make_transition(1, index, done=index in {2, 4}) for index in range(5)]
    save_episode(tmp_path, "yinhu", transitions, "hash")
    try:
        next(TrajectoryDataset(tmp_path, boss_id="yinhu").episodes())
    except ValueError as error:
        assert "episode boundary" in str(error)
    else:
        raise AssertionError("invalid trajectory was accepted")


def test_disk_replay_keeps_raw_frames_and_samples_padded_terminal_sequences(tmp_path) -> None:
    sequence_length = 8
    replay = DiskPrioritizedSequenceReplay(
        tmp_path,
        capacity=64,
        frame_shape=(24, 32, 3),
        feature_dim=len(HUD_KEYS),
        action_dim=ActionToken.size(),
        sequence_length=sequence_length,
        burn_in=2,
        reset=True,
    )
    for index in range(20):
        replay.add(make_transition(1, index, done=index == 19))
    assert replay.frames.dtype == np.uint8
    assert replay.sequence_count > 0
    batch = replay.sample(2, beta=0.4, rng=np.random.default_rng(1))
    assert batch.frames.shape == (2, sequence_length + 1, 24, 32, 3)
    assert batch.frames.dtype == np.uint8
    assert batch.actions.shape == (2, sequence_length)
