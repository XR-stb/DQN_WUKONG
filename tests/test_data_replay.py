from __future__ import annotations

import json

import numpy as np

from wukong_rl.data import (
    TrajectoryDataset,
    repair_legacy_potion_inputs,
    save_episode,
    transitions_from_episode,
)
from wukong_rl.replay import DiskPrioritizedSequenceReplay
from wukong_rl.types import (
    ACTION_MASK_SIZE,
    COMBAT_MASK_SLICE,
    ActionCommand,
    CombatToken,
    HUD_KEYS,
)

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
    transition.action = ActionCommand(combat=CombatToken.SKILL_1)
    skill_index = COMBAT_MASK_SLICE.start + int(CombatToken.SKILL_1)
    transition.observation.action_mask[skill_index] = False
    transition.next_observation.previous_action = ActionCommand(combat=CombatToken.SKILL_1)
    transition.raw_input = '{"token":"SKILL_1"}'
    save_episode(tmp_path, "yinhu", [transition], "hash")
    episode = next(TrajectoryDataset(tmp_path, boss_id="yinhu").episodes())
    rebuilt = list(transitions_from_episode(episode, 1))
    assert rebuilt[0].action == ActionCommand()
    assert rebuilt[0].next_observation.previous_action == ActionCommand()
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


def test_repair_legacy_q_potion_inputs_is_non_destructive_and_mask_aware(tmp_path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "repaired"
    transitions = [make_transition(1, index, done=index == 4) for index in range(5)]
    raw_inputs = (
        '{"keys":[],"buttons":[],"token":"LIGHT_ATTACK"}',
        '{"keys":["q"],"buttons":[],"token":"LIGHT_ATTACK"}',
        '{"keys":[],"buttons":[],"token":"LIGHT_ATTACK"}',
        '{"keys":["q"],"buttons":[],"token":"LIGHT_ATTACK"}',
        '{"keys":[],"buttons":[],"token":"LIGHT_ATTACK"}',
    )
    for transition, raw_input in zip(transitions, raw_inputs, strict=True):
        transition.raw_input = raw_input
    potion_index = COMBAT_MASK_SLICE.start + int(CombatToken.DRINK_POTION)
    transitions[3].observation.action_mask[potion_index] = False
    episode_dir = save_episode(source, "yinhu", transitions, "hash")
    source_version = TrajectoryDataset(source, boss_id="yinhu").version

    result = repair_legacy_potion_inputs(source, output, boss_id="yinhu")

    original = next(TrajectoryDataset(source, boss_id="yinhu").episodes(memory_map=False))
    repaired = next(TrajectoryDataset(output, boss_id="yinhu").episodes(memory_map=False))
    try:
        assert original.trajectory["actions"][:, 1].tolist() == [int(CombatToken.LIGHT_ATTACK)] * 5
        assert repaired.trajectory["actions"][1, 1] == int(CombatToken.DRINK_POTION)
        assert repaired.trajectory["actions"][3, 1] == int(CombatToken.NONE)
        assert repaired.trajectory["previous_actions"][2, 1] == int(CombatToken.DRINK_POTION)
        assert repaired.trajectory["previous_actions"][4, 1] == int(CombatToken.NONE)
        assert '"token":"DRINK_POTION"' in str(repaired.trajectory["raw_inputs"][1])
        assert result.repaired_inputs == 2
        assert result.executable_potion_actions == 1
        assert result.masked_potion_actions == 1
        assert result.source_version == source_version
        assert result.output_version != source_version
        assert (output / "repair_manifest.json").is_file()
        assert (episode_dir / "frames.npy").read_bytes() == (
            repaired.directory / "frames.npy"
        ).read_bytes()
    finally:
        original.close()
        repaired.close()


def test_disk_replay_keeps_raw_frames_and_samples_padded_terminal_sequences(tmp_path) -> None:
    sequence_length = 8
    replay = DiskPrioritizedSequenceReplay(
        tmp_path,
        capacity=64,
        frame_shape=(24, 32, 3),
        feature_dim=len(HUD_KEYS),
        action_dim=ACTION_MASK_SIZE,
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
    assert batch.actions.shape == (2, sequence_length, 2)


def test_dataset_split_is_result_stratified_and_not_chronological(tmp_path) -> None:
    dataset = object.__new__(TrajectoryDataset)
    paths = []
    specifications = [
        ("lost", 100),
        ("won", 900),
        ("lost", 1000),
        ("won", 1100),
        ("lost", 1200),
        ("won", 3000),
        ("lost", 3200),
        ("lost", 5000),
    ]
    for index, (result, transitions) in enumerate(specifications):
        directory = tmp_path / f"episode-{index:02d}"
        directory.mkdir()
        path = directory / "manifest.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": 3,
                    "episode_id": directory.name,
                    "boss_id": "yinhu",
                    "created_at": float(index),
                    "config_hash": "hash",
                    "transitions": transitions,
                    "frame_shape": [90, 160, 3],
                    "feature_dim": len(HUD_KEYS),
                    "result": result,
                }
            ),
            encoding="utf-8",
        )
        paths.append(path)
    dataset.manifest_paths = paths
    dataset.total_transitions = sum(item[1] for item in specifications)

    training, validation = dataset.split(validation_fraction=0.25)

    validation_results = {
        json.loads(path.read_text(encoding="utf-8"))["result"] for path in validation
    }
    assert len(training) == 6
    assert len(validation) == 2
    assert validation_results == {"won", "lost"}
    assert paths[-1] not in validation
