from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .types import ActionToken, Transition


@dataclass(slots=True)
class ReplayBatch:
    frames: np.ndarray
    features: np.ndarray
    confidence: np.ndarray
    action_masks: np.ndarray
    previous_actions: np.ndarray
    previous_rewards: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    weights: np.ndarray
    start_ids: np.ndarray
    demonstrations: np.ndarray


class DiskPrioritizedSequenceReplay:
    """Disk-mapped transition ring with prioritized recurrent sequence starts.

    Raw frames are stored as uint8.  Candidate starts are added only when a full
    sequence exists or an episode ended, so samples never cross episode boundaries.
    """

    SCHEMA_VERSION = 3

    def __init__(
        self,
        directory: str | Path,
        capacity: int,
        frame_shape: tuple[int, int, int],
        feature_dim: int,
        action_dim: int,
        sequence_length: int,
        burn_in: int,
        alpha: float = 0.6,
        demonstration: bool = False,
        reset: bool = False,
    ) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.capacity = int(capacity)
        self.frame_shape = tuple(frame_shape)
        self.feature_dim = int(feature_dim)
        self.action_dim = int(action_dim)
        self.sequence_length = int(sequence_length)
        self.burn_in = int(burn_in)
        self.alpha = float(alpha)
        self.demonstration = bool(demonstration)
        self._meta_path = self.directory / "metadata.json"
        if reset:
            self._remove_storage()
        self._open_storage()
        self._load_metadata()

    def _remove_storage(self) -> None:
        for path in self.directory.glob("*.npy"):
            path.unlink()
        if self._meta_path.exists():
            self._meta_path.unlink()

    def _array(self, name: str, dtype, shape: tuple[int, ...], fill=None):
        path = self.directory / f"{name}.npy"
        if path.exists():
            array = np.lib.format.open_memmap(path, mode="r+")
            if array.shape != shape or array.dtype != np.dtype(dtype):
                raise ValueError(f"replay array mismatch for {path}")
            return array
        array = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
        if fill is not None:
            array.fill(fill)
        return array

    def _open_storage(self) -> None:
        c = self.capacity
        self.frames = self._array("frames", np.uint8, (c, *self.frame_shape))
        self.features = self._array("features", np.float32, (c, self.feature_dim))
        self.confidence = self._array("confidence", np.float32, (c, self.feature_dim))
        self.action_masks = self._array("action_masks", np.bool_, (c, self.action_dim))
        self.previous_actions = self._array("previous_actions", np.int16, (c, 2))
        self.previous_rewards = self._array("previous_rewards", np.float32, (c,))
        self.actions = self._array("actions", np.int16, (c, 2))
        self.rewards = self._array("rewards", np.float32, (c,))
        self.terminated = self._array("terminated", np.bool_, (c,))
        self.truncated = self._array("truncated", np.bool_, (c,))
        self.episode_ids = self._array("episode_ids", np.int64, (c,), -1)
        self.step_ids = self._array("step_ids", np.int32, (c,), -1)
        self.global_ids = self._array("global_ids", np.int64, (c,), -1)
        self.candidate_ids = self._array("candidate_ids", np.int64, (c,), -1)
        self.priorities = self._array("priorities", np.float32, (c,), 0)

    def _load_metadata(self) -> None:
        if self._meta_path.exists():
            metadata = json.loads(self._meta_path.read_text(encoding="utf-8"))
            if metadata.get("schema_version") != self.SCHEMA_VERSION:
                raise ValueError("unsupported replay schema")
            expected = {
                "capacity": self.capacity,
                "frame_shape": list(self.frame_shape),
                "feature_dim": self.feature_dim,
                "action_dim": self.action_dim,
                "sequence_length": self.sequence_length,
                "burn_in": self.burn_in,
                "demonstration": self.demonstration,
            }
            mismatches = {
                key: (metadata.get(key), value)
                for key, value in expected.items()
                if metadata.get(key) != value
            }
            if mismatches:
                raise ValueError(f"replay metadata mismatch: {mismatches}")
            self.next_global_id = int(metadata.get("next_global_id", 0))
            self.current_episode_start = int(metadata.get("current_episode_start", 0))
            self.max_priority = float(metadata.get("max_priority", 1.0))
        else:
            self.next_global_id = 0
            self.current_episode_start = 0
            self.max_priority = 1.0
            self.flush()

    def flush(self) -> None:
        for value in vars(self).values():
            if isinstance(value, np.memmap):
                value.flush()
        temporary = self._meta_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(
                {
                    "schema_version": self.SCHEMA_VERSION,
                    "capacity": self.capacity,
                    "frame_shape": self.frame_shape,
                    "feature_dim": self.feature_dim,
                    "action_dim": self.action_dim,
                    "sequence_length": self.sequence_length,
                    "burn_in": self.burn_in,
                    "next_global_id": self.next_global_id,
                    "current_episode_start": self.current_episode_start,
                    "max_priority": self.max_priority,
                    "demonstration": self.demonstration,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        os.replace(temporary, self._meta_path)

    def __len__(self) -> int:
        return min(self.next_global_id, self.capacity)

    @property
    def sequence_count(self) -> int:
        return int(np.count_nonzero(self.priorities > 0))

    def _slot_valid(self, global_id: int) -> bool:
        return self.global_ids[global_id % self.capacity] == global_id

    def _add_candidate(self, start_id: int) -> None:
        if start_id < 0 or not self._slot_valid(start_id):
            return
        slot = start_id % self.capacity
        self.candidate_ids[slot] = start_id
        self.priorities[slot] = self.max_priority

    def add(self, transition: Transition) -> int:
        global_id = self.next_global_id
        slot = global_id % self.capacity
        self.candidate_ids[slot] = -1
        self.priorities[slot] = 0.0
        observation = transition.observation
        self.frames[slot] = observation.frame
        self.features[slot] = observation.features
        self.confidence[slot] = observation.feature_confidence
        self.action_masks[slot] = observation.action_mask
        self.previous_actions[slot] = observation.previous_action.as_array()
        self.previous_rewards[slot] = observation.previous_reward
        self.actions[slot] = transition.action.as_array()
        self.rewards[slot] = transition.reward
        self.terminated[slot] = transition.terminated
        self.truncated[slot] = transition.truncated
        self.episode_ids[slot] = transition.episode_id
        self.step_ids[slot] = transition.step_id
        self.global_ids[slot] = global_id
        self.next_global_id += 1

        complete_start = global_id - self.sequence_length
        if complete_start >= self.current_episode_start:
            self._add_candidate(complete_start)

        if transition.done:
            minimum_start = max(self.current_episode_start, global_id - self.sequence_length + 1)
            maximum_start = global_id - self.burn_in
            for start_id in range(minimum_start, maximum_start + 1):
                self._add_candidate(start_id)
            self.current_episode_start = self.next_global_id
        return global_id

    def _valid_candidate_slots(self) -> np.ndarray:
        slots = np.flatnonzero(self.priorities > 0)
        if slots.size == 0:
            return slots
        ids = np.asarray(self.candidate_ids[slots])
        valid = ids >= 0
        valid &= np.asarray(self.global_ids[ids % self.capacity]) == ids
        invalid_slots = slots[~valid]
        if invalid_slots.size:
            self.priorities[invalid_slots] = 0.0
            self.candidate_ids[invalid_slots] = -1
        return slots[valid]

    def sample(self, batch_size: int, beta: float, rng: np.random.Generator) -> ReplayBatch:
        candidate_slots = self._valid_candidate_slots()
        if candidate_slots.size < batch_size:
            raise RuntimeError(
                f"not enough replay sequences: have {candidate_slots.size}, need {batch_size}"
            )
        raw_priorities = np.asarray(self.priorities[candidate_slots], dtype=np.float64)
        probabilities = np.power(np.maximum(raw_priorities, 1.0e-6), self.alpha)
        probabilities /= probabilities.sum()
        selected_positions = rng.choice(
            candidate_slots.size, size=batch_size, replace=candidate_slots.size < batch_size, p=probabilities
        )
        selected_slots = candidate_slots[selected_positions]
        selected_probabilities = probabilities[selected_positions]
        weights = np.power(candidate_slots.size * selected_probabilities, -beta)
        weights /= max(weights.max(), 1.0e-6)
        start_ids = np.asarray(self.candidate_ids[selected_slots], dtype=np.int64)
        arrays = [self._materialize(start_id) for start_id in start_ids]
        stacked = [np.stack(values) for values in zip(*arrays)]
        return ReplayBatch(
            frames=stacked[0],
            features=stacked[1],
            confidence=stacked[2],
            action_masks=stacked[3],
            previous_actions=stacked[4],
            previous_rewards=stacked[5],
            actions=stacked[6],
            rewards=stacked[7],
            terminated=stacked[8],
            truncated=stacked[9],
            weights=np.asarray(weights, dtype=np.float32),
            start_ids=start_ids,
            demonstrations=np.full(batch_size, self.demonstration, dtype=np.bool_),
        )

    def _materialize(self, start_id: int):
        length = self.sequence_length
        frame = np.empty((length + 1, *self.frame_shape), dtype=np.uint8)
        features = np.empty((length + 1, self.feature_dim), dtype=np.float32)
        confidence = np.empty_like(features)
        masks = np.empty((length + 1, self.action_dim), dtype=np.bool_)
        previous_actions = np.zeros((length + 1, 2), dtype=np.int64)
        previous_rewards = np.zeros(length + 1, dtype=np.float32)
        actions = np.zeros((length, 2), dtype=np.int64)
        rewards = np.zeros(length, dtype=np.float32)
        terminated = np.ones(length, dtype=np.bool_)
        truncated = np.zeros(length, dtype=np.bool_)
        first_slot = start_id % self.capacity
        episode_id = int(self.episode_ids[first_slot])
        last_observation_slot = first_slot
        ended = False
        for offset in range(length):
            global_id = start_id + offset
            slot = global_id % self.capacity
            valid = self._slot_valid(global_id) and int(self.episode_ids[slot]) == episode_id and not ended
            if valid:
                frame[offset] = self.frames[slot]
                features[offset] = self.features[slot]
                confidence[offset] = self.confidence[slot]
                masks[offset] = self.action_masks[slot]
                previous_actions[offset] = self.previous_actions[slot]
                previous_rewards[offset] = self.previous_rewards[slot]
                actions[offset] = self.actions[slot]
                rewards[offset] = self.rewards[slot]
                terminated[offset] = self.terminated[slot]
                truncated[offset] = self.truncated[slot]
                last_observation_slot = slot
                ended = bool(terminated[offset] or truncated[offset])
            else:
                frame[offset] = self.frames[last_observation_slot]
                features[offset] = self.features[last_observation_slot]
                confidence[offset] = self.confidence[last_observation_slot]
                masks[offset] = self.action_masks[last_observation_slot]
                previous_actions[offset] = self.previous_actions[last_observation_slot]
                previous_rewards[offset] = self.previous_rewards[last_observation_slot]
        next_id = start_id + length
        next_slot = next_id % self.capacity
        if not ended and self._slot_valid(next_id) and int(self.episode_ids[next_slot]) == episode_id:
            last_observation_slot = next_slot
        frame[length] = self.frames[last_observation_slot]
        features[length] = self.features[last_observation_slot]
        confidence[length] = self.confidence[last_observation_slot]
        masks[length] = self.action_masks[last_observation_slot]
        previous_actions[length] = self.previous_actions[last_observation_slot]
        previous_rewards[length] = self.previous_rewards[last_observation_slot]
        return (
            frame,
            features,
            confidence,
            masks,
            previous_actions,
            previous_rewards,
            actions,
            rewards,
            terminated,
            truncated,
        )

    def update_priorities(self, start_ids: np.ndarray, priorities: np.ndarray) -> None:
        for start_id, priority in zip(start_ids, priorities):
            slot = int(start_id) % self.capacity
            if self.candidate_ids[slot] == start_id:
                value = float(max(priority, 1.0e-6))
                self.priorities[slot] = value
                self.max_priority = max(self.max_priority, value)


def concatenate_batches(batches: list[ReplayBatch]) -> ReplayBatch:
    if not batches:
        raise ValueError("at least one replay batch is required")
    return ReplayBatch(
        **{
            field: np.concatenate([getattr(batch, field) for batch in batches], axis=0)
            for field in ReplayBatch.__dataclass_fields__
        }
    )
