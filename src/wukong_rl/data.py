from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from .types import ActionToken, EpisodeState, Observation, Transition


DATASET_SCHEMA_VERSION = 2


@dataclass(slots=True, frozen=True)
class EpisodeManifest:
    schema_version: int
    episode_id: str
    boss_id: str
    created_at: float
    config_hash: str
    transitions: int
    frame_shape: tuple[int, int, int]
    feature_dim: int
    result: str

    @classmethod
    def from_path(cls, path: Path) -> "EpisodeManifest":
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["frame_shape"] = tuple(payload["frame_shape"])
        return cls(**payload)


def save_episode(
    root: str | Path,
    boss_id: str,
    transitions: Iterable[Transition],
    config_hash: str,
    episode_id: str | None = None,
) -> Path:
    items = list(transitions)
    if not items:
        raise ValueError("cannot save an empty episode")
    episode_id = episode_id or f"{int(time.time())}-{uuid.uuid4().hex[:8]}"
    root = Path(root)
    episode_dir = root / boss_id / episode_id
    if episode_dir.exists():
        raise FileExistsError(episode_dir)
    episode_dir.mkdir(parents=True)

    observations = [item.observation for item in items] + [items[-1].next_observation]
    frames = np.stack([obs.frame for obs in observations]).astype(np.uint8, copy=False)
    features = np.stack([obs.features for obs in observations]).astype(np.float32, copy=False)
    confidence = np.stack([obs.feature_confidence for obs in observations]).astype(np.float32, copy=False)
    masks = np.stack([obs.action_mask for obs in observations]).astype(np.bool_, copy=False)
    previous_actions = np.asarray([int(obs.previous_action) for obs in observations], dtype=np.int16)
    previous_rewards = np.asarray([obs.previous_reward for obs in observations], dtype=np.float32)
    timestamps = np.asarray([obs.timestamp for obs in observations], dtype=np.float64)

    np.save(episode_dir / "frames.npy", frames, allow_pickle=False)
    np.savez(
        episode_dir / "trajectory.npz",
        features=features,
        confidence=confidence,
        action_masks=masks,
        previous_actions=previous_actions,
        previous_rewards=previous_rewards,
        timestamps=timestamps,
        actions=np.asarray([int(item.action) for item in items], dtype=np.int16),
        rewards=np.asarray([item.reward for item in items], dtype=np.float32),
        terminated=np.asarray([item.terminated for item in items], dtype=np.bool_),
        truncated=np.asarray([item.truncated for item in items], dtype=np.bool_),
        raw_inputs=np.asarray([item.raw_input for item in items], dtype=np.str_),
    )
    last_state = items[-1].next_observation.episode_state
    manifest = EpisodeManifest(
        schema_version=DATASET_SCHEMA_VERSION,
        episode_id=episode_id,
        boss_id=boss_id,
        created_at=time.time(),
        config_hash=config_hash,
        transitions=len(items),
        frame_shape=tuple(int(value) for value in frames.shape[1:]),
        feature_dim=int(features.shape[1]),
        result=last_state.value,
    )
    temporary = episode_dir / "manifest.json.tmp"
    temporary.write_text(
        json.dumps(asdict(manifest), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, episode_dir / "manifest.json")
    return episode_dir


class TrajectoryEpisode:
    def __init__(self, directory: str | Path, memory_map: bool = True) -> None:
        self.directory = Path(directory)
        self.manifest = EpisodeManifest.from_path(self.directory / "manifest.json")
        if self.manifest.schema_version != DATASET_SCHEMA_VERSION:
            raise ValueError(
                f"dataset schema {self.manifest.schema_version} is unsupported; "
                f"expected {DATASET_SCHEMA_VERSION}"
            )
        self.frames = np.load(
            self.directory / "frames.npy", mmap_mode="r" if memory_map else None, allow_pickle=False
        )
        self.trajectory = np.load(self.directory / "trajectory.npz", allow_pickle=False)
        self.validate()

    def validate(self) -> None:
        count = self.manifest.transitions
        if self.frames.shape[0] != count + 1:
            raise ValueError(f"{self.directory}: frame count does not match manifest")
        if tuple(self.frames.shape[1:]) != self.manifest.frame_shape:
            raise ValueError(f"{self.directory}: frame shape does not match manifest")
        for key in (
            "features",
            "confidence",
            "action_masks",
            "previous_actions",
            "previous_rewards",
            "timestamps",
        ):
            if self.trajectory[key].shape[0] != count + 1:
                raise ValueError(f"{self.directory}: {key} observation count mismatch")
        for key in ("actions", "rewards", "terminated", "truncated", "raw_inputs"):
            if self.trajectory[key].shape[0] != count:
                raise ValueError(f"{self.directory}: {key} transition count mismatch")
        actions = self.trajectory["actions"]
        if np.any(actions < 0) or np.any(actions >= ActionToken.size()):
            raise ValueError(f"{self.directory}: invalid action token")
        if self.frames.dtype != np.uint8 or self.frames.ndim != 4 or self.frames.shape[-1] != 3:
            raise ValueError(f"{self.directory}: frames must be HxWx3 uint8")
        if not np.isfinite(self.trajectory["features"]).all():
            raise ValueError(f"{self.directory}: non-finite HUD feature")
        confidence = self.trajectory["confidence"]
        if not np.isfinite(confidence).all() or np.any((confidence < 0) | (confidence > 1)):
            raise ValueError(f"{self.directory}: invalid detection confidence")
        if not np.isfinite(self.trajectory["rewards"]).all():
            raise ValueError(f"{self.directory}: non-finite reward")
        timestamps = self.trajectory["timestamps"]
        if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) < 0):
            raise ValueError(f"{self.directory}: timestamps must be finite and monotonic")
        masks = self.trajectory["action_masks"]
        if masks.dtype != np.bool_ or not masks[:, int(ActionToken.IDLE)].all():
            raise ValueError(f"{self.directory}: every action mask must allow IDLE")
        terminated = self.trajectory["terminated"]
        truncated = self.trajectory["truncated"]
        if np.any(terminated & truncated):
            raise ValueError(f"{self.directory}: a transition cannot be terminated and truncated")
        done = terminated | truncated
        if not done[-1] or np.any(done[:-1]):
            raise ValueError(f"{self.directory}: episode boundary must occur only at the final transition")
        for raw_input in self.trajectory["raw_inputs"]:
            if str(raw_input):
                try:
                    json.loads(str(raw_input))
                except json.JSONDecodeError as error:
                    raise ValueError(f"{self.directory}: malformed raw input JSON") from error

    def __len__(self) -> int:
        return self.manifest.transitions

    def close(self) -> None:
        self.trajectory.close()


class TrajectoryDataset:
    def __init__(self, root: str | Path, boss_id: str | None = None) -> None:
        root = Path(root)
        boss_root = root / boss_id if boss_id else root
        search_root = boss_root if boss_id and boss_root.exists() else root
        self.manifest_paths = sorted(search_root.glob("**/manifest.json"))
        if not self.manifest_paths:
            raise FileNotFoundError(f"no trajectory manifests found under {search_root}")
        manifests = [EpisodeManifest.from_path(path) for path in self.manifest_paths]
        if boss_id:
            mismatched = [item.episode_id for item in manifests if item.boss_id != boss_id]
            if mismatched:
                raise ValueError(f"dataset contains episodes for a different boss: {mismatched[:3]}")
        digest = hashlib.sha256()
        for path in self.manifest_paths:
            digest.update(path.relative_to(search_root).as_posix().encode("utf-8"))
            digest.update(path.read_bytes())
        self.version = digest.hexdigest()[:16]
        self.total_transitions = sum(item.transitions for item in manifests)

    def episodes(self, memory_map: bool = True) -> Iterator[TrajectoryEpisode]:
        for manifest_path in self.manifest_paths:
            yield TrajectoryEpisode(manifest_path.parent, memory_map=memory_map)

    def split(self, validation_fraction: float = 0.1) -> tuple[list[Path], list[Path]]:
        if not 0.0 < validation_fraction < 1.0:
            raise ValueError("validation_fraction must be within (0, 1)")
        validation_count = max(1, int(round(len(self.manifest_paths) * validation_fraction)))
        if len(self.manifest_paths) == 1:
            return self.manifest_paths, self.manifest_paths
        return self.manifest_paths[:-validation_count], self.manifest_paths[-validation_count:]


def observation_from_episode(episode: TrajectoryEpisode, index: int) -> Observation:
    trajectory = episode.trajectory
    result = episode.manifest.result if index == len(episode) else EpisodeState.FIGHTING.value
    try:
        episode_state = EpisodeState(result)
    except ValueError:
        episode_state = EpisodeState.FIGHTING
    previous_action = (
        effective_action_from_episode(episode, index - 1)
        if index > 0
        else ActionToken.IDLE
    )
    return Observation(
        frame=np.asarray(episode.frames[index]),
        features=np.asarray(trajectory["features"][index]),
        feature_confidence=np.asarray(trajectory["confidence"][index]),
        action_mask=np.asarray(trajectory["action_masks"][index]),
        timestamp=float(trajectory["timestamps"][index]),
        episode_state=episode_state,
        previous_action=previous_action,
        previous_reward=float(trajectory["previous_rewards"][index]),
    )


def effective_action_from_episode(episode: TrajectoryEpisode, index: int) -> ActionToken:
    """Map recorded human intent to the executable policy action."""
    action = ActionToken(int(episode.trajectory["actions"][index]))
    mask = episode.trajectory["action_masks"][index]
    return action if bool(mask[int(action)]) else ActionToken.IDLE


def transitions_from_episode(
    episode: TrajectoryEpisode, episode_id: int, demonstration: bool = True
) -> Iterator[Transition]:
    trajectory = episode.trajectory
    for index in range(len(episode)):
        yield Transition(
            observation=observation_from_episode(episode, index),
            action=effective_action_from_episode(episode, index),
            reward=float(trajectory["rewards"][index]),
            next_observation=observation_from_episode(episode, index + 1),
            terminated=bool(trajectory["terminated"][index]),
            truncated=bool(trajectory["truncated"][index]),
            timestamp=float(trajectory["timestamps"][index + 1]),
            episode_id=episode_id,
            step_id=index,
            demonstration=demonstration,
            raw_input=str(trajectory["raw_inputs"][index]),
        )
