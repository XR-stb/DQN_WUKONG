from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from .types import (
    ACTION_MASK_SIZE,
    COMBAT_MASK_SLICE,
    ActionCommand,
    ActionToken,
    CombatToken,
    EpisodeState,
    MovementToken,
    Observation,
    Transition,
    canonicalize_command,
)


DATASET_SCHEMA_VERSION = 3
SUPPORTED_DATASET_SCHEMA_VERSIONS = {2, DATASET_SCHEMA_VERSION}


def _movement_from_keys(keys: Iterable[str]) -> MovementToken:
    values = {str(key).lower() for key in keys}
    forward = "w" in values and "s" not in values
    back = "s" in values and "w" not in values
    left = "a" in values and "d" not in values
    right = "d" in values and "a" not in values
    if forward and left:
        return MovementToken.FORWARD_LEFT
    if forward and right:
        return MovementToken.FORWARD_RIGHT
    if back and left:
        return MovementToken.BACK_LEFT
    if back and right:
        return MovementToken.BACK_RIGHT
    if forward:
        return MovementToken.FORWARD
    if back:
        return MovementToken.BACK
    if left:
        return MovementToken.LEFT
    if right:
        return MovementToken.RIGHT
    return MovementToken.NONE


_LEGACY_COMBAT_MASK_MAP = {
    CombatToken.LIGHT_ATTACK: ActionToken.LIGHT_ATTACK,
    CombatToken.HEAVY_HOLD: ActionToken.HEAVY_HOLD,
    CombatToken.DODGE: ActionToken.DODGE,
    CombatToken.SKILL_1: ActionToken.SKILL_1,
    CombatToken.SKILL_2: ActionToken.SKILL_2,
    CombatToken.SKILL_3: ActionToken.SKILL_3,
    CombatToken.SKILL_4: ActionToken.SKILL_4,
    CombatToken.FABAO: ActionToken.FABAO,
    CombatToken.TISHEN: ActionToken.TISHEN,
    CombatToken.DRINK_POTION: ActionToken.DRINK_POTION,
}


def _branch_mask_from_legacy(mask: np.ndarray) -> np.ndarray:
    result = np.ones(ACTION_MASK_SIZE, dtype=np.bool_)
    combat = result[COMBAT_MASK_SLICE]
    for branch_action, legacy_action in _LEGACY_COMBAT_MASK_MAP.items():
        combat[int(branch_action)] = bool(mask[int(legacy_action)])
    combat[int(CombatToken.NONE)] = True
    return result


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


@dataclass(slots=True, frozen=True)
class DatasetRepairResult:
    source: str
    output: str
    source_version: str
    output_version: str
    episodes: int
    transitions: int
    q_press_edges: int
    repaired_inputs: int
    executable_potion_actions: int
    masked_potion_actions: int
    hardlinked_frame_files: int
    copied_frame_files: int


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
    previous_actions = np.stack([obs.previous_action.as_array() for obs in observations]).astype(
        np.int16, copy=False
    )
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
        actions=np.stack([item.action.as_array() for item in items]).astype(np.int16, copy=False),
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
        if self.manifest.schema_version not in SUPPORTED_DATASET_SCHEMA_VERSIONS:
            raise ValueError(
                f"dataset schema {self.manifest.schema_version} is unsupported; "
                f"expected one of {sorted(SUPPORTED_DATASET_SCHEMA_VERSIONS)}"
            )
        self.frames = np.load(
            self.directory / "frames.npy", mmap_mode="r" if memory_map else None, allow_pickle=False
        )
        self.trajectory = np.load(self.directory / "trajectory.npz", allow_pickle=False)
        self._branched_cache: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
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
        if self.manifest.schema_version == 2:
            if actions.shape != (count,) or np.any(actions < 0) or np.any(actions >= ActionToken.size()):
                raise ValueError(f"{self.directory}: invalid legacy action token")
        elif (
            actions.shape != (count, 2)
            or np.any(actions[:, 0] < 0)
            or np.any(actions[:, 0] >= MovementToken.size())
            or np.any(actions[:, 1] < 0)
            or np.any(actions[:, 1] >= CombatToken.size())
        ):
            raise ValueError(f"{self.directory}: invalid branched action command")
        previous_actions = self.trajectory["previous_actions"]
        expected_previous_shape = (count + 1,) if self.manifest.schema_version == 2 else (count + 1, 2)
        if previous_actions.shape != expected_previous_shape:
            raise ValueError(f"{self.directory}: invalid previous action shape")
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
        expected_mask_size = ActionToken.size() if self.manifest.schema_version == 2 else ACTION_MASK_SIZE
        if masks.dtype != np.bool_ or masks.shape != (count + 1, expected_mask_size):
            raise ValueError(f"{self.directory}: invalid action mask shape")
        if self.manifest.schema_version == 2 and not masks[:, int(ActionToken.IDLE)].all():
            raise ValueError(f"{self.directory}: every legacy action mask must allow IDLE")
        if self.manifest.schema_version == 3 and (
            not masks[:, int(MovementToken.NONE)].all()
            or not masks[:, MovementToken.size() + int(CombatToken.NONE)].all()
        ):
            raise ValueError(f"{self.directory}: every action branch must allow NONE")
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

    def branched_actions(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return commands, observation masks, and previous commands for v2/v3 data."""

        if self._branched_cache is not None:
            return self._branched_cache
        count = len(self)
        if self.manifest.schema_version == 3:
            stored_commands = np.asarray(self.trajectory["actions"], dtype=np.int16)
            masks = np.asarray(self.trajectory["action_masks"], dtype=np.bool_)
            commands = np.stack(
                [
                    canonicalize_command(ActionCommand.from_array(command), masks[index]).as_array()
                    for index, command in enumerate(stored_commands)
                ]
            ).astype(np.int16, copy=False)
            stored_previous = np.asarray(self.trajectory["previous_actions"], dtype=np.int16)
            previous = np.zeros((count + 1, 2), dtype=np.int16)
            previous[0] = canonicalize_command(
                ActionCommand.from_array(stored_previous[0]), masks[0]
            ).as_array()
            previous[1:] = commands
        else:
            legacy_actions = np.asarray(self.trajectory["actions"], dtype=np.int64)
            raw_inputs = self.trajectory["raw_inputs"]
            masks = np.stack(
                [_branch_mask_from_legacy(mask) for mask in self.trajectory["action_masks"]]
            )
            commands = np.zeros((count, 2), dtype=np.int16)
            for index, legacy_value in enumerate(legacy_actions):
                payload = {}
                if str(raw_inputs[index]):
                    try:
                        payload = json.loads(str(raw_inputs[index]))
                    except json.JSONDecodeError:
                        payload = {}
                movement = _movement_from_keys(payload.get("keys", ()))
                combat = ActionCommand.from_legacy(int(legacy_value)).combat
                command = canonicalize_command(ActionCommand(movement, combat), masks[index])
                commands[index] = command.as_array()
            previous = np.zeros((count + 1, 2), dtype=np.int16)
            previous[1:] = commands
        self._branched_cache = commands, masks, previous
        return self._branched_cache

    def close(self) -> None:
        self._branched_cache = None
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
            # Labels and masks live in trajectory.npz. Hashing only manifests
            # made two datasets with different supervision appear identical.
            trajectory_path = path.parent / "trajectory.npz"
            with trajectory_path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            # Frames are immutable after recording and expensive to hash on
            # every training startup. Their relative path and byte size still
            # catch missing/truncated files without scanning hundreds of MB.
            frames_path = path.parent / "frames.npy"
            digest.update(str(frames_path.stat().st_size).encode("ascii"))
        self.version = digest.hexdigest()[:16]
        self.total_transitions = sum(item.transitions for item in manifests)

    def episodes(self, memory_map: bool = True) -> Iterator[TrajectoryEpisode]:
        for manifest_path in self.manifest_paths:
            yield TrajectoryEpisode(manifest_path.parent, memory_map=memory_map)

    def split(self, validation_fraction: float = 0.2) -> tuple[list[Path], list[Path]]:
        if not 0.0 < validation_fraction < 1.0:
            raise ValueError("validation_fraction must be within (0, 1)")
        if len(self.manifest_paths) == 1:
            return self.manifest_paths, self.manifest_paths
        manifests = {
            path: EpisodeManifest.from_path(path) for path in self.manifest_paths
        }
        validation_count = min(
            len(self.manifest_paths) - 1,
            max(2, int(round(len(self.manifest_paths) * validation_fraction))),
        )
        selected: list[Path] = []

        # Reserve representative wins and losses first. A chronological tail
        # split previously put 93% of validation windows in one long failed
        # episode, making aggregate accuracy a misleading release signal.
        for result in (EpisodeState.WON.value, EpisodeState.LOST.value):
            group = [
                path for path in self.manifest_paths if manifests[path].result == result
            ]
            if len(group) < 2 or len(selected) >= validation_count:
                continue
            median_length = float(np.median([manifests[path].transitions for path in group]))
            selected.append(
                min(
                    group,
                    key=lambda path: (
                        abs(manifests[path].transitions - median_length),
                        path.as_posix(),
                    ),
                )
            )

        target_transitions = max(
            1, int(round(self.total_transitions * validation_fraction))
        )
        while len(selected) < validation_count:
            selected_transitions = sum(manifests[path].transitions for path in selected)
            candidates = [path for path in self.manifest_paths if path not in selected]
            selected.append(
                min(
                    candidates,
                    key=lambda path: (
                        abs(
                            target_transitions
                            - selected_transitions
                            - manifests[path].transitions
                        ),
                        path.as_posix(),
                    ),
                )
            )

        validation_set = set(selected)
        training = [path for path in self.manifest_paths if path not in validation_set]
        validation = [path for path in self.manifest_paths if path in validation_set]
        return training, validation


def repair_legacy_potion_inputs(
    source: str | Path,
    output: str | Path,
    *,
    boss_id: str = "yinhu",
) -> DatasetRepairResult:
    """Create a repaired copy of demonstrations recorded before Q was mapped.

    The old recorder still preserved held keys in ``raw_inputs``. A rising Q
    edge can therefore be restored without guessing from pixels. The action is
    changed to DRINK_POTION only when its recorded mask allowed it; otherwise
    it becomes IDLE, matching the current recorder's execution semantics.
    """

    source_root = Path(source).resolve()
    output_root = Path(output).resolve()
    if source_root == output_root:
        raise ValueError("repair output must be different from the source dataset")
    if source_root in output_root.parents or output_root in source_root.parents:
        raise ValueError("repair source and output cannot contain one another")
    if output_root.exists():
        raise FileExistsError(output_root)

    dataset = TrajectoryDataset(source_root, boss_id=boss_id)
    temporary_root = output_root.with_name(f".{output_root.name}.tmp-{uuid.uuid4().hex[:8]}")
    temporary_root.mkdir(parents=True)
    q_press_edges = 0
    repaired_inputs = 0
    executable_potion_actions = 0
    masked_potion_actions = 0
    hardlinked_frame_files = 0
    copied_frame_files = 0
    try:
        for manifest_path in dataset.manifest_paths:
            episode = TrajectoryEpisode(manifest_path.parent, memory_map=True)
            relative_episode = manifest_path.parent.relative_to(source_root)
            target_episode = temporary_root / relative_episode
            target_episode.mkdir(parents=True)
            try:
                source_frames = episode.directory / "frames.npy"
                target_frames = target_episode / "frames.npy"
                try:
                    os.link(source_frames, target_frames)
                    hardlinked_frame_files += 1
                except OSError:
                    shutil.copy2(source_frames, target_frames)
                    copied_frame_files += 1

                arrays = {
                    key: np.array(episode.trajectory[key], copy=True)
                    for key in episode.trajectory.files
                    if key != "raw_inputs"
                }
                actions = arrays["actions"]
                action_masks = arrays["action_masks"]
                previous_actions = arrays["previous_actions"]
                raw_inputs = [str(value) for value in episode.trajectory["raw_inputs"]]
                previous_q_down = False
                for index, raw_input in enumerate(raw_inputs):
                    payload = json.loads(raw_input) if raw_input else {}
                    keys = {str(key).lower() for key in payload.get("keys", ())}
                    q_down = "q" in keys
                    q_rising = index > 0 and q_down and not previous_q_down
                    if q_rising:
                        q_press_edges += 1
                    if q_rising and payload.get("token") != ActionToken.DRINK_POTION.name:
                        if episode.manifest.schema_version == 2:
                            allowed = bool(action_masks[index, int(ActionToken.DRINK_POTION)])
                            repaired_action = (
                                ActionToken.DRINK_POTION if allowed else ActionToken.IDLE
                            )
                            actions[index] = int(repaired_action)
                            if index + 1 < previous_actions.shape[0]:
                                previous_actions[index + 1] = int(repaired_action)
                        else:
                            mask_index = MovementToken.size() + int(CombatToken.DRINK_POTION)
                            allowed = bool(action_masks[index, mask_index])
                            repaired_combat = (
                                CombatToken.DRINK_POTION if allowed else CombatToken.NONE
                            )
                            actions[index, 1] = int(repaired_combat)
                            if index + 1 < previous_actions.shape[0]:
                                previous_actions[index + 1, 1] = int(repaired_combat)
                        payload["token"] = ActionToken.DRINK_POTION.name
                        raw_inputs[index] = json.dumps(
                            payload, ensure_ascii=False, separators=(",", ":")
                        )
                        repaired_inputs += 1
                        if allowed:
                            executable_potion_actions += 1
                        else:
                            masked_potion_actions += 1
                    previous_q_down = q_down

                arrays["raw_inputs"] = np.asarray(raw_inputs, dtype=np.str_)
                np.savez(target_episode / "trajectory.npz", **arrays)
                shutil.copy2(manifest_path, target_episode / "manifest.json")
            finally:
                episode.close()

        repair_manifest = {
            "schema_version": 1,
            "transformation": "legacy_q_to_potion_rising_edges",
            "created_at": time.time(),
            "source": str(source_root),
            "source_version": dataset.version,
            "boss_id": boss_id,
            "episodes": len(dataset.manifest_paths),
            "transitions": dataset.total_transitions,
            "q_press_edges": q_press_edges,
            "repaired_inputs": repaired_inputs,
            "executable_potion_actions": executable_potion_actions,
            "masked_potion_actions": masked_potion_actions,
            "hardlinked_frame_files": hardlinked_frame_files,
            "copied_frame_files": copied_frame_files,
        }
        (temporary_root / "repair_manifest.json").write_text(
            json.dumps(repair_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(temporary_root, output_root)
    except BaseException:
        if temporary_root.exists():
            shutil.rmtree(temporary_root)
        raise

    repaired_dataset = TrajectoryDataset(output_root, boss_id=boss_id)
    return DatasetRepairResult(
        source=str(source_root),
        output=str(output_root),
        source_version=dataset.version,
        output_version=repaired_dataset.version,
        episodes=len(dataset.manifest_paths),
        transitions=dataset.total_transitions,
        q_press_edges=q_press_edges,
        repaired_inputs=repaired_inputs,
        executable_potion_actions=executable_potion_actions,
        masked_potion_actions=masked_potion_actions,
        hardlinked_frame_files=hardlinked_frame_files,
        copied_frame_files=copied_frame_files,
    )


def observation_from_episode(episode: TrajectoryEpisode, index: int) -> Observation:
    trajectory = episode.trajectory
    _, masks, previous_commands = episode.branched_actions()
    result = episode.manifest.result if index == len(episode) else EpisodeState.FIGHTING.value
    try:
        episode_state = EpisodeState(result)
    except ValueError:
        episode_state = EpisodeState.FIGHTING
    return Observation(
        frame=np.asarray(episode.frames[index]),
        features=np.asarray(trajectory["features"][index]),
        feature_confidence=np.asarray(trajectory["confidence"][index]),
        action_mask=np.asarray(masks[index]),
        timestamp=float(trajectory["timestamps"][index]),
        episode_state=episode_state,
        previous_action=ActionCommand.from_array(previous_commands[index]),
        previous_reward=float(trajectory["previous_rewards"][index]),
    )


def effective_action_from_episode(episode: TrajectoryEpisode, index: int) -> ActionCommand:
    """Map v2/v3 recorded human input to the executable branched command."""
    commands, masks, _ = episode.branched_actions()
    return canonicalize_command(ActionCommand.from_array(commands[index]), masks[index])


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
