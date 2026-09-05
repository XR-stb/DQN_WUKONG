from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

from .agent import R2D3Agent
from .data import TrajectoryEpisode
from .types import ActionToken


CORE_POLICY_ACTIONS = (
    ActionToken.IDLE,
    ActionToken.RUN_FORWARD,
    ActionToken.RUN_BACK,
    ActionToken.RUN_LEFT,
    ActionToken.RUN_RIGHT,
    ActionToken.LIGHT_ATTACK,
    ActionToken.HEAVY_HOLD,
    ActionToken.DODGE,
)


@dataclass(slots=True)
class BcMetrics:
    loss: float
    accuracy: float
    class_recall: dict[int, float]
    confusion: np.ndarray


def core_balanced_score(metrics: BcMetrics) -> float:
    """Prefer policies that cover every observed core action without collapsing accuracy."""
    rows = metrics.confusion.sum(axis=1)
    indices = np.asarray(
        [int(action) for action in CORE_POLICY_ACTIONS if rows[int(action)] > 0],
        dtype=np.int64,
    )
    if not indices.size:
        return 0.0
    recalls = np.asarray([metrics.class_recall[int(index)] for index in indices], dtype=np.float64)
    # The harmonic mean heavily penalizes a zero-recall action (for example,
    # never dodging), while the accuracy factor prevents a uniformly poor
    # classifier from winning only because its recalls happen to be balanced.
    harmonic_recall = float(indices.size / np.sum(1.0 / np.maximum(recalls, 1.0e-6)))
    return float(metrics.accuracy * harmonic_recall)


class BehaviorCloningTrainer:
    def __init__(self, agent: R2D3Agent, sequence_length: int = 32, seed: int = 7) -> None:
        self.agent = agent
        self.sequence_length = sequence_length
        self.rng = np.random.default_rng(seed)

    def _load_episodes(self, paths: list[Path]) -> list[TrajectoryEpisode]:
        return [TrajectoryEpisode(path.parent, memory_map=True) for path in paths]

    @staticmethod
    def _effective_actions(episode: TrajectoryEpisode) -> np.ndarray:
        raw_actions = np.asarray(episode.trajectory["actions"], dtype=np.int64)
        masks = np.asarray(episode.trajectory["action_masks"][:-1], dtype=np.bool_)
        valid = np.take_along_axis(masks, raw_actions[:, None], axis=1).squeeze(1)
        return np.where(valid, raw_actions, int(ActionToken.IDLE))

    def estimate_class_weights(self, paths: list[Path]) -> np.ndarray:
        episodes = self._load_episodes(paths)
        try:
            counts = np.zeros(self.agent.action_dim, dtype=np.int64)
            for episode in episodes:
                counts += np.bincount(
                    self._effective_actions(episode), minlength=self.agent.action_dim
                )
        finally:
            for episode in episodes:
                episode.close()
        # Keep unseen classes neutral so validation batches cannot become a
        # zero-weight/NaN loss. Present rare classes receive bounded emphasis.
        weights = np.ones(self.agent.action_dim, dtype=np.float32)
        present = counts > 0
        if present.any():
            maximum = float(counts[present].max())
            weights[present] = np.minimum(
                np.sqrt(maximum / counts[present]), 4.0
            ).astype(np.float32)
        return weights

    def _sample_batch(self, episodes: list[TrajectoryEpisode], batch_size: int):
        length = self.sequence_length
        selections: list[tuple[TrajectoryEpisode, int]] = []
        eligible = [episode for episode in episodes if len(episode) >= length]
        if not eligible:
            raise ValueError(f"no demonstration episode contains {length} transitions")
        window_counts = np.asarray(
            [len(episode) - length + 1 for episode in eligible], dtype=np.float64
        )
        episode_probabilities = window_counts / window_counts.sum()
        for _ in range(batch_size):
            episode = eligible[int(self.rng.choice(len(eligible), p=episode_probabilities))]
            start = int(self.rng.integers(0, len(episode) - length + 1))
            selections.append((episode, start))
        frames = np.stack([np.asarray(ep.frames[start : start + length]) for ep, start in selections])
        features = np.stack(
            [ep.trajectory["features"][start : start + length] for ep, start in selections]
        )
        confidence = np.stack(
            [ep.trajectory["confidence"][start : start + length] for ep, start in selections]
        )
        action_masks = np.stack(
            [ep.trajectory["action_masks"][start : start + length] for ep, start in selections]
        )
        previous_rewards = np.stack(
            [ep.trajectory["previous_rewards"][start : start + length] for ep, start in selections]
        )
        effective_sequences: list[np.ndarray] = []
        previous_sequences: list[np.ndarray] = []
        for episode, start in selections:
            effective = self._effective_actions(episode)
            effective_sequences.append(effective[start : start + length])
            previous = np.empty(length, dtype=np.int64)
            previous[0] = effective[start - 1] if start else int(ActionToken.IDLE)
            previous[1:] = effective[start : start + length - 1]
            previous_sequences.append(previous)
        previous_actions = np.stack(previous_sequences)
        actions = np.stack(effective_sequences)
        return (
            frames,
            features,
            confidence,
            action_masks,
            previous_actions,
            previous_rewards,
            actions,
        )

    def _step(
        self,
        episodes: list[TrajectoryEpisode],
        batch_size: int,
        train: bool,
        class_weights: np.ndarray | None = None,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        values = self._sample_batch(episodes, batch_size)
        frames, features, confidence, action_masks, previous_actions, previous_rewards, actions = [
            torch.from_numpy(np.asarray(value)).to(self.agent.device) for value in values
        ]
        self.agent.online.train(train)
        with torch.set_grad_enabled(train):
            q_values, _ = self.agent.online(
                frames,
                features.float(),
                confidence.float(),
                previous_actions.long(),
                previous_rewards.float(),
            )
            masked_q = q_values.masked_fill(
                ~action_masks.bool(), torch.finfo(q_values.dtype).min
            )
            weight = (
                torch.from_numpy(class_weights).to(self.agent.device)
                if class_weights is not None
                else None
            )
            loss = functional.cross_entropy(
                masked_q.flatten(0, 1), actions.long().flatten(), weight=weight
            )
        if train:
            self.agent.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.agent.online.parameters(), self.agent.config.gradient_clip
            )
            self.agent.optimizer.step()
        predictions = masked_q.argmax(dim=-1).detach().cpu().numpy().ravel()
        targets = actions.detach().cpu().numpy().ravel()
        return float(loss.detach().cpu()), predictions, targets

    def run_epoch(
        self,
        paths: list[Path],
        *,
        batch_size: int,
        steps: int,
        train: bool,
        class_weights: np.ndarray | None = None,
    ) -> BcMetrics:
        episodes = self._load_episodes(paths)
        losses: list[float] = []
        predictions: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        try:
            for _ in range(steps):
                loss, predicted, target = self._step(
                    episodes, batch_size, train, class_weights
                )
                losses.append(loss)
                predictions.append(predicted)
                targets.append(target)
        finally:
            for episode in episodes:
                episode.close()
        predicted = np.concatenate(predictions)
        target = np.concatenate(targets)
        confusion = np.zeros((self.agent.action_dim, self.agent.action_dim), dtype=np.int64)
        np.add.at(confusion, (target, predicted), 1)
        recall = {
            index: float(confusion[index, index] / max(confusion[index].sum(), 1))
            for index in range(self.agent.action_dim)
        }
        return BcMetrics(
            loss=float(np.mean(losses)),
            accuracy=float((predicted == target).mean()),
            class_recall=recall,
            confusion=confusion,
        )
