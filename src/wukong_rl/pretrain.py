from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

from .agent import R2D3Agent
from .data import TrajectoryEpisode


@dataclass(slots=True)
class BcMetrics:
    loss: float
    accuracy: float
    class_recall: dict[int, float]
    confusion: np.ndarray


class BehaviorCloningTrainer:
    def __init__(self, agent: R2D3Agent, sequence_length: int = 32, seed: int = 7) -> None:
        self.agent = agent
        self.sequence_length = sequence_length
        self.rng = np.random.default_rng(seed)

    def _load_episodes(self, paths: list[Path]) -> list[TrajectoryEpisode]:
        return [TrajectoryEpisode(path.parent, memory_map=True) for path in paths]

    def _sample_batch(self, episodes: list[TrajectoryEpisode], batch_size: int):
        length = self.sequence_length
        selections: list[tuple[TrajectoryEpisode, int]] = []
        eligible = [episode for episode in episodes if len(episode) >= length]
        if not eligible:
            raise ValueError(f"no demonstration episode contains {length} transitions")
        for _ in range(batch_size):
            episode = eligible[int(self.rng.integers(0, len(eligible)))]
            start = int(self.rng.integers(0, len(episode) - length + 1))
            selections.append((episode, start))
        frames = np.stack([np.asarray(ep.frames[start : start + length]) for ep, start in selections])
        features = np.stack(
            [ep.trajectory["features"][start : start + length] for ep, start in selections]
        )
        confidence = np.stack(
            [ep.trajectory["confidence"][start : start + length] for ep, start in selections]
        )
        previous_actions = np.stack(
            [ep.trajectory["previous_actions"][start : start + length] for ep, start in selections]
        )
        previous_rewards = np.stack(
            [ep.trajectory["previous_rewards"][start : start + length] for ep, start in selections]
        )
        actions = np.stack(
            [ep.trajectory["actions"][start : start + length] for ep, start in selections]
        )
        return frames, features, confidence, previous_actions, previous_rewards, actions

    def _step(self, episodes: list[TrajectoryEpisode], batch_size: int, train: bool) -> tuple[float, np.ndarray, np.ndarray]:
        values = self._sample_batch(episodes, batch_size)
        frames, features, confidence, previous_actions, previous_rewards, actions = [
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
            loss = functional.cross_entropy(q_values.flatten(0, 1), actions.long().flatten())
        if train:
            self.agent.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.agent.online.parameters(), self.agent.config.gradient_clip
            )
            self.agent.optimizer.step()
        predictions = q_values.argmax(dim=-1).detach().cpu().numpy().ravel()
        targets = actions.detach().cpu().numpy().ravel()
        return float(loss.detach().cpu()), predictions, targets

    def run_epoch(
        self,
        paths: list[Path],
        *,
        batch_size: int,
        steps: int,
        train: bool,
    ) -> BcMetrics:
        episodes = self._load_episodes(paths)
        losses: list[float] = []
        predictions: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        try:
            for _ in range(steps):
                loss, predicted, target = self._step(episodes, batch_size, train)
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
