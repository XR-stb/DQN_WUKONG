from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

from .agent import R2D3Agent
from .data import TrajectoryEpisode
from .types import (
    COMBAT_MASK_SLICE,
    MOVEMENT_MASK_SLICE,
    CombatToken,
    MovementToken,
)


@dataclass(slots=True)
class BcMetrics:
    loss: float
    joint_accuracy: float
    movement_accuracy: float
    combat_accuracy: float
    movement_recall: dict[int, float]
    combat_recall: dict[int, float]
    movement_confusion: np.ndarray
    combat_confusion: np.ndarray

    @property
    def accuracy(self) -> float:
        return self.joint_accuracy


@dataclass(slots=True)
class ClosedLoopMetrics:
    joint_accuracy: float
    movement_accuracy: float
    combat_accuracy: float
    static_escape_rate: float
    mean_first_action_step: float | None
    max_idle_run: int
    episodes: int
    movement_active_rate: float
    combat_active_rate: float
    movement_switch_rate: float
    combat_switch_rate: float
    movement_counts: dict[str, int]
    combat_counts: dict[str, int]


def assess_bc_release(
    metrics: ClosedLoopMetrics,
    validation: BcMetrics | None = None,
) -> tuple[bool, list[str]]:
    """Reject checkpoints with the failure modes seen in live evaluation."""

    reasons: list[str] = []
    if metrics.static_escape_rate < 0.8:
        reasons.append("static_escape_rate<0.8")
    if metrics.joint_accuracy < 0.12:
        reasons.append("closed_loop_joint_accuracy<0.12")
    if metrics.movement_accuracy < 0.25:
        reasons.append("closed_loop_movement_accuracy<0.25")
    if metrics.combat_accuracy < 0.40:
        reasons.append("closed_loop_combat_accuracy<0.40")
    if metrics.max_idle_run > 64:
        reasons.append("max_idle_run>64")
    if not 0.10 <= metrics.movement_active_rate <= 0.90:
        reasons.append("movement_active_rate_outside_[0.10,0.90]")
    if not 0.03 <= metrics.combat_active_rate <= 0.65:
        reasons.append("combat_active_rate_outside_[0.03,0.65]")
    predicted_count = max(sum(metrics.combat_counts.values()), 1)
    if metrics.combat_counts.get(CombatToken.LIGHT_ATTACK.name, 0) / predicted_count < 0.02:
        reasons.append("closed_loop_light_attack_rate<0.02")
    if metrics.combat_counts.get(CombatToken.DODGE.name, 0) / predicted_count < 0.005:
        reasons.append("closed_loop_dodge_rate<0.005")
    if validation is not None:
        if validation.combat_recall.get(int(CombatToken.LIGHT_ATTACK), 0.0) < 0.15:
            reasons.append("validation_light_attack_recall<0.15")
        if validation.combat_recall.get(int(CombatToken.DODGE), 0.0) < 0.03:
            reasons.append("validation_dodge_recall<0.03")
    return not reasons, reasons


def _harmonic_recall(confusion: np.ndarray, indices: list[int]) -> float:
    observed = [index for index in indices if confusion[index].sum() > 0]
    if not observed:
        return 0.0
    recalls = np.asarray(
        [confusion[index, index] / confusion[index].sum() for index in observed],
        dtype=np.float64,
    )
    return float(len(observed) / np.sum(1.0 / np.maximum(recalls, 1.0e-6)))


def core_balanced_score(metrics: BcMetrics, closed_loop: ClosedLoopMetrics | None = None) -> float:
    """Select checkpoints that cover both branches and survive autoregressive starts."""

    movement = _harmonic_recall(
        metrics.movement_confusion, list(range(MovementToken.size()))
    )
    combat = _harmonic_recall(
        metrics.combat_confusion,
        [
            int(CombatToken.NONE),
            int(CombatToken.LIGHT_ATTACK),
            int(CombatToken.HEAVY_HOLD),
            int(CombatToken.DODGE),
        ],
    )
    score = metrics.joint_accuracy * 0.5 * (movement + combat)
    if closed_loop is not None:
        score *= (0.25 + 0.75 * closed_loop.static_escape_rate) * np.sqrt(
            max(closed_loop.joint_accuracy, 1.0e-6)
        )
        score *= float(np.exp(-max(closed_loop.max_idle_run - 32, 0) / 64.0))
    return float(score)


class BehaviorCloningTrainer:
    def __init__(
        self,
        agent: R2D3Agent,
        sequence_length: int = 32,
        burn_in: int = 0,
        seed: int = 7,
        episode_prefix_fraction: float = 0.25,
        cold_start_loss_weight: float = 0.1,
        engagement_context: int = 4,
        previous_action_dropout: float = 0.25,
        balanced_window_fraction: float = 0.25,
    ) -> None:
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if burn_in < 0:
            raise ValueError("burn_in cannot be negative")
        if not 0.0 < episode_prefix_fraction <= 1.0:
            raise ValueError("episode_prefix_fraction must be within (0, 1]")
        if not 0.0 <= cold_start_loss_weight <= 1.0:
            raise ValueError("cold_start_loss_weight must be within [0, 1]")
        if engagement_context < 0:
            raise ValueError("engagement_context cannot be negative")
        if not 0.0 <= previous_action_dropout <= 1.0:
            raise ValueError("previous_action_dropout must be within [0, 1]")
        if not 0.0 <= balanced_window_fraction < 1.0:
            raise ValueError("balanced_window_fraction must be within [0, 1)")
        if episode_prefix_fraction + balanced_window_fraction > 1.0:
            raise ValueError("prefix and balanced fractions cannot exceed one batch")
        self.agent = agent
        self.sequence_length = sequence_length
        self.burn_in = burn_in
        self.rng = np.random.default_rng(seed)
        self.episode_prefix_fraction = episode_prefix_fraction
        self.cold_start_loss_weight = cold_start_loss_weight
        self.engagement_context = engagement_context
        self.previous_action_dropout = previous_action_dropout
        self.balanced_window_fraction = balanced_window_fraction

    def _load_episodes(self, paths: list[Path]) -> list[TrajectoryEpisode]:
        return [TrajectoryEpisode(path.parent, memory_map=True) for path in paths]

    @staticmethod
    def _effective_actions(episode: TrajectoryEpisode) -> np.ndarray:
        commands, masks, _ = episode.branched_actions()
        effective = np.empty_like(commands, dtype=np.int64)
        for index, command in enumerate(commands):
            movement = int(command[0]) if masks[index, int(command[0])] else 0
            combat_index = MovementToken.size() + int(command[1])
            combat = int(command[1]) if masks[index, combat_index] else 0
            effective[index] = (movement, combat)
        return effective

    @staticmethod
    def _action_masks(episode: TrajectoryEpisode) -> np.ndarray:
        _, masks, _ = episode.branched_actions()
        return np.asarray(masks, dtype=np.bool_)

    @staticmethod
    def _class_weights(counts: np.ndarray) -> np.ndarray:
        weights = np.ones(len(counts), dtype=np.float32)
        present = counts > 0
        if present.any():
            maximum = float(counts[present].max())
            weights[present] = np.minimum(
                np.sqrt(maximum / counts[present]), 4.0
            ).astype(np.float32)
        return weights

    def estimate_class_weights(self, paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
        episodes = self._load_episodes(paths)
        movement_counts = np.zeros(MovementToken.size(), dtype=np.int64)
        combat_counts = np.zeros(CombatToken.size(), dtype=np.int64)
        try:
            for episode in episodes:
                commands = self._effective_actions(episode)
                movement_counts += np.bincount(
                    commands[:, 0], minlength=MovementToken.size()
                )
                combat_counts += np.bincount(
                    commands[:, 1], minlength=CombatToken.size()
                )
        finally:
            for episode in episodes:
                episode.close()
        movement_weights = self._class_weights(movement_counts)
        combat_weights = self._class_weights(combat_counts)
        combat_weights[int(CombatToken.LIGHT_ATTACK)] = max(
            combat_weights[int(CombatToken.LIGHT_ATTACK)], 2.0
        )
        combat_weights[int(CombatToken.HEAVY_HOLD)] = min(
            combat_weights[int(CombatToken.HEAVY_HOLD)], 3.0
        )
        combat_weights[int(CombatToken.DODGE)] = max(
            combat_weights[int(CombatToken.DODGE)], 4.0
        )
        return movement_weights, combat_weights

    def _engagement_start(self, episode: TrajectoryEpisode, length: int) -> int | None:
        commands = self._effective_actions(episode)
        active = np.flatnonzero(np.any(commands != 0, axis=1))
        if not active.size:
            return None
        start = max(0, int(active[0]) - self.engagement_context)
        return start if start + length <= len(episode) else None

    def _sample_batch(self, episodes: list[TrajectoryEpisode], batch_size: int):
        length = self.burn_in + self.sequence_length
        eligible = [episode for episode in episodes if len(episode) >= length]
        if not eligible:
            raise ValueError(f"no demonstration episode contains {length} transitions")
        window_counts = np.asarray(
            [len(episode) - length + 1 for episode in eligible], dtype=np.float64
        )
        episode_probabilities = window_counts / window_counts.sum()
        cold_candidates = [
            (episode, start)
            for episode in eligible
            if (start := self._engagement_start(episode, length)) is not None
        ]
        prefix_count = min(
            batch_size,
            max(1, int(round(batch_size * self.episode_prefix_fraction))),
        )
        balanced_groups: list[list[tuple[TrajectoryEpisode, int]]] = []
        for branch_index, tokens in (
            (
                0,
                (
                    MovementToken.FORWARD,
                    MovementToken.BACK,
                    MovementToken.LEFT,
                    MovementToken.RIGHT,
                ),
            ),
            (
                1,
                (
                    CombatToken.LIGHT_ATTACK,
                    CombatToken.HEAVY_HOLD,
                    CombatToken.DODGE,
                    CombatToken.DRINK_POTION,
                ),
            ),
        ):
            for token in tokens:
                group: list[tuple[TrajectoryEpisode, int]] = []
                for episode in eligible:
                    indices = np.flatnonzero(
                        self._effective_actions(episode)[:, branch_index] == int(token)
                    )
                    group.extend((episode, int(index)) for index in indices)
                if group:
                    balanced_groups.append(group)
        balanced_count = min(
            batch_size - prefix_count,
            int(round(batch_size * self.balanced_window_fraction)),
        )
        selections: list[tuple[TrajectoryEpisode, int, bool]] = []
        for sample_index in range(batch_size):
            if sample_index < prefix_count and cold_candidates:
                episode, start = cold_candidates[int(self.rng.integers(len(cold_candidates)))]
                selections.append((episode, start, True))
            elif sample_index < prefix_count + balanced_count and balanced_groups:
                group = balanced_groups[int(self.rng.integers(len(balanced_groups)))]
                episode, target_index = group[int(self.rng.integers(len(group)))]
                target_offset = self.burn_in + int(
                    self.rng.integers(max(self.sequence_length, 1))
                )
                start = int(
                    np.clip(target_index - target_offset, 0, len(episode) - length)
                )
                selections.append((episode, start, False))
            else:
                episode = eligible[
                    int(self.rng.choice(len(eligible), p=episode_probabilities))
                ]
                start = int(self.rng.integers(0, len(episode) - length + 1))
                selections.append((episode, start, False))

        frames = np.stack(
            [np.asarray(ep.frames[start : start + length]) for ep, start, _ in selections]
        )
        features = np.stack(
            [ep.trajectory["features"][start : start + length] for ep, start, _ in selections]
        )
        confidence = np.stack(
            [ep.trajectory["confidence"][start : start + length] for ep, start, _ in selections]
        )
        action_masks = np.stack(
            [self._action_masks(ep)[start : start + length] for ep, start, _ in selections]
        )
        previous_rewards = np.stack(
            [ep.trajectory["previous_rewards"][start : start + length] for ep, start, _ in selections]
        )
        previous_actions = np.stack(
            [ep.branched_actions()[2][start : start + length] for ep, start, _ in selections]
        ).copy()
        actions = np.stack(
            [self._effective_actions(ep)[start : start + length] for ep, start, _ in selections]
        )
        cold_starts = np.asarray([cold for _, _, cold in selections], dtype=np.bool_)
        previous_actions[cold_starts, 0] = 0
        previous_rewards[cold_starts, 0] = 0.0
        return (
            frames,
            features,
            confidence,
            action_masks,
            previous_actions,
            previous_rewards,
            cold_starts,
            actions,
        )

    @staticmethod
    def _predicted_commands(q_values: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        movement = q_values[..., MOVEMENT_MASK_SLICE].masked_fill(
            ~masks[..., MOVEMENT_MASK_SLICE].bool(), torch.finfo(q_values.dtype).min
        ).argmax(dim=-1)
        combat = q_values[..., COMBAT_MASK_SLICE].masked_fill(
            ~masks[..., COMBAT_MASK_SLICE].bool(), torch.finfo(q_values.dtype).min
        ).argmax(dim=-1)
        return torch.stack([movement, combat], dim=-1)

    def _roll_forward(
        self,
        frames: torch.Tensor,
        features: torch.Tensor,
        confidence: torch.Tensor,
        action_masks: torch.Tensor,
        previous_actions: torch.Tensor,
        previous_rewards: torch.Tensor,
        state,
        model_feedback_probability: float,
        initial_feedback: torch.Tensor | None = None,
        previous_action_dropout: float = 0.0,
    ):
        if not 0.0 <= model_feedback_probability <= 1.0:
            raise ValueError("model_feedback_probability must be within [0, 1]")
        visual = self.agent.online.encode_visual(frames)
        q_steps: list[torch.Tensor] = []
        last_prediction = initial_feedback
        for timestep in range(frames.shape[1]):
            action_input = previous_actions[:, timestep]
            if last_prediction is not None and model_feedback_probability > 0.0:
                use_feedback = torch.from_numpy(
                    self.rng.random(action_input.shape[0]) < model_feedback_probability
                ).to(device=action_input.device)[:, None]
                action_input = torch.where(use_feedback, last_prediction, action_input)
            if previous_action_dropout > 0.0:
                drop_previous = torch.from_numpy(
                    self.rng.random(action_input.shape[0]) < previous_action_dropout
                ).to(device=action_input.device)[:, None]
                action_input = torch.where(
                    drop_previous, torch.zeros_like(action_input), action_input
                )
            q_values, state = self.agent.online.forward_from_visual(
                visual[:, timestep : timestep + 1],
                features[:, timestep : timestep + 1].float(),
                confidence[:, timestep : timestep + 1].float(),
                action_input[:, None].long(),
                previous_rewards[:, timestep : timestep + 1].float(),
                state,
            )
            q_steps.append(q_values)
            last_prediction = self._predicted_commands(
                q_values[:, 0], action_masks[:, timestep]
            ).detach()
        return torch.cat(q_steps, dim=1), state, last_prediction

    @staticmethod
    def _classification_loss(
        q_values: torch.Tensor,
        masks: torch.Tensor,
        targets: torch.Tensor,
        class_weights: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> torch.Tensor:
        losses = []
        for branch_index, branch_slice in enumerate(
            (MOVEMENT_MASK_SLICE, COMBAT_MASK_SLICE)
        ):
            branch_q = q_values[..., branch_slice].masked_fill(
                ~masks[..., branch_slice].bool(), torch.finfo(q_values.dtype).min
            )
            weight = None if class_weights is None else class_weights[branch_index]
            losses.append(
                functional.cross_entropy(
                    branch_q.flatten(0, 1),
                    targets[..., branch_index].long().flatten(),
                    weight=weight,
                )
            )
        return torch.stack(losses).mean()

    def _step(
        self,
        episodes: list[TrajectoryEpisode],
        batch_size: int,
        train: bool,
        class_weights: tuple[np.ndarray, np.ndarray] | None = None,
        model_feedback_probability: float = 0.0,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        values = self._sample_batch(episodes, batch_size)
        (
            frames,
            features,
            confidence,
            action_masks,
            previous_actions,
            previous_rewards,
            cold_starts,
            actions,
        ) = [
            torch.from_numpy(np.asarray(value)).to(self.agent.device) for value in values
        ]
        self.agent.online.train(train)
        previous_action_dropout = self.previous_action_dropout if train else 0.0
        learning_slice = slice(self.burn_in, None)
        state = None
        burn_prediction = None
        if self.burn_in:
            with torch.no_grad():
                _, state, burn_prediction = self._roll_forward(
                    frames[:, : self.burn_in],
                    features[:, : self.burn_in].float(),
                    confidence[:, : self.burn_in].float(),
                    action_masks[:, : self.burn_in],
                    previous_actions[:, : self.burn_in].long(),
                    previous_rewards[:, : self.burn_in].float(),
                    None,
                    model_feedback_probability,
                    previous_action_dropout=previous_action_dropout,
                )
            state = state.detach()
        weight_tensors = (
            tuple(torch.from_numpy(value).to(self.agent.device) for value in class_weights)
            if class_weights is not None
            else None
        )
        with torch.set_grad_enabled(train):
            q_values, _, _ = self._roll_forward(
                frames[:, learning_slice],
                features[:, learning_slice].float(),
                confidence[:, learning_slice].float(),
                action_masks[:, learning_slice],
                previous_actions[:, learning_slice].long(),
                previous_rewards[:, learning_slice].float(),
                state,
                model_feedback_probability,
                initial_feedback=burn_prediction,
                previous_action_dropout=previous_action_dropout,
            )
            learning_masks = action_masks[:, learning_slice]
            learning_targets = actions[:, learning_slice]
            loss = self._classification_loss(
                q_values, learning_masks, learning_targets, weight_tensors
            )
            if self.burn_in and cold_starts.any() and self.cold_start_loss_weight:
                prefix_q, _, _ = self._roll_forward(
                    frames[cold_starts, : self.burn_in],
                    features[cold_starts, : self.burn_in].float(),
                    confidence[cold_starts, : self.burn_in].float(),
                    action_masks[cold_starts, : self.burn_in],
                    previous_actions[cold_starts, : self.burn_in].long(),
                    previous_rewards[cold_starts, : self.burn_in].float(),
                    None,
                    model_feedback_probability,
                    previous_action_dropout=previous_action_dropout,
                )
                prefix_loss = self._classification_loss(
                    prefix_q,
                    action_masks[cold_starts, : self.burn_in],
                    actions[cold_starts, : self.burn_in],
                    weight_tensors,
                )
                loss = loss + self.cold_start_loss_weight * prefix_loss
        if train:
            self.agent.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.agent.online.parameters(), self.agent.config.gradient_clip
            )
            self.agent.optimizer.step()
        predictions = self._predicted_commands(q_values, learning_masks).detach().cpu().numpy()
        targets = learning_targets.detach().cpu().numpy()
        return float(loss.detach().cpu()), predictions.reshape(-1, 2), targets.reshape(-1, 2)

    @staticmethod
    def _confusion(targets: np.ndarray, predictions: np.ndarray, size: int) -> np.ndarray:
        confusion = np.zeros((size, size), dtype=np.int64)
        np.add.at(confusion, (targets, predictions), 1)
        return confusion

    def run_epoch(
        self,
        paths: list[Path],
        *,
        batch_size: int,
        steps: int,
        train: bool,
        class_weights: tuple[np.ndarray, np.ndarray] | None = None,
        model_feedback_probability: float = 0.0,
    ) -> BcMetrics:
        episodes = self._load_episodes(paths)
        losses: list[float] = []
        predictions: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        try:
            for _ in range(steps):
                loss, predicted, target = self._step(
                    episodes,
                    batch_size,
                    train,
                    class_weights,
                    model_feedback_probability,
                )
                losses.append(loss)
                predictions.append(predicted)
                targets.append(target)
        finally:
            for episode in episodes:
                episode.close()
        predicted = np.concatenate(predictions)
        target = np.concatenate(targets)
        movement_confusion = self._confusion(
            target[:, 0], predicted[:, 0], MovementToken.size()
        )
        combat_confusion = self._confusion(
            target[:, 1], predicted[:, 1], CombatToken.size()
        )
        movement_recall = {
            index: float(
                movement_confusion[index, index] / max(movement_confusion[index].sum(), 1)
            )
            for index in range(MovementToken.size())
        }
        combat_recall = {
            index: float(combat_confusion[index, index] / max(combat_confusion[index].sum(), 1))
            for index in range(CombatToken.size())
        }
        return BcMetrics(
            loss=float(np.mean(losses)),
            joint_accuracy=float(np.all(predicted == target, axis=1).mean()),
            movement_accuracy=float((predicted[:, 0] == target[:, 0]).mean()),
            combat_accuracy=float((predicted[:, 1] == target[:, 1]).mean()),
            movement_recall=movement_recall,
            combat_recall=combat_recall,
            movement_confusion=movement_confusion,
            combat_confusion=combat_confusion,
        )

    @torch.no_grad()
    def evaluate_closed_loop(
        self,
        paths: list[Path],
        *,
        static_ticks: int = 32,
    ) -> ClosedLoopMetrics:
        episodes = self._load_episodes(paths)
        predictions: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        first_action_steps: list[int] = []
        static_escapes = 0
        max_idle_run = 0
        movement_switches = 0
        combat_switches = 0
        switch_denominator = 0
        self.agent.online.eval()
        try:
            for episode in episodes:
                commands = self._effective_actions(episode)
                masks = self._action_masks(episode)[:-1]
                previous_rewards = np.asarray(
                    episode.trajectory["previous_rewards"][:-1], dtype=np.float32
                )
                frame_tensor = torch.from_numpy(np.asarray(episode.frames[:-1]).copy()).to(
                    self.agent.device
                )
                visual_chunks = []
                for start in range(0, len(episode), 256):
                    visual_chunks.append(
                        self.agent.online.encode_visual(
                            frame_tensor[start : start + 256][None]
                        )[0]
                    )
                visual = torch.cat(visual_chunks)
                features = torch.from_numpy(
                    np.asarray(episode.trajectory["features"][:-1]).copy()
                ).to(self.agent.device)
                confidence = torch.from_numpy(
                    np.asarray(episode.trajectory["confidence"][:-1]).copy()
                ).to(self.agent.device)
                mask_tensor = torch.from_numpy(masks.copy()).to(self.agent.device)
                reward_tensor = torch.from_numpy(previous_rewards.copy()).to(self.agent.device)
                state = None
                previous = torch.zeros((1, 2), dtype=torch.long, device=self.agent.device)
                episode_predictions = []
                idle_run = 0
                for timestep in range(len(episode)):
                    q_values, state = self.agent.online.forward_from_visual(
                        visual[timestep : timestep + 1][None],
                        features[timestep : timestep + 1][None],
                        confidence[timestep : timestep + 1][None],
                        previous[:, None],
                        reward_tensor[timestep : timestep + 1][None],
                        state,
                    )
                    previous = self._predicted_commands(
                        q_values[:, 0], mask_tensor[timestep : timestep + 1]
                    )
                    command = previous[0].cpu().numpy()
                    episode_predictions.append(command)
                    if np.any(command != 0):
                        idle_run = 0
                    else:
                        idle_run += 1
                        max_idle_run = max(max_idle_run, idle_run)
                episode_predictions_array = np.asarray(episode_predictions)
                predictions.append(episode_predictions_array)
                targets.append(commands)
                if len(episode_predictions_array) > 1:
                    movement_switches += int(
                        np.count_nonzero(np.diff(episode_predictions_array[:, 0]))
                    )
                    combat_switches += int(
                        np.count_nonzero(np.diff(episode_predictions_array[:, 1]))
                    )
                    switch_denominator += len(episode_predictions_array) - 1

                state = None
                previous = torch.zeros((1, 2), dtype=torch.long, device=self.agent.device)
                first_action = None
                for timestep in range(static_ticks):
                    q_values, state = self.agent.online.forward_from_visual(
                        visual[0:1][None],
                        features[0:1][None],
                        confidence[0:1][None],
                        previous[:, None],
                        torch.tensor(
                            [[0.0 if timestep == 0 else -0.001]],
                            device=self.agent.device,
                        ),
                        state,
                    )
                    previous = self._predicted_commands(q_values[:, 0], mask_tensor[0:1])
                    if torch.any(previous != 0):
                        first_action = timestep
                        break
                if first_action is not None:
                    static_escapes += 1
                    first_action_steps.append(first_action)
        finally:
            for episode in episodes:
                episode.close()
        predicted = np.concatenate(predictions)
        target = np.concatenate(targets)
        movement_counts = {
            action.name: int(np.count_nonzero(predicted[:, 0] == int(action)))
            for action in MovementToken
        }
        combat_counts = {
            action.name: int(np.count_nonzero(predicted[:, 1] == int(action)))
            for action in CombatToken
        }
        return ClosedLoopMetrics(
            joint_accuracy=float(np.all(predicted == target, axis=1).mean()),
            movement_accuracy=float((predicted[:, 0] == target[:, 0]).mean()),
            combat_accuracy=float((predicted[:, 1] == target[:, 1]).mean()),
            static_escape_rate=static_escapes / max(len(episodes), 1),
            mean_first_action_step=(
                float(np.mean(first_action_steps)) if first_action_steps else None
            ),
            max_idle_run=max_idle_run,
            episodes=len(episodes),
            movement_active_rate=float((predicted[:, 0] != int(MovementToken.NONE)).mean()),
            combat_active_rate=float((predicted[:, 1] != int(CombatToken.NONE)).mean()),
            movement_switch_rate=movement_switches / max(switch_denominator, 1),
            combat_switch_rate=combat_switches / max(switch_denominator, 1),
            movement_counts=movement_counts,
            combat_counts=combat_counts,
        )
