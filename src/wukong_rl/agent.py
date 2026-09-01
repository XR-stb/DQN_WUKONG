from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as functional

from .config import ModelConfig
from .model import RecurrentDuelingQNetwork, RecurrentState
from .replay import ReplayBatch
from .types import ActionToken, Observation


@dataclass(slots=True)
class LearnerMetrics:
    loss: float
    td_loss: float
    demo_loss: float
    mean_q: float
    mean_target: float
    gradient_norm: float
    priorities: np.ndarray


class R2D3Agent:
    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        config: ModelConfig,
        device: str | torch.device | None = None,
    ) -> None:
        self.config = config
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.online = RecurrentDuelingQNetwork(feature_dim, action_dim, config.hidden_size).to(self.device)
        self.target = copy.deepcopy(self.online).to(self.device).eval()
        for parameter in self.target.parameters():
            parameter.requires_grad_(False)
        self.optimizer = torch.optim.AdamW(
            self.online.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        self.amp_enabled = self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)
        self.action_dim = action_dim
        self.feature_dim = feature_dim
        self.learner_steps = 0

    def initial_state(self, batch_size: int = 1) -> RecurrentState:
        return self.online.initial_state(batch_size, self.device)

    @torch.no_grad()
    def act(
        self,
        observation: Observation,
        state: RecurrentState | None,
        epsilon: float,
        rng: np.random.Generator,
    ) -> tuple[ActionToken, RecurrentState, np.ndarray]:
        frames = torch.from_numpy(observation.frame[None, None]).to(self.device)
        features = torch.from_numpy(observation.features[None, None]).to(self.device)
        confidence = torch.from_numpy(observation.feature_confidence[None, None]).to(self.device)
        previous_actions = torch.tensor(
            [[int(observation.previous_action)]], dtype=torch.long, device=self.device
        )
        previous_rewards = torch.tensor(
            [[observation.previous_reward]], dtype=torch.float32, device=self.device
        )
        self.online.eval()
        q_values, next_state = self.online(
            frames, features, confidence, previous_actions, previous_rewards, state
        )
        q_numpy = q_values[0, 0].float().cpu().numpy()
        valid = np.flatnonzero(observation.action_mask)
        if valid.size == 0:
            valid = np.asarray([int(ActionToken.IDLE)])
        if rng.random() < epsilon:
            action_index = int(rng.choice(valid))
        else:
            masked_q = np.where(observation.action_mask, q_numpy, -np.inf)
            action_index = int(np.argmax(masked_q))
        return ActionToken(action_index), next_state.detach(), q_numpy

    def _to_tensor(self, value: np.ndarray, dtype=None) -> torch.Tensor:
        tensor = torch.from_numpy(np.asarray(value)).to(self.device, non_blocking=True)
        return tensor.to(dtype=dtype) if dtype is not None else tensor

    def _run_with_burn_in(
        self,
        network: RecurrentDuelingQNetwork,
        frames: torch.Tensor,
        features: torch.Tensor,
        confidence: torch.Tensor,
        previous_actions: torch.Tensor,
        previous_rewards: torch.Tensor,
        training: bool,
    ) -> torch.Tensor:
        burn = self.config.burn_in
        state = None
        if burn:
            with torch.no_grad():
                _, state = network(
                    frames[:, :burn],
                    features[:, :burn],
                    confidence[:, :burn],
                    previous_actions[:, :burn],
                    previous_rewards[:, :burn],
                )
                state = state.detach()
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            q_values, _ = network(
                frames[:, burn:],
                features[:, burn:],
                confidence[:, burn:],
                previous_actions[:, burn:],
                previous_rewards[:, burn:],
                state,
            )
        return q_values

    def learn(self, batch: ReplayBatch) -> LearnerMetrics:
        frames = self._to_tensor(batch.frames)
        features = self._to_tensor(batch.features, torch.float32)
        confidence = self._to_tensor(batch.confidence, torch.float32)
        action_masks = self._to_tensor(batch.action_masks, torch.bool)
        previous_actions = self._to_tensor(batch.previous_actions, torch.long)
        previous_rewards = self._to_tensor(batch.previous_rewards, torch.float32)
        actions = self._to_tensor(batch.actions, torch.long)
        rewards = self._to_tensor(batch.rewards, torch.float32)
        terminated = self._to_tensor(batch.terminated, torch.bool)
        truncated = self._to_tensor(batch.truncated, torch.bool)
        weights = self._to_tensor(batch.weights, torch.float32)
        demonstrations = self._to_tensor(batch.demonstrations, torch.bool)
        burn = self.config.burn_in
        unroll = self.config.unroll
        n_step = self.config.n_step

        self.online.train()
        with torch.amp.autocast(device_type=self.device.type, enabled=self.amp_enabled):
            online_q = self._run_with_burn_in(
                self.online,
                frames,
                features,
                confidence,
                previous_actions,
                previous_rewards,
                training=True,
            )
            with torch.no_grad():
                target_q = self._run_with_burn_in(
                    self.target,
                    frames,
                    features,
                    confidence,
                    previous_actions,
                    previous_rewards,
                    training=False,
                )
            chosen_actions = actions[:, burn : burn + unroll]
            chosen_masks = action_masks[:, burn : burn + unroll]
            chosen_valid = chosen_masks.gather(
                -1, chosen_actions.unsqueeze(-1)
            ).squeeze(-1)
            chosen_actions = torch.where(
                chosen_valid, chosen_actions, torch.zeros_like(chosen_actions)
            )
            predicted = online_q[:, :unroll].gather(-1, chosen_actions.unsqueeze(-1)).squeeze(-1)

            returns = torch.zeros_like(predicted)
            alive = torch.ones_like(predicted)
            discount = 1.0
            for offset in range(n_step):
                index = slice(burn + offset, burn + offset + unroll)
                returns = returns + discount * alive * rewards[:, index]
                done = terminated[:, index] | truncated[:, index]
                alive = alive * (~done).float()
                discount *= self.config.gamma

            next_online = online_q[:, n_step : n_step + unroll].detach()
            next_masks = action_masks[:, burn + n_step : burn + n_step + unroll]
            next_actions = next_online.masked_fill(~next_masks, -torch.inf).argmax(dim=-1)
            next_target = target_q[:, n_step : n_step + unroll].gather(
                -1, next_actions.unsqueeze(-1)
            ).squeeze(-1)
            targets = returns + discount * alive * next_target
            td_errors = targets.detach() - predicted
            element_loss = functional.smooth_l1_loss(predicted, targets.detach(), reduction="none")
            td_loss = (element_loss.mean(dim=1) * weights).mean()

            demo_loss = torch.zeros((), device=self.device)
            if demonstrations.any():
                demo_q = online_q[demonstrations, :unroll]
                demo_actions = chosen_actions[demonstrations]
                demo_masks = chosen_masks[demonstrations]
                margins = torch.full_like(demo_q, self.config.demo_margin)
                margins.scatter_(-1, demo_actions.unsqueeze(-1), 0.0)
                expert_q = demo_q.gather(-1, demo_actions.unsqueeze(-1)).squeeze(-1)
                competing_q = (demo_q + margins).masked_fill(~demo_masks, -torch.inf)
                demo_loss = competing_q.max(dim=-1).values.sub(expert_q).mean()
            loss = td_loss + self.config.demo_loss_weight * demo_loss

        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.online.parameters(), self.config.gradient_clip
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.learner_steps += 1
        if self.learner_steps % self.config.target_update_interval == 0:
            self.sync_target()
        absolute_td = td_errors.detach().abs()
        priorities = (
            0.9 * absolute_td.max(dim=1).values + 0.1 * absolute_td.mean(dim=1)
        ).float().cpu().numpy()
        return LearnerMetrics(
            loss=float(loss.detach().cpu()),
            td_loss=float(td_loss.detach().cpu()),
            demo_loss=float(demo_loss.detach().cpu()),
            mean_q=float(predicted.detach().mean().cpu()),
            mean_target=float(targets.detach().mean().cpu()),
            gradient_norm=float(gradient_norm.detach().cpu()),
            priorities=priorities,
        )

    def sync_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())

    def exploration(self, step: int, start: float, end: float, decay_steps: int) -> float:
        fraction = min(max(step / max(decay_steps, 1), 0.0), 1.0)
        return float(start + fraction * (end - start))

    @staticmethod
    def action_entropy(q_values: np.ndarray, mask: np.ndarray) -> float:
        valid = q_values[np.asarray(mask, dtype=bool)]
        if valid.size <= 1:
            return 0.0
        shifted = valid - valid.max()
        probabilities = np.exp(shifted) / np.exp(shifted).sum()
        return float(-(probabilities * np.log(probabilities + 1.0e-8)).sum() / math.log(valid.size))
