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
from .types import (
    ACTION_MASK_SIZE,
    COMBAT_MASK_SLICE,
    MOVEMENT_MASK_SLICE,
    ActionCommand,
    CombatToken,
    MovementToken,
    Observation,
    split_action_mask,
)


@dataclass(slots=True)
class LearnerMetrics:
    loss: float
    td_loss: float
    demo_loss: float
    mean_q: float
    mean_target: float
    gradient_norm: float
    update_skipped: bool
    priorities: np.ndarray


class R2D3Agent:
    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        config: ModelConfig,
        device: str | torch.device | None = None,
    ) -> None:
        if action_dim != ACTION_MASK_SIZE:
            raise ValueError(f"action_dim must be {ACTION_MASK_SIZE}")
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
    ) -> tuple[ActionCommand, RecurrentState, np.ndarray]:
        frames = torch.from_numpy(observation.frame[None, None]).to(self.device)
        features = torch.from_numpy(observation.features[None, None]).to(self.device)
        confidence = torch.from_numpy(observation.feature_confidence[None, None]).to(self.device)
        previous_actions = torch.from_numpy(observation.previous_action.as_array()[None, None]).to(
            self.device, dtype=torch.long
        )
        previous_rewards = torch.tensor(
            [[observation.previous_reward]], dtype=torch.float32, device=self.device
        )
        self.online.eval()
        q_values, next_state = self.online(
            frames, features, confidence, previous_actions, previous_rewards, state
        )
        q_numpy = q_values[0, 0].float().cpu().numpy()
        movement_mask, combat_mask = split_action_mask(observation.action_mask)
        explore = rng.random() < epsilon

        def select(values: np.ndarray, mask: np.ndarray) -> int:
            valid = np.flatnonzero(mask)
            if explore:
                return int(rng.choice(valid))
            return int(np.argmax(np.where(mask, values, -np.inf)))

        movement = select(q_numpy[MOVEMENT_MASK_SLICE], movement_mask)
        combat = select(q_numpy[COMBAT_MASK_SLICE], combat_mask)
        return (
            ActionCommand(MovementToken(movement), CombatToken(combat)),
            next_state.detach(),
            q_numpy,
        )

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
            returns = torch.zeros_like(rewards[:, burn : burn + unroll])
            alive = torch.ones_like(returns)
            discount = 1.0
            for offset in range(n_step):
                index = slice(burn + offset, burn + offset + unroll)
                returns = returns + discount * alive * rewards[:, index]
                done = terminated[:, index] | truncated[:, index]
                alive = alive * (~done).float()
                discount *= self.config.gamma

            next_masks = action_masks[:, burn + n_step : burn + n_step + unroll]
            chosen_masks = action_masks[:, burn : burn + unroll]
            branch_specs = (
                (MOVEMENT_MASK_SLICE, 0),
                (COMBAT_MASK_SLICE, 1),
            )
            branch_predictions: list[torch.Tensor] = []
            branch_targets: list[torch.Tensor] = []
            branch_errors: list[torch.Tensor] = []
            branch_td_losses: list[torch.Tensor] = []
            branch_demo_losses: list[torch.Tensor] = []
            for branch_slice, branch_index in branch_specs:
                branch_online = online_q[..., branch_slice]
                branch_target = target_q[..., branch_slice]
                branch_actions = actions[:, burn : burn + unroll, branch_index]
                branch_masks = chosen_masks[..., branch_slice]
                chosen_valid = branch_masks.gather(
                    -1, branch_actions.unsqueeze(-1)
                ).squeeze(-1)
                branch_actions = torch.where(
                    chosen_valid, branch_actions, torch.zeros_like(branch_actions)
                )
                predicted = branch_online[:, :unroll].gather(
                    -1, branch_actions.unsqueeze(-1)
                ).squeeze(-1)
                next_online = branch_online[:, n_step : n_step + unroll].detach()
                branch_next_masks = next_masks[..., branch_slice]
                next_actions = next_online.masked_fill(
                    ~branch_next_masks, -torch.inf
                ).argmax(dim=-1)
                next_target = branch_target[:, n_step : n_step + unroll].gather(
                    -1, next_actions.unsqueeze(-1)
                ).squeeze(-1)
                targets = returns + discount * alive * next_target
                td_errors = targets.detach() - predicted
                element_loss = functional.smooth_l1_loss(
                    predicted, targets.detach(), reduction="none"
                )
                branch_td_losses.append((element_loss.mean(dim=1) * weights).mean())
                branch_predictions.append(predicted)
                branch_targets.append(targets)
                branch_errors.append(td_errors)

                if demonstrations.any():
                    demo_q = branch_online[demonstrations, :unroll]
                    demo_actions = branch_actions[demonstrations]
                    demo_masks = branch_masks[demonstrations]
                    margins = torch.full_like(demo_q, self.config.demo_margin)
                    margins.scatter_(-1, demo_actions.unsqueeze(-1), 0.0)
                    expert_q = demo_q.gather(
                        -1, demo_actions.unsqueeze(-1)
                    ).squeeze(-1)
                    competing_q = (demo_q + margins).masked_fill(~demo_masks, -torch.inf)
                    branch_demo_losses.append(
                        competing_q.max(dim=-1).values.sub(expert_q).mean()
                    )

            td_loss = torch.stack(branch_td_losses).mean()
            demo_loss = (
                torch.stack(branch_demo_losses).mean()
                if branch_demo_losses
                else torch.zeros((), device=self.device)
            )
            loss = td_loss + self.config.demo_loss_weight * demo_loss

        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.online.parameters(), self.config.gradient_clip
        )
        update_skipped = not bool(torch.isfinite(gradient_norm).item())
        if update_skipped:
            # GradScaler normally skips a CUDA optimizer step after detecting
            # non-finite gradients, but CPU training has no active scaler. Keep
            # the same safety guarantee on every device and make the skip
            # observable in learner metrics.
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.update()
        else:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.learner_steps += 1
            if self.learner_steps % self.config.target_update_interval == 0:
                self.sync_target()
        absolute_td = torch.stack(branch_errors).detach().abs().amax(dim=0)
        priorities = (
            0.9 * absolute_td.max(dim=1).values + 0.1 * absolute_td.mean(dim=1)
        ).float().cpu().numpy()
        return LearnerMetrics(
            loss=float(loss.detach().cpu()),
            td_loss=float(td_loss.detach().cpu()),
            demo_loss=float(demo_loss.detach().cpu()),
            mean_q=float(torch.stack(branch_predictions).detach().mean().cpu()),
            mean_target=float(torch.stack(branch_targets).detach().mean().cpu()),
            gradient_norm=(
                0.0 if update_skipped else float(gradient_norm.detach().cpu())
            ),
            update_skipped=update_skipped,
            priorities=priorities,
        )

    def sync_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())

    def exploration(self, step: int, start: float, end: float, decay_steps: int) -> float:
        fraction = min(max(step / max(decay_steps, 1), 0.0), 1.0)
        return float(start + fraction * (end - start))

    @staticmethod
    def action_entropy(q_values: np.ndarray, mask: np.ndarray) -> float:
        entropies: list[float] = []
        for branch_slice in (MOVEMENT_MASK_SLICE, COMBAT_MASK_SLICE):
            valid = q_values[branch_slice][np.asarray(mask[branch_slice], dtype=bool)]
            if valid.size <= 1:
                entropies.append(0.0)
                continue
            shifted = valid - valid.max()
            probabilities = np.exp(shifted) / np.exp(shifted).sum()
            entropies.append(
                float(
                    -(probabilities * np.log(probabilities + 1.0e-8)).sum()
                    / math.log(valid.size)
                )
            )
        return float(np.mean(entropies))

    @staticmethod
    def action_diagnostics(q_values: np.ndarray, mask: np.ndarray) -> dict[str, object]:
        diagnostics: dict[str, object] = {}
        for name, branch_slice, enum_type in (
            ("movement", MOVEMENT_MASK_SLICE, MovementToken),
            ("combat", COMBAT_MASK_SLICE, CombatToken),
        ):
            values = np.where(mask[branch_slice], q_values[branch_slice], -np.inf)
            order = np.argsort(values)[::-1]
            valid_order = [int(index) for index in order if np.isfinite(values[index])]
            top = valid_order[:2]
            diagnostics[f"{name}_top"] = [enum_type(index).name for index in top]
            diagnostics[f"{name}_top_q"] = [float(values[index]) for index in top]
            diagnostics[f"{name}_margin"] = (
                float(values[top[0]] - values[top[1]]) if len(top) > 1 else None
            )
            diagnostics[f"{name}_valid"] = int(np.count_nonzero(mask[branch_slice]))
        return diagnostics
