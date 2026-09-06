from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.net(inputs)


class ImpalaStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.convolution = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.residual = nn.Sequential(ResidualBlock(out_channels), ResidualBlock(out_channels))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.residual(self.pool(self.convolution(inputs)))


class VisualEncoder(nn.Module):
    def __init__(self, output_dim: int = 256) -> None:
        super().__init__()
        self.stages = nn.Sequential(
            ImpalaStage(3, 16),
            ImpalaStage(16, 32),
            ImpalaStage(32, 32),
            nn.ReLU(inplace=False),
            nn.AdaptiveAvgPool2d((4, 6)),
            nn.Flatten(),
            nn.Linear(32 * 4 * 6, output_dim),
            nn.ReLU(inplace=False),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.stages(frames.float().div(255.0))


@dataclass(slots=True)
class RecurrentState:
    hidden: torch.Tensor
    cell: torch.Tensor

    def detach(self) -> "RecurrentState":
        return RecurrentState(self.hidden.detach(), self.cell.detach())


class RecurrentDuelingQNetwork(nn.Module):
    def __init__(self, feature_dim: int, action_dim: int, hidden_size: int = 256) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.visual = VisualEncoder(output_dim=256)
        self.action_embedding = nn.Embedding(action_dim, 32)
        scalar_input = feature_dim * 2 + 32 + 1
        self.fusion = nn.Sequential(
            nn.Linear(256 + scalar_input, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=False),
        )
        self.memory = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.value = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ReLU(inplace=False), nn.Linear(hidden_size, 1)
        )
        self.advantage = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_size, action_dim),
        )

    def initial_state(self, batch_size: int, device: torch.device | None = None) -> RecurrentState:
        device = device or next(self.parameters()).device
        shape = (1, batch_size, self.hidden_size)
        return RecurrentState(torch.zeros(shape, device=device), torch.zeros(shape, device=device))

    def encode_visual(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 5:
            raise ValueError("frames must be [batch, time, height, width, channels]")
        batch, timesteps = frames.shape[:2]
        visual = frames.permute(0, 1, 4, 2, 3).reshape(
            batch * timesteps, frames.shape[-1], frames.shape[2], frames.shape[3]
        )
        return self.visual(visual).reshape(batch, timesteps, -1)

    def forward_from_visual(
        self,
        visual: torch.Tensor,
        features: torch.Tensor,
        confidence: torch.Tensor,
        previous_actions: torch.Tensor,
        previous_rewards: torch.Tensor,
        state: RecurrentState | None = None,
    ) -> tuple[torch.Tensor, RecurrentState]:
        batch = visual.shape[0]
        previous_actions = previous_actions.long().clamp(0, self.action_dim - 1)
        action_embedding = self.action_embedding(previous_actions)
        scalars = torch.cat(
            [features * confidence, confidence, action_embedding, previous_rewards.unsqueeze(-1)],
            dim=-1,
        )
        fused = self.fusion(torch.cat([visual, scalars], dim=-1))
        if state is None:
            state = self.initial_state(batch, fused.device)
        memory, (hidden, cell) = self.memory(fused, (state.hidden, state.cell))
        value = self.value(memory)
        advantage = self.advantage(memory)
        q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q_values, RecurrentState(hidden, cell)

    def forward(
        self,
        frames: torch.Tensor,
        features: torch.Tensor,
        confidence: torch.Tensor,
        previous_actions: torch.Tensor,
        previous_rewards: torch.Tensor,
        state: RecurrentState | None = None,
    ) -> tuple[torch.Tensor, RecurrentState]:
        return self.forward_from_visual(
            self.encode_visual(frames),
            features,
            confidence,
            previous_actions,
            previous_rewards,
            state,
        )
