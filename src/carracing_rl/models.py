from __future__ import annotations

import torch
from torch import nn


class NatureCNN(nn.Module):
    """Convolutional encoder used by many DQN Atari-style agents."""

    def __init__(self, input_shape: tuple[int, int, int], hidden_dim: int = 512) -> None:
        super().__init__()
        channels, height, width = input_shape
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, channels, height, width)
            feature_dim = self.features(dummy).shape[1]
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(self.features(x))


class DQNNetwork(nn.Module):
    """DQN Q-network with optional dueling value/advantage heads."""

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        num_actions: int,
        hidden_dim: int = 512,
        dueling: bool = True,
    ) -> None:
        super().__init__()
        self.dueling = dueling
        self.encoder = NatureCNN(input_shape, hidden_dim=hidden_dim)
        if dueling:
            self.value = nn.Linear(hidden_dim, 1)
            self.advantage = nn.Linear(hidden_dim, num_actions)
        else:
            self.head = nn.Linear(hidden_dim, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        if not self.dueling:
            return self.head(features)
        value = self.value(features)
        advantage = self.advantage(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)
