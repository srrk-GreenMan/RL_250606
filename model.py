import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AtariCNN(nn.Module):
    """Atari-style CNN feature extractor used in many RL algorithms."""

    def __init__(self, input_shape):
        super().__init__()
        c, h, w = input_shape
        self.cnn = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            sample = torch.zeros(1, *input_shape)
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, 512),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)
        return self.linear(x)

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(QNetwork, self).__init__()
        input_dim = int(np.prod(state_dim))  # For (4, 84, 84), this is 28224

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        ) # [TODO] Define Q network with one hidden layer

    def forward(self, x):
        return self.net(x)


class ActorCritic(nn.Module):
    """Simple shared backbone actor-critic network."""

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        input_dim = int(np.prod(state_dim))

        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.shared(x)
        logits = self.policy_head(x)
        value = self.value_head(x)
        return logits, value
