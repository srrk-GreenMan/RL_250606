import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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