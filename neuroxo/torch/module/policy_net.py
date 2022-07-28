import torch
import torch.nn as nn
from dpipe.layers import ResBlock2d, PostActivation2d
from dpipe.torch import to_device

from neuroxo.torch.module.layers import MaskedSoftmax


class ProbaPolicyNN(nn.Module):
    def __init__(self, n: int = 10, in_channels: int = 2, n_blocks: int = 11, n_features: int = 128):
        super().__init__()
        self.n = n
        self.in_channels = in_channels
        self.n_blocks = n_blocks
        self.n_features = n_features

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, n_features, kernel_size=3, padding=1, bias=False),
            nn.Sequential(*[ResBlock2d(n_features, n_features, kernel_size=3, padding=1) for _ in range(n_features)]),
            PostActivation2d(n_features, n_features, kernel_size=1, padding=0),
            nn.Conv2d(n_features, 1, kernel_size=1, padding=0, bias=True),
        )

        self.tanh = nn.Tanh()
        self.flatten = nn.Flatten()
        self.masked_softmax = MaskedSoftmax(dim=1)

    def forward(self, x):
        logits = self.model(x)
        policy = self.tanh(logits)
        proba = self.masked_softmax(self.flatten(logits), self.flatten(torch.sum(x[:, :2, ...], dim=1).unsqueeze(1)))
        return torch.reshape(proba, shape=(-1, 1, self.n, self.n)), policy


class RandomProbaPolicyNN(nn.Module):
    # FIXME: should actually generate and return random variables
    def __init__(self, n=10, in_channels=5):
        super().__init__()
        self.n = n
        self.in_channels = in_channels

        self.tanh = nn.Tanh()
        self.flatten = nn.Flatten()
        self.masked_softmax = MaskedSoftmax(dim=1)

    def forward(self, x):
        return to_device(torch.zeros((1, 1, self.n, self.n), dtype=torch.float32), device=x)

    def predict_action_values(self, logits):
        return self.tanh(logits)

    def predict_proba(self, x, logits):
        p = self.masked_softmax(self.flatten(logits), self.flatten(x[:, -1, ...].unsqueeze(1)))
        return torch.reshape(p, shape=(-1, 1, self.n, self.n))
