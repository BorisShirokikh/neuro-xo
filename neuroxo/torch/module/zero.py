import torch
import torch.nn as nn
from dpipe.layers import ResBlock2d

from neuroxo.torch.module.layers import MaskedSoftmax


class NeuroXOZeroNN(nn.Module):
    def __init__(self, n: int = 10, in_channels: int = 2, n_blocks=19, n_features=256):
        super().__init__()
        self.n = n
        self.in_channels = in_channels
        self.n_blocks = n_blocks
        self.n_features = n_features

        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, n_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_features, n_features),
            nn.ReLU(),

            *[ResBlock2d(n_features) for _ in range(n_blocks)]
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(n_features, 2, kernel_size=1, stride=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * n ** 2, n ** 2),
        )
        self.flatten = nn.Flatten()
        self.masked_softmax = MaskedSoftmax(dim=1)

        self.value_head = nn.Sequential(
            nn.Conv2d(n_features, 1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(n ** 2, n_features * n ** 2),
            nn.ReLU(),
            nn.Linear(n_features * n ** 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        logits = self.backbone(x)
        proba = self.masked_softmax(self.flatten(logits), self.flatten(torch.sum(x[:, :2, ...], dim=1).unsqueeze(1)))
        return torch.reshape(proba, shape=(-1, 1, self.n, self.n)), self.value_head(logits).squeeze()
