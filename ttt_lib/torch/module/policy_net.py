import torch
import torch.nn as nn
from dpipe.layers import ResBlock2d
from dpipe.torch import to_device

from ttt_lib.torch.module.layers import Bias, MaskedSoftmax


class PolicyNetworkQ10(nn.Module):
    def __init__(self, structure=None, n=10, in_channels=5):
        super().__init__()
        if structure is None:
            structure = (128, 64)
        self.structure = structure
        self.n = n
        self.in_channels = in_channels

        n_features = structure[0]
        n_features_policy = structure[1]
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, n_features, kernel_size=3, padding=1, bias=False),
            # ResBlock actually starts with BN -> ReLU
            # nn.BatchNorm2d(n_features),
            # nn.ReLU(inplace=True),
            ResBlock2d(n_features, n_features, kernel_size=3, padding=1),
            ResBlock2d(n_features, n_features, kernel_size=3, padding=1),
            ResBlock2d(n_features, n_features, kernel_size=3, padding=1),
            ResBlock2d(n_features, n_features, kernel_size=3, padding=1),
            ResBlock2d(n_features, n_features, kernel_size=3, padding=1),
            # policy head
            nn.Conv2d(n_features, n_features_policy, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_features_policy, 1, kernel_size=1, padding=0, bias=True),
        )

        self.tanh = nn.Tanh()
        self.flatten = nn.Flatten()
        self.masked_softmax = MaskedSoftmax(dim=1)

    def forward(self, x):
        return self.model(x)

    def predict_action_values(self, logits):
        return self.tanh(logits)

    def predict_proba(self, x, logits):
        p = self.masked_softmax(self.flatten(logits), self.flatten(x[:, -1, ...].unsqueeze(1)))
        return torch.reshape(p, shape=(-1, 1, self.n, self.n))


class PolicyNetworkRandom(nn.Module):
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
