import torch
import torch.nn as nn
from dpipe.layers import ResBlock2d
from dpipe.torch import to_device

from ttt_lib.torch.module.layers import Bias, MaskedSoftmax


class PolicyNetworkQ3(nn.Module):
    def __init__(self, structure=None, n=3, in_channels=5):
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
            nn.ReLU(inplace=True),
            ResBlock2d(n_features, n_features, kernel_size=3, padding=1, batch_norm_module=nn.Identity),
            # policy head
            nn.Conv2d(n_features, n_features_policy, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_features_policy, 1, kernel_size=1, padding=0, bias=False),
            nn.Flatten(),
            Bias(n_channels=n ** 2),
            nn.Tanh(),
        )
        self.flatten = nn.Flatten()
        self.masked_softmax = MaskedSoftmax(dim=1)

    def forward(self, x):
        return torch.reshape(self.model(x), shape=(-1, 1, self.n, self.n))

    def predict_proba(self, x, q):
        p = self.masked_softmax(self.flatten(q), self.flatten(x[:, -1, ...].unsqueeze(1)))
        return torch.reshape(p, shape=(-1, 1, self.n, self.n))


class PolicyNetworkQ6(nn.Module):
    def __init__(self, structure=None, n=6, in_channels=5):
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
            nn.ReLU(inplace=True),
            ResBlock2d(n_features, n_features, kernel_size=3, padding=1, batch_norm_module=nn.Identity),
            ResBlock2d(n_features, n_features, kernel_size=3, padding=1, batch_norm_module=nn.Identity),
            ResBlock2d(n_features, n_features, kernel_size=3, padding=1, batch_norm_module=nn.Identity),
            # policy head
            nn.Conv2d(n_features, n_features_policy, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_features_policy, 1, kernel_size=1, padding=0, bias=False),
            nn.Flatten(),
            Bias(n_channels=n ** 2),
            nn.Tanh(),
        )
        self.flatten = nn.Flatten()
        self.masked_softmax = MaskedSoftmax(dim=1)

    def forward(self, x):
        return torch.reshape(self.model(x), shape=(-1, 1, self.n, self.n))

    def predict_proba(self, x, q):
        p = self.masked_softmax(self.flatten(q), self.flatten(x[:, -1, ...].unsqueeze(1)))
        return torch.reshape(p, shape=(-1, 1, self.n, self.n))


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
            nn.ReLU(inplace=True),
            ResBlock2d(n_features, n_features, kernel_size=3, padding=1, batch_norm_module=nn.Identity),
            ResBlock2d(n_features, n_features, kernel_size=3, padding=1, batch_norm_module=nn.Identity),
            ResBlock2d(n_features, n_features, kernel_size=3, padding=1, batch_norm_module=nn.Identity),
            ResBlock2d(n_features, n_features, kernel_size=3, padding=1, batch_norm_module=nn.Identity),
            # policy head
            nn.Conv2d(n_features, n_features_policy, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_features_policy, 1, kernel_size=1, padding=0, bias=False),
            nn.Flatten(),
            Bias(n_channels=n ** 2),
            nn.Tanh(),
        )
        self.flatten = nn.Flatten()
        self.masked_softmax = MaskedSoftmax(dim=1)

    def forward(self, x):
        return torch.reshape(self.model(x), shape=(-1, 1, self.n, self.n))

    def predict_proba(self, x, q):
        p = self.masked_softmax(self.flatten(q), self.flatten(x[:, -1, ...].unsqueeze(1)))
        return torch.reshape(p, shape=(-1, 1, self.n, self.n))


class PolicyNetworkQ10Light(nn.Module):
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
            nn.ReLU(inplace=True),
            ResBlock2d(n_features, n_features, kernel_size=3, padding=1, batch_norm_module=nn.Identity),
            ResBlock2d(n_features, n_features, kernel_size=3, padding=1, batch_norm_module=nn.Identity),
            # policy head
            nn.Conv2d(n_features, n_features_policy, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_features_policy, 1, kernel_size=1, padding=0, bias=False),
            nn.Flatten(),
            Bias(n_channels=n ** 2),
            nn.Tanh(),
        )
        self.flatten = nn.Flatten()
        self.masked_softmax = MaskedSoftmax(dim=1)

    def forward(self, x):
        return torch.reshape(self.model(x), shape=(-1, 1, self.n, self.n))

    def predict_proba(self, x, q):
        p = self.masked_softmax(self.flatten(q), self.flatten(x[:, -1, ...].unsqueeze(1)))
        return torch.reshape(p, shape=(-1, 1, self.n, self.n))


class PolicyNetworkRandom(nn.Module):
    def __init__(self, n=3, in_channels=5):
        super().__init__()
        self.n = n
        self.in_channels = in_channels

        self.flatten = nn.Flatten()
        self.masked_softmax = MaskedSoftmax(dim=1)

    def forward(self, x):
        return to_device(torch.zeros((1, 1, self.n, self.n), dtype=torch.float32), device=x)

    def predict_proba(self, x, q):
        p = self.masked_softmax(self.flatten(q), self.flatten(x[:, -1, ...].unsqueeze(1)))
        return torch.reshape(p, shape=(-1, 1, self.n, self.n))
