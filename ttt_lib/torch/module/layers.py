import math

import torch
import torch.nn as nn


class Bias(nn.Module):
    def __init__(self, n_channels, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Bias, self).__init__()
        self.n_channels = n_channels
        self.bias = nn.Parameter(torch.empty(n_channels, **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1 / math.sqrt(self.n_channels)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input + self.bias

    def extra_repr(self) -> str:
        return f'n_channels={self.n_channels}, bias={self.bias}'


class MaskedSoftmax(nn.Softmax):
    def forward(self, input: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        masked_input = torch.exp(input - input.max(dim=1, keepdim=True)[0]) * (mask == 1).float()
        return masked_input / torch.clip(torch.sum(masked_input, dim=1, keepdim=True), 1e-6)
