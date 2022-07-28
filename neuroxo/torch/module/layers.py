import torch
import torch.nn as nn


class MaskedSoftmax(nn.Softmax):
    def forward(self, input: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        masked_input = torch.exp(input - input.max(dim=1, keepdim=True)[0]) * (mask == 1).float()
        return masked_input / torch.clip(torch.sum(masked_input, dim=1, keepdim=True), 1e-6)
