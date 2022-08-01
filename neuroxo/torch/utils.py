import torch


def get_available_moves(features):
    return 1 - torch.sum(features[:, :2, ...], dim=1).unsqueeze(1)
