import pytorch_lightning as pl

import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, n_channels):
        super().__init__()

        self.conv_path = nn.Sequential(
            nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_channels),
        )

        self.out_path = nn.ReLU()

    def forward(self, x):
        return self.out_path(x + self.conv_path(x))


class CNN(pl.LightningModule):
    def __init__(self, n_chans_hidden=256, n_blocks=19, board_size=10):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(1, n_chans_hidden, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_chans_hidden, n_chans_hidden),
            nn.ReLU(),
        )
        self.residual_tower = nn.Sequential(
            *[ResBlock(n_chans_hidden) for _ in range(n_blocks)]
        )

        self.backbone = nn.Sequential(
            self.conv_block,
            self.residual_tower
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(n_chans_hidden, 2, kernel_size=1, stride=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=2 * (board_size ** 2), out_features=board_size ** 2 + 1)  # +1 for the "resign" action
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(n_chans_hidden, 1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.backbone(x)

        return self.policy_head(x), self.value_head(x)
