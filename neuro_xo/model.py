import pytorch_lightning as pl
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np

from .dataset import DatasetXO
from .torch import ResBlock2d


class AlphaXOZero(pl.LightningModule):
    def __init__(
            self,
            batch_size,
            num_workers,
            optimizer,
            scheduler,
            *,
            board_size=10,
            hid_features=256, n_blocks=19
    ):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(1, hid_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hid_features, hid_features),
            nn.ReLU(),

            *[ResBlock2d(hid_features) for _ in range(n_blocks)]
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(hid_features, 2, kernel_size=1, stride=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * board_size ** 2, board_size ** 2),
            nn.Softmax(dim=-1)
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(hid_features, 1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(board_size ** 2, hid_features * board_size ** 2),
            nn.ReLU(),
            nn.Linear(hid_features * board_size ** 2, 1),
            nn.Tanh()
        )

        self.train_dataset = DatasetXO()

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.optimizer = optimizer
        self.scheduler = scheduler

        self.board_size = board_size

    def forward(self, x):
        x = self.backbone(x)

        return self.policy_head(x), self.value_head(x)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, num_workers=self.num_workers, batch_size=self.batch_size, shuffle=True)

    def configure_optimizers(self):
        return [self.optimizer], [self.scheduler]

    def on_epoch_start(self):
        # FIXME: random training dataset initialization
        for _ in range(100):
            self.train_dataset.s.append(np.random.rand(self.board_size, self.board_size))
            self.train_dataset.pi.append(np.random.rand(self.board_size ** 2))
            self.train_dataset.z.append(np.random.randint(2))

    def training_step(self, batch, batch_idx):
        s, pi, z = batch
        p, v = self(s)

        # loss = loss(p, v, pi, z)
        loss = np.nan

        return loss
