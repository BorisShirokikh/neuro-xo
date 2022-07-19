import argparse

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger

from alphaxo_.train import AlphaXOZero


def train(model, experiment_name, use_gpu, logdir='./'):
    logger = TensorBoardLogger(save_dir=f'{logdir}/logs', name=experiment_name)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='mean_iou',
        dirpath=f'{logdir}/logs/{experiment_name}',
        filename='{epoch:02d}-{mean_iou:.3f}',
        mode='max')

    trainer = pl.Trainer(
        max_epochs=100,
        gpus=1 if use_gpu else None,
        benchmark=True,
        check_val_every_n_epoch=1,
        logger=logger,
        callbacks=[checkpoint_callback])

    trainer.fit(model)

    torch.cuda.synchronize()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', required=True, type=str)

    args = parser.parse_args()

    model = AlphaXOZero(
        data_path='path/to/data',
        optimizer=...,
        scheduler=...
    )

    train(model, args.experiment_name, use_gpu=False)
