import os
from pathlib import Path
import argparse

from torch.utils.tensorboard import SummaryWriter

from dpipe.io import choose_existing
from dpipe.torch import save_model_state, load_model_state

from alphaxo.torch.module.policy_net import PolicyNetworkQ6
from alphaxo.train_algorithms.q_learning import PolicyPlayer, train_q_learning
from alphaxo.field import Field


if __name__ == '__main__':

    Q_EXP_PATH = choose_existing(
        Path('/nmnt/x4-hdd/experiments/rl/q_6x6/'),
    )

    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_name', required=True, type=str)
    parser.add_argument('--n_step_q', required=True, type=int)

    parser.add_argument('--lr', required=False, type=float, default=3e-4)

    parser.add_argument('--duel_name', required=False, type=str, default=None)
    parser.add_argument('--preload_path', required=False, type=str, default=None)

    args = parser.parse_known_args()[0]

    exp_path = Q_EXP_PATH / args.exp_name
    os.makedirs(exp_path, exist_ok=True)

    duel_path = None if args.duel_name is None else Q_EXP_PATH / args.duel_name

    log_dir = exp_path / 'logs'
    log_dir.mkdir(exist_ok=True)

    logger = SummaryWriter(log_dir=log_dir)

    n_step_q = args.n_step_q

    # hyperparameters:
    n = 6
    kernel_len = 5
    lr = args.lr
    device = 'cuda'

    field = Field(n=n, kernel_len=kernel_len, device=device, check_device=device)

    cnn_features = (128, 64)
    model = PolicyNetworkQ6(n=n, structure=cnn_features)

    preload_path = args.preload_path
    if preload_path is not None:
        load_model_state(model, preload_path)

    n_episodes = 600000
    eps_init = 0.5
    ep2eps = {int(1e5): 0.4, int(2e5): 0.3, int(3e5): 0.3, int(4e5): 0.2, int(5e5): 0.1, }

    player = PolicyPlayer(model=model, field=field, eps=eps_init, device=device)

    # TRAIN:
    train_q_learning(player, logger, exp_path, n_episodes, augm=True, n_step_q=n_step_q,
                     ep2eps=ep2eps, lr=lr,
                     episodes_per_epoch=10000, n_duels=100, episodes_per_model_save=10000,
                     duel_path=duel_path)
    # SAVE:
    save_model_state(player.model, exp_path / 'model.pth')
