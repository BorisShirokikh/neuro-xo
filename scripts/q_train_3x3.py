import os
from pathlib import Path
import argparse

from torch.utils.tensorboard import SummaryWriter

from dpipe.io import choose_existing
from dpipe.torch import save_model_state, load_model_state

from ttt_lib.q_learning import PolicyNetworkQ, PolicyPlayer, train_q_learning
from ttt_lib.field import Field


if __name__ == '__main__':

    Q_EXP_PATH = choose_existing(
        Path('/nmnt/x2-hdd/experiments/rl/q_3x3'),
        Path('/experiments/borish/RL/ttt/mces')
    )

    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_name', required=True, type=str)
    parser.add_argument('--n_step_q', required=True, type=int, choices=(1, 2))

    parser.add_argument('--field_size', required=False, type=int, default=3)
    parser.add_argument('--win_len', required=False, type=int, default=3)
    parser.add_argument('--device', required=False, type=str, default='cuda')
    parser.add_argument('--n_episodes', required=False, type=int, default=1000)
    parser.add_argument('--eps', required=False, type=float, default=0.0)
    parser.add_argument('--lr', required=False, type=float, default=0.01)
    parser.add_argument('--augm', required=False, action='store_true', default=False)

    parser.add_argument('--preload_path', required=False, type=str, default=None)

    args = parser.parse_known_args()[0]

    exp_path = Q_EXP_PATH / args.exp_name
    os.makedirs(exp_path, exist_ok=True)

    log_dir = exp_path / 'logs'
    log_dir.mkdir()

    logger = SummaryWriter(log_dir=log_dir)

    n_step_q = args.n_step_q

    # hyperparameters:
    n = args.field_size
    kernel_len = args.win_len
    device = args.device
    n_episodes = args.n_episodes
    eps = args.eps
    lr = args.lr
    augm = args.augm

    field = Field(n=n, kernel_len=kernel_len, device=device, check_device=device)

    cnn_features = (128, 64)
    model = PolicyNetworkQ(n=n, structure=cnn_features)
    preload_path = args.preload_path
    if preload_path is not None:
        load_model_state(model, Path(preload_path) / 'model.pth')

    player = PolicyPlayer(model=model, field=field, eps=eps, device=device)

    # TRAIN:
    loss = train_q_learning(player, n_episodes=n_episodes, n_step_q=n_step_q, eps=eps, lr=lr, augm=augm, logger=logger)
    # SAVE:
    # save(sa2g, exp_path / 'sa2g.json')
    save_model_state(player.model, exp_path / 'model.pth')
